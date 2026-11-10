#!/usr/bin/env python3
"""Targeted AD MMSE multi-IV sensitivity using the FROZEN MAP/ADNI GReX.

Reuses SAME fit_single_iv_arx and fixed-effect Cochran Q/I2/LOO formulas as
scripts_stage2/08_candidate_robustness.py. No re-selection of MMSE hits.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))

def yes(s):return s.astype('string').fillna('false').str.strip().str.lower().isin(['true','1','yes'])
def to_num(s):return pd.to_numeric(s,errors='coerce')

def load_multiiv_summary():
    """Use the existing PD fixed-effect/Q/I2/LOO formulas, not their visit processor."""
    script=REPO/'scripts_stage2/08_candidate_robustness.py'
    spec=importlib.util.spec_from_file_location('frozen_stage2_multiiv',script)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod

_AD_FINALIZER=None

def _init_ad_worker(covpath,pcpath,manifestpath,componentdir):
    """Use the AD original *exact* visit and covariate initialization."""
    global _AD_FINALIZER
    script=REPO/'scripts_stage2/20_finalize_ad_mmse.py'
    spec=importlib.util.spec_from_file_location('frozen_ad_mmse_finalize',script)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    mod._init_rob(str(covpath),str(pcpath),str(manifestpath),str(componentdir),str(REPO))
    _AD_FINALIZER=mod

def _fit_ad_existing(t):
    """Targeted candidate-gene additional IV only; replicate original AD _rob_task fit.

    Original scripts_stage2/20_finalize_ad_mmse.py:_rob_task is the source
    of truth for sample matching, PCs, eduyrs, followup lags, and AR settings.
    """
    from gengps.amp_pd.single_iv_causal import fit_single_iv_arx
    gene,snp,aid,pf,cohort,lag,outcome,tv,min_part,age_cubic,support,pcs=t
    rec={'gene':gene,'lead_snp':snp,'artifact_id':aid,'cohort':cohort,'lag_order':int(lag)}
    try:
        mod=_AD_FINALIZER
        if mod is None:raise RuntimeError('AD worker was not initialized')
        pred=pd.read_parquet(mod._PREDS/f'{aid}.parquet')
        must=['cohort','participant_id','time','grex_single_iv','iv_dosage']
        if not set(must).issubset(pred):raise ValueError(f'Frozen component lacks {set(must)-set(pred)}')
        pred=pred.loc[pred.cohort.astype(str).eq(cohort),must].copy()
        cov=mod._COV.loc[mod._COV.cohort.eq(cohort)].copy()
        pred['participant_id']=pred.participant_id.astype(str)
        for df in (pred,cov):df['time_key']=mod.as_num(df.time).round(10)
        if pred.duplicated(['participant_id','time_key']).any():raise ValueError('duplicate component visit keys')
        if cov.duplicated(['participant_id','time_key']).any():raise ValueError('duplicate phenotype visit keys')
        d=cov.merge(pred[['participant_id','time_key','grex_single_iv','iv_dosage']],
                    on=['participant_id','time_key'],how='inner',validate='one_to_one')
        pc_cols=[c for c in mod._PCS[cohort] if c in d.columns]
        if len(pc_cols)!=len(mod._PCS[cohort]):raise ValueError('missing selected genotype PC column')
        baseline_cov=pc_cols+(['eduyrs'] if 'eduyrs' in d else [])
        r=fit_single_iv_arx(d,'mmse','grex_single_iv',int(lag),baseline_covariates=baseline_cov,
                            time_varying_covariates=[],auto_clinical_covariates=True,
                            min_participants=int(min_part),adjust_current_age_cubic=True,
                            ar_time_powers=(0,1,2,3),compute_ar=True,compute_cluster_details=True)
        rec.update(ok=r.ok,reason=r.reason,n_obs=r.n_obs,n_participants=r.n_participants,
                   pvalue=r.pvalue,joint_F=r.joint_F,coef_sum=r.coef_sum,
                   coef_sum_se_cluster=r.coef_sum_se_cluster,
                   coef_sum_p_cluster=r.coef_sum_p_cluster,
                   coef_lag1=r.coef_lag1,ar_pvalue=r.ar_pvalue,ar_F=r.ar_F)
        for j,b in enumerate(r.exposure_coefs,1):rec[f'coef_x_lag{j}']=b
        return rec,None
    except Exception as exc:
        rec.update(ok=False,reason='worker_exception',coef_sum=np.nan,coef_sum_se_cluster=np.nan)
        return rec,{'gene':gene,'lead_snp':snp,'artifact_id':aid,'cohort':cohort,
                    'lag_order':lag,'error':repr(exc),'traceback':traceback.format_exc()}

def run_ad_fits(tasks,covfile,pcfile,manifestfile,componentdir,njobs):
    args=(covfile,pcfile,manifestfile,componentdir)
    rows=[];errors=[]
    if njobs<=1:
        _init_ad_worker(*args)
        source=map(_fit_ad_existing,tasks)
        for row,err in source:
            rows.append(row)
            if err:errors.append(err)
    else:
        with ProcessPoolExecutor(max_workers=njobs,initializer=_init_ad_worker,initargs=args) as pool:
            for row,err in pool.map(_fit_ad_existing,tasks,chunksize=6):
                rows.append(row)
                if err:errors.append(err)
    return pd.DataFrame(rows),errors

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def build_tasks(root,final,alliv,component_dir):
    required={'gene','lead_snp','artifact_id','selected_lag_order','MAP_q_bh_global','MAP_pvalue'}
    if not required.issubset(final):raise ValueError(f'AD final missing {required-set(final)}')
    if final[['gene','lead_snp','artifact_id']].duplicated().any():raise ValueError('Duplicate selected IVs')
    final=final.copy()
    for col in ('MAP_q_bh_global','MAP_pvalue','MAP_bic'):
        if col in final:final[col]=to_num(final[col])
    if 'MAP_bic' not in final:final['MAP_bic']=np.nan
    ref=(final.sort_values(['gene','MAP_q_bh_global','MAP_pvalue','MAP_bic','lead_snp'],kind='mergesort')
         .drop_duplicates('gene')[['gene','lead_snp','artifact_id','selected_lag_order']]
         .rename(columns={'lead_snp':'reference_lead_snp','artifact_id':'reference_artifact_id','selected_lag_order':'reference_lag'}))
    if ref['reference_lag'].isna().any():raise ValueError('A final candidate gene has no frozen selected lag')
    for i in ref.reference_lag:
        if int(i)!=i or not 1<=int(i)<=4:raise ValueError(f'Invalid selected lag {i}')
    alliv=alliv.copy()
    if alliv[['gene','lead_snp']].duplicated().any():raise ValueError('AD matching IV list contains multiple artifacts per gene+SNP')
    candidates=alliv.merge(ref,on='gene',how='inner',validate='many_to_one')
    tasks=[];missing=[]
    for r in candidates.itertuples(index=False):
        file=component_dir/(str(r.artifact_id)+'.parquet')
        if not file.is_file():
            missing.append({'gene':r.gene,'lead_snp':r.lead_snp,'artifact_id':r.artifact_id,
                            'status':'missing_frozen_component','component':str(file)})
            continue
        for co in ('MAP','ADNI'):
            tasks.append((str(r.gene),str(r.lead_snp),str(r.artifact_id),str(file),co,
                          int(r.reference_lag),'mmse',[],50,True,False,True))
    return ref,candidates,tasks,missing

def prepare(root,final_path,stage1_path,ad_candidates,component_dir):
    final=pd.read_csv(final_path,low_memory=False,dtype={'gene':str,'lead_snp':str,'artifact_id':str})
    if 'PRIMARY_replicated_same_lag_direction' in final:
        final=final.loc[yes(final.PRIMARY_replicated_same_lag_direction)].copy()
    if final.empty:raise ValueError('Zero AD MMSE significant/replicated candidates')
    st=pd.read_csv(stage1_path,low_memory=False,dtype={'gene':str,'lead_snp':str,'artifact_id':str})
    ad=pd.read_csv(ad_candidates,low_memory=False,dtype={'gene':str,'lead_snp':str,'artifact_id':str})
    required={'gene','lead_snp','artifact_id'}
    for obj,name in ((st,'Stage-I'),(ad,'AD shared')):
        if not required.issubset(obj):raise ValueError(f'{name} missing required {required-set(obj)}')
    if 'iv_primary_eligible' in st:st=st.loc[yes(st.iv_primary_eligible)].copy()
    st=st[list(required)].drop_duplicates()
    ad=ad[list(required)].drop_duplicates()
    if st.duplicated(['gene','lead_snp']).any() or ad.duplicated(['gene','lead_snp']).any():raise ValueError('Multiple artifacts per gene+SNP')
    alliv=ad.merge(st,on=['gene','lead_snp','artifact_id'],how='inner',validate='one_to_one')
    ref,candidates,tasks,missing=build_tasks(root,final,alliv,component_dir)
    expected=final[['gene','lead_snp','artifact_id']].merge(ad,on=['gene','lead_snp','artifact_id'],how='left',indicator=True)
    if not expected._merge.eq('both').all():
        raise ValueError('Final replicated MMSE IV is not in the existing frozen AD shared candidates; refusing inconsistent sensitivity fits')
    return final,ref,candidates,tasks,missing

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--final-file',default='')
    ap.add_argument('--out-dir',default='')
    ap.add_argument('--n-jobs',type=int,default=8)
    ap.add_argument('--min-participants',type=int,default=None)
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args();root=Path(a.root)
    base=root/'single_iv_causal/ad_mmse';std=base/'standard_final'
    out=Path(a.out_dir) if a.out_dir else std/'posthoc_evidence'
    out.mkdir(parents=True,exist_ok=True)
    final_path=Path(a.final_file) if a.final_file else std/'AD_MMSE_FINAL_evidence_graded.csv'
    stage1=root/'single_iv_causal/iv_validation/single_iv_primary_eligible.csv'
    ad_candidates=base/'prep/AD_MMSE_shared_frozen_candidates.csv'
    component_dir=base/'single_iv_genetic_component'
    config_path=base/'temporal_causal_agecubic_full_followup_fast/requested_run_config.json'
    if not config_path.is_file():raise FileNotFoundError(f'Missing original Stage-II run configuration {config_path}')
    cfg=json.loads(config_path.read_text())
    for key,val in [('discovery_cohort','MAP'),('outcome_col','mmse'),('adjust_current_age_cubic',True),('use_genotype_pcs',True),('require_stage1_followup_support',False)]:
        if cfg.get(key)!=val:raise ValueError(f'Original AD MMSE config {key}={cfg.get(key)} not expected {val}')
    n=int(cfg.get('min_participants',50))
    if a.min_participants is not None and a.min_participants!=n:raise ValueError(f'Requested min participants {a.min_participants} differs from original {n}')
    final,ref,candidates,tasks,missing=prepare(root,final_path,stage1,ad_candidates,component_dir)
    tasks=[t[:8]+(n,)+t[9:] for t in tasks]  # identical frozen min-participants threshold
    ref.to_csv(out/'gene_reference_lag.csv',index=False)
    candidates.to_csv(out/'all_AD_shared_Stage1_eligible_IVs_for_candidate_genes.csv',index=False)
    pd.DataFrame(missing,columns=['gene','lead_snp','artifact_id','status','component']).to_csv(out/'missing_component_IVs.csv',index=False)
    planned={'analysis':'ad_mmse','source_final':str(final_path),'source_config':str(config_path),
             'n_final_pairs':len(final),'n_final_genes':final.gene.nunique(),
             'n_stage1_AD_shared_IVs_for_candidate_genes':len(candidates),
             'n_missing_components':len(missing),'n_planned_cohort_specific_fits':len(tasks),
             'original_min_participants':n,'original_age_cubic':True,'original_selected_genotype_pcs':True,
             'models_reused':'Exact AD sample matching/PCs/eduyrs from scripts_stage2/20_finalize_ad_mmse.py; fit_single_iv_arx; fixed_effect, summarize_multi_iv from scripts_stage2/08_candidate_robustness.py',
             'scope':'candidate genes only, all existing eligible independently pruned IVs in AD matched genotype universe',
             'limitation':'available frozen AD shared IVs only; no new genotyping or full Stage-II screening'}
    (out/'multiiv_preparation_manifest.json').write_text(json.dumps(planned,indent=2,default=str))
    print('[PREPARE]',json.dumps(planned,indent=2,default=str),flush=True)
    if a.dry_run:return
    if not tasks:
        pd.DataFrame(columns=['gene','lead_snp','artifact_id','cohort','lag_order','ok','reason','coef_sum','coef_sum_se_cluster']).to_csv(out/'all_AD_shared_IVs_commonlag_details.csv',index=False)
        pd.DataFrame(columns=['gene','cohort','multiIV_n','multiIV_p']).to_csv(out/'multi_iv_gene_consistency_summary.csv',index=False)
        pd.DataFrame(columns=['gene','cohort','left_out_snp']).to_csv(out/'multi_iv_leave_one_out.csv',index=False)
        return
    mod=load_multiiv_summary()  # only existing multiIV Q/I2/LOO statistics
    cov_file=base/'prep/AD_MMSE_phenotype_cov.csv'
    pc_file=base/'genotype_pcs/single_iv_genotype_pcs.csv'
    pc_manifest=base/'genotype_pcs/pc_selection.json'
    details,errors=run_ad_fits(tasks,str(cov_file),str(pc_file),str(pc_manifest),
                               str(component_dir),a.n_jobs)
    pd.DataFrame(errors,columns=['gene','lead_snp','artifact_id','cohort','lag_order','error','traceback']).to_csv(out/'multi_iv_fit_errors.csv',index=False)
    if details.empty:
        details=pd.DataFrame(columns=['gene','lead_snp','artifact_id','cohort','lag_order','ok','reason','coef_sum','coef_sum_se_cluster'])
    details=details.merge(ref,on='gene',how='left',validate='many_to_one')
    details.to_csv(out/'all_AD_shared_IVs_commonlag_details.csv',index=False)
    all_summary=[];all_loo=[]
    for co in ('MAP','ADNI'):
        z=details.loc[details.cohort.eq(co)]
        if z.empty:continue
        summary,loo=mod.summarize_multi_iv(z)
        if summary.empty:continue
        summary.insert(1,'cohort',co)
        all_summary.append(summary)
        if not loo.empty:
            loo.insert(1,'cohort',co);all_loo.append(loo)
    summ=(pd.concat(all_summary,ignore_index=True) if all_summary else pd.DataFrame(columns=['gene','cohort','multiIV_n','multiIV_p']))
    if not summ.empty:
        from gengps.amp_pd.single_iv_causal import bh_adjust
        for co in ('MAP','ADNI'):
            ix=summ.cohort.eq(co)
            for pcol,qcol in [('multiIV_p','multiIV_q_BH'),('multiIV_Q_p','multiIV_Q_q_BH')]:
                if pcol in summ:
                    summ.loc[ix,qcol]=bh_adjust(to_num(summ.loc[ix,pcol]))
        summ=summ.merge(ref,on='gene',how='left',validate='many_to_one')
    summ.to_csv(out/'multi_iv_gene_consistency_summary.csv',index=False)
    (pd.concat(all_loo,ignore_index=True) if all_loo else pd.DataFrame(columns=['gene','cohort','left_out_snp','n_remaining','beta','se','p'])).to_csv(out/'multi_iv_leave_one_out.csv',index=False)
    planned.update({'n_cohort_models_returned':len(details),'n_model_errors':len(errors),
                    'n_multi_iv_genes_MAP':int(summ.loc[summ.cohort.eq('MAP'),'multiIV_p'].notna().sum()) if 'multiIV_p' in summ else 0,
                    'n_multi_iv_genes_ADNI':int(summ.loc[summ.cohort.eq('ADNI'),'multiIV_p'].notna().sum()) if 'multiIV_p' in summ else 0})
    (out/'multiiv_run_manifest.json').write_text(json.dumps(planned,indent=2,default=str))
    print('[DONE]',json.dumps(planned,indent=2,default=str))
if __name__=='__main__':main()
