#!/usr/bin/env python3
"""AD MMSE finalization FROM SAVED MAP + ADNI FAST outputs (without repeating discovery).

MAP discovery: BH-FDR on all *successful* (IV, gene, lag) tests (same family as primary).
ADNI primary: same discovery-FDR-significant lag, nominal p<.05, nonzero matching coef_sum sign.
Robustness: full-detail AR-type reduced-form cluster Wald/F, NOT classical Anderson-Rubin.
All outputs are descriptive genetic-temporal associations; causal identification is unproven.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd

KEY = ['gene','lead_snp','artifact_id']
LK = KEY+['lag_order']
_COV = None
_PCS = None
_PREDS = None

def as_num(s): return pd.to_numeric(s, errors='coerce')
def finite(s): return np.isfinite(as_num(s).to_numpy(dtype=float))
def valid_bool(s): return s.astype('string').fillna('false').str.lower().isin(['true','1','yes'])
def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(2**20),b''):h.update(part)
    return h.hexdigest()

def read_csv(path):
    if not path.exists():raise FileNotFoundError(f'Missing completed Stage2 result: {path}')
    return pd.read_csv(path,low_memory=False)

def check_rows(df, name, cohort):
    required=set(LK+['cohort','ok','pvalue','bic','coef_lag1','coef_sum','n_participants'])
    miss=required-set(df.columns)
    if miss:raise ValueError(f'{name} missing required columns: {sorted(miss)}')
    df=df.copy()
    for k in KEY:df[k]=df[k].astype(str)
    df['lag_order']=as_num(df['lag_order']).astype('Int64')
    df=df[df['cohort'].astype(str).eq(cohort)].copy()
    if df.empty:raise ValueError(f'{name} has no {cohort} rows; wrong output or failed run')
    if df[LK].duplicated().any():raise ValueError(f'{name} has duplicate IV-gene-lag rows')
    df['ok_flag']=valid_bool(df['ok'])
    return df

def bh(p):
    from gengps.amp_pd.single_iv_causal import bh_adjust
    return bh_adjust(as_num(p))

def finalize_from_existing(dis,rep,cand):
    """Return annotated discovery, replication, all MAP-FDR pairs, primary replicated pairs."""
    cand=cand.copy()
    for k in KEY:
        if k not in cand:raise ValueError(f'Candidate file missing {k}')
        cand[k]=cand[k].astype(str)
    if cand[KEY].duplicated().any():raise ValueError('Candidate file has duplicate gene-SNP-artifact IDs')
    if not set(map(tuple,dis[KEY].to_numpy())).issubset(set(map(tuple,cand[KEY].to_numpy()))):
        raise ValueError('Discovery has IVs not in the AD-MMSE shared candidate file')
    dis=dis.copy();rep=rep.copy()
    dis['q_bh_global_recomputed']=bh(dis.pvalue.where(dis.ok_flag))
    if 'q_bh_global' in dis.columns:
        orig=as_num(dis.q_bh_global);new=as_num(dis.q_bh_global_recomputed)
        mask=orig.notna()&new.notna()
        if mask.any() and not np.allclose(orig[mask],new[mask],atol=1e-9,rtol=1e-8):
            raise ValueError('Existing discovery FDR differs from independent BH recalculation; do not finalize until audited')
    dis['q_bh_global']=dis['q_bh_global_recomputed']
    dis['MAP_discovery_FDR']=dis.ok_flag & as_num(dis.q_bh_global).lt(.05)
    sig=dis.loc[dis.MAP_discovery_FDR].copy()
    if sig.empty:raise ValueError('Zero MAP-FDR discoveries: generate an empty report manually; refusing misleading FINAL')
    sigkeys=sig[KEY].drop_duplicates()
    r=rep.merge(sig[LK+['pvalue','q_bh_global','coef_lag1','coef_sum','bic']].rename(columns={
        'pvalue':'MAP_pvalue','q_bh_global':'MAP_q_bh_global','coef_lag1':'MAP_coef_lag1','coef_sum':'MAP_coef_sum','bic':'MAP_bic',
    }),on=LK,how='left',validate='one_to_one')
    r['same_discovery_FDR_lag']=r.MAP_q_bh_global.notna()
    a=as_num(r.MAP_coef_lag1);b=as_num(r.coef_lag1)
    r['lag1_direction_concordant']=a.notna()&b.notna()&a.ne(0)&b.ne(0)&(np.sign(a)==np.sign(b))
    c=as_num(r.MAP_coef_sum);d=as_num(r.coef_sum)
    r['coef_sum_direction_concordant']=c.notna()&d.notna()&c.ne(0)&d.ne(0)&(np.sign(c)==np.sign(d))
    r['ADNI_nominal_p05']=r.ok_flag & as_num(r.pvalue).lt(.05)
    r['same_lag_nominal_only']=r.same_discovery_FDR_lag&r.ADNI_nominal_p05
    r['PRIMARY_same_lag_direction']=r.same_lag_nominal_only&r.coef_sum_direction_concordant
    # Diagnostics only, not selection: nominal p at ANY lag, even if not MAP FDR.
    r['any_lag_nominal_sensitivity']=r.ADNI_nominal_p05
    # All selected rows must correspond to an actual MAP BH-significant lag.
    def best(z):
        return z.sort_values(['bic','pvalue','lag_order'],kind='mergesort',na_position='last').iloc[0]
    records=[]
    for identity,ds in sig.groupby(KEY,sort=False):
        ds=ds.sort_values(['bic','pvalue','lag_order'],kind='mergesort',na_position='last')
        rp=r.loc[(r[KEY]==pd.Series(identity,index=KEY)).all(axis=1)].copy()
        matched=rp.loc[rp.PRIMARY_same_lag_direction]
        nominal=rp.loc[rp.same_lag_nominal_only]
        if not matched.empty:
            rr=best(matched);dl=int(rr.lag_order);selected_by='minimum_ADNI_BIC_among_MAP_FDR_same_lag_P05_coef_sum_concordant'
        else:
            dl=int(ds.iloc[0].lag_order);selected_by='minimum_MAP_BIC_among_MAP_FDR_lags_no_coef_sum_concordant_replication'
        dr=ds.loc[ds.lag_order.eq(dl)].iloc[0]
        # Always take ADNI at chosen lag for transparent independent reporting.
        adj=rp.loc[rp.lag_order.eq(dl)]
        ar=adj.iloc[0] if not adj.empty else None
        row={k:v for k,v in zip(KEY,identity)}
        for c in ['lag_order','pvalue','q_bh_global','joint_F','joint_df1','joint_df2','bic','partial_r2','coef_lag1','coef_sum','n_obs','n_participants','ok']:
            row['MAP_'+c]=dr.get(c,np.nan)
            row['ADNI_'+c]=ar.get(c,np.nan) if ar is not None else np.nan
        row['selected_lag_order']=dl;row['selected_by']=selected_by
        row['n_MAP_FDR_lags']=len(ds)
        row['MAP_FDR_lags']=';'.join(str(x) for x in sorted(ds.lag_order.dropna().astype(int).unique()))
        row['replicated_same_lag_nominal_only']=not nominal.empty
        row['PRIMARY_replicated_same_lag_direction']=not matched.empty
        row['ADNI_any_lag_nominal_sensitivity']=bool(rp.any_lag_nominal_sensitivity.any())
        row['ADNI_any_same_lag_opposite_direction']=bool((nominal.lag1_direction_concordant==False).any()) if len(nominal) else False
        row['selected_lag1_direction_concordant']=bool(ar.lag1_direction_concordant) if ar is not None else False
        row['selected_coef_sum_direction_concordant']=bool(ar.coef_sum_direction_concordant) if ar is not None else False
        row['n_ADNI_same_lag_direction_hits']=int(len(matched))
        records.append(row)
    master=pd.DataFrame(records).merge(cand,on=KEY,how='left',validate='one_to_one')
    master['is_final_primary']=master.PRIMARY_replicated_same_lag_direction.astype(bool)
    primary=master.loc[master.is_final_primary].copy()
    # For primary results, selecting a MAP-FDR-significant ADNI hit at the SAME lag is mandatory.
    assert all(primary.MAP_q_bh_global.lt(.05)&as_num(primary.ADNI_pvalue).lt(.05)&primary.selected_coef_sum_direction_concordant)
    return dis,r,master,primary

def _init_rob(covpath,pcpath,manifestpath,componentdir,core):
    global _COV,_PCS,_PREDS
    sys.path.insert(0,str(core))
    _COV=pd.read_csv(covpath,low_memory=False)
    for col in ['participant_id','cohort']:_COV[col]=_COV[col].astype(str)
    _PCS={};_PREDS=Path(componentdir)
    pc=pd.read_csv(pcpath,low_memory=False,dtype={'participant_id':str,'cohort':str})
    manifest=json.loads(Path(manifestpath).read_text())
    for cohort in ['MAP','ADNI']:
        selected=manifest.get('cohorts',{}).get(cohort,{}).get('selected_columns',[])
        if not selected:raise ValueError(f'{cohort} genotype PC selection is empty')
        _PCS[cohort]=selected
    active=sorted(set(sum(_PCS.values(),[])))
    missing=set(active)-set(pc.columns)
    if missing:raise ValueError(f'PC matrix missing columns {sorted(missing)}')
    _COV=_COV.merge(pc[['cohort','participant_id']+active].drop_duplicates(['cohort','participant_id']),
                    on=['cohort','participant_id'],how='left',validate='many_to_one')

def _rob_task(t):
    from gengps.amp_pd.single_iv_causal import fit_single_iv_arx
    gene,snp,aid,cohort,lag,original_p,original_bic,original_lag1,original_sum,min_part=t
    record=dict(gene=gene,lead_snp=snp,artifact_id=aid,cohort=cohort,lag_order=int(lag),
                original_pvalue=original_p,original_bic=original_bic,
                original_coef_lag1=original_lag1,original_coef_sum=original_sum)
    try:
        path=_PREDS/f'{aid}.parquet'
        if not path.is_file():raise FileNotFoundError(path)
        pred=pd.read_parquet(path)
        must=['cohort','participant_id','time','grex_single_iv','iv_dosage']
        if not set(must).issubset(pred):raise ValueError(f'Predicted component lacks {set(must)-set(pred)}')
        pred=pred.loc[pred.cohort.astype(str).eq(cohort),must].copy()
        cov=_COV.loc[_COV.cohort.eq(cohort)].copy()
        pred['participant_id']=pred.participant_id.astype(str)
        for df in [pred,cov]:df['time_key']=as_num(df['time']).round(10)
        if pred.duplicated(['participant_id','time_key']).any():raise ValueError('duplicate component visit keys')
        if cov.duplicated(['participant_id','time_key']).any():raise ValueError('duplicate phenotype visit keys')
        d=cov.merge(pred[['participant_id','time_key','grex_single_iv','iv_dosage']],
                    on=['participant_id','time_key'],how='inner',validate='one_to_one')
        pc_cols=[z for z in _PCS[cohort] if z in d.columns]
        if len(pc_cols)!=len(_PCS[cohort]):raise ValueError('missing selected genotype PC column')
        base_cov=pc_cols+(['eduyrs'] if 'eduyrs' in d else [])
        fit=fit_single_iv_arx(d,'mmse','grex_single_iv',int(lag),baseline_covariates=base_cov,
            time_varying_covariates=[],auto_clinical_covariates=True,min_participants=min_part,
            adjust_current_age_cubic=True,ar_time_powers=(0,1,2,3),compute_ar=True,compute_cluster_details=True)
        record.update(fit_ok=fit.ok,fit_reason=fit.reason,n_obs=fit.n_obs,n_participants=fit.n_participants,
                      pvalue=fit.pvalue,bic=fit.bic,coef_lag1=fit.coef_lag1,coef_lag1_se_cluster=fit.coef_lag1_se_cluster,
                      coef_lag1_p_cluster=fit.coef_lag1_p_cluster,coef_sum=fit.coef_sum,
                      coef_sum_se_cluster=fit.coef_sum_se_cluster,coef_sum_p_cluster=fit.coef_sum_p_cluster,
                      RF_cluster_F=fit.ar_F,RF_cluster_pvalue=fit.ar_pvalue,RF_df1=fit.ar_df1,
                      RF_df2=fit.ar_df2,RF_rank=fit.ar_rank,RF_condition=fit.ar_condition)
        # Compare -log10(P) for tiny P values; absolute P tolerance would conceal
        # huge relative errors (e.g. 1e-20 vs 1e-12).
        if fit.pvalue>0 and original_p>0:
            pp=np.isclose(-np.log10(fit.pvalue),-np.log10(original_p),rtol=5e-6,atol=1e-5)
        else:pp=bool(fit.pvalue==original_p)
        bb=np.isclose(fit.bic,original_bic,rtol=5e-7,atol=1e-5,equal_nan=False)
        b1=np.isclose(fit.coef_lag1,original_lag1,rtol=1e-5,atol=1e-8,equal_nan=False)
        bs=np.isclose(fit.coef_sum,original_sum,rtol=1e-5,atol=1e-8,equal_nan=False)
        record['FAST_fullfit_numeric_match']=bool(pp and bb and b1 and bs)
        if not record['FAST_fullfit_numeric_match']:record['error']=f'FAST/full mismatch: p {original_p} vs {fit.pvalue}, BIC {original_bic} vs {fit.bic}, lag1 {original_lag1} vs {fit.coef_lag1}, sum {original_sum} vs {fit.coef_sum}'
    except Exception as ex:
        record.update(fit_ok=False,FAST_fullfit_numeric_match=False,error=repr(ex),traceback=traceback.format_exc())
    return record

def prepare_tasks(master):
    tasks=[]
    for x in master.itertuples(index=False):
        # Re-evaluate exactly the selected lag, never an arbitrary replication lag.
        for cohort in ['MAP','ADNI']:
            p=getattr(x,f'{cohort}_pvalue',np.nan);bic=getattr(x,f'{cohort}_bic',np.nan)
            beta=getattr(x,f'{cohort}_coef_lag1',np.nan);coefsum=getattr(x,f'{cohort}_coef_sum',np.nan)
            if not all(np.isfinite(v) for v in (p,bic,beta,coefsum)):continue
            tasks.append((str(x.gene),str(x.lead_snp),str(x.artifact_id),cohort,
                           int(x.selected_lag_order),float(p),float(bic),float(beta),float(coefsum),
                           int(master.attrs.get('min_participants',50))))
    return tasks

def fit_selected(master, args, basedir, core):
    out=basedir/'AD_MMSE_robustness_selected_lags.csv'
    manifest=basedir/'AD_MMSE_robustness_checkpoint.json'
    tasks=prepare_tasks(master)
    # Checkpoint signature includes frozen phenotype/PC/fit code and exact component file status.
    # This prevents stale full-detail results after a covariate or GReX update.
    fingerprint={str(z):sha_file(Path(z)) for z in [args.phenotype_cov,args.pc_file,args.pc_manifest,
                                                 core/'gengps'/'amp_pd'/'single_iv_causal.py']}
    for aid in sorted({t[2] for t in tasks}):
        pf=Path(args.component_dir)/f'{aid}.parquet'
        if not pf.is_file():raise FileNotFoundError(f'Selected GReX component missing: {pf}')
        st=pf.stat();fingerprint[str(pf)]=(st.st_size,st.st_mtime_ns)
    signature=hashlib.sha256(json.dumps({'tasks':tasks,'sources':fingerprint},sort_keys=True).encode()).hexdigest()
    if out.exists() and manifest.exists() and args.resume_robustness:
        meta=json.loads(manifest.read_text())
        if meta.get('signature')==signature and meta.get('complete') is True:
            cached=pd.read_csv(out,low_memory=False)
            if len(cached)==len(tasks):
                print(f'[RESUME] Reusing {len(cached)} exact-lag full-detail fits',flush=True)
                return cached
    print(f'[ROBUSTNESS] {len(tasks)} exact-lag fits (MAP + ADNI), n_jobs={args.n_jobs}',flush=True)
    conv=(str(args.phenotype_cov),str(args.pc_file),str(args.pc_manifest),
          str(args.component_dir),str(core))
    rows=[]
    if args.n_jobs==1:
        _init_rob(*conv)
        it=map(_rob_task,tasks)
        for i,row in enumerate(it,1):
            rows.append(row)
            if i%50==0 or i==len(tasks):print(f'[ROBUSTNESS] {i}/{len(tasks)}',flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs,initializer=_init_rob,initargs=conv) as executor:
            for i,row in enumerate(executor.map(_rob_task,tasks,chunksize=max(1,args.chunksize)),1):
                rows.append(row)
                if i%50==0 or i==len(tasks):print(f'[ROBUSTNESS] {i}/{len(tasks)}',flush=True)
    res=pd.DataFrame(rows);res.to_csv(out,index=False)
    manifest.write_text(json.dumps({'signature':signature,'complete':True,'n_tasks':len(tasks),
                                    'n_errors':int((~valid_bool(res.FAST_fullfit_numeric_match)).sum())},indent=2))
    return res

def add_robustness(master,rob):
    z=master.copy()
    if rob is None:
        for co in ['MAP','ADNI']:
            for col in ['RF_cluster_pvalue','RF_cluster_F','coef_lag1_p_cluster','coef_sum_p_cluster','FAST_fullfit_numeric_match']:
                z[f'{co}_{col}']=np.nan
        return z
    for co in ['MAP','ADNI']:
        cur=rob.loc[rob.cohort.eq(co)].copy()
        subset=['gene','lead_snp','artifact_id','lag_order']
        cols=['RF_cluster_pvalue','RF_cluster_F','RF_df1','RF_df2','RF_rank','coef_lag1_p_cluster',
              'coef_lag1_se_cluster','coef_sum_p_cluster','coef_sum_se_cluster','FAST_fullfit_numeric_match','fit_ok','error']
        for col in cols:
            if col not in cur.columns:cur=cur.assign(**{col:np.nan})
        addon=cur[subset+cols].rename(columns={c:f'{co}_{c}' for c in cols})
        z=z.merge(addon,left_on=KEY+['selected_lag_order'],right_on=LK,how='left',validate='one_to_one')
        z=z.drop(columns=['lag_order'])
    return z

def drop_legacy_stage1_strength_columns(df):
    """No first-stage F, F>=10, or strength classification in AD final output.

    Keep Stage2 nested-test joint_F and reduced-form RF_cluster_F: they are
    outcome/robustness statistics, NOT a first-stage instrument-strength F.
    """
    import re
    patterns = [
        r'(^|_)first_stage(_|$)',
        r'^Stage1_strength_status$',
        r'(^|_)strength_ok($|_)',
        r'(^|_)firststage(_|$)',
    ]
    gone=[c for c in df.columns if any(re.search(p,c,flags=re.I) for p in patterns)]
    return df.drop(columns=gone,errors='ignore')


def evidence(master):
    """Evidence based on cross-cohort replication and RF tests only.

    In particular NO first-stage instrument-strength F is computed or used.
    All RF p/q values are descriptive post-selection robustness statistics,
    not classical Anderson-Rubin tests and not primary significance gates.
    """
    z=drop_legacy_stage1_strength_columns(master.copy())
    pmap=as_num(z.get('MAP_RF_cluster_pvalue',pd.Series(np.nan,index=z.index)))
    padni=as_num(z.get('ADNI_RF_cluster_pvalue',pd.Series(np.nan,index=z.index)))
    mapok=(valid_bool(z['MAP_FAST_fullfit_numeric_match'])
           if 'MAP_FAST_fullfit_numeric_match' in z else pd.Series(False,index=z.index))
    adniok=(valid_bool(z['ADNI_FAST_fullfit_numeric_match'])
            if 'ADNI_FAST_fullfit_numeric_match' in z else pd.Series(False,index=z.index))
    # BH is restricted to the selected lag of every MAP FDR-discovered pair,
    # including those not replicated in ADNI. It is NOT a global first-stage
    # or original MAP IV x lag discovery p-family and is NOT a selection gate.
    tested=pmap.where(mapok & pmap.ge(0) & pmap.le(1))
    z['MAP_RF_q_BH_selected_MAP_FDR_pairs']=bh(tested)
    z['MAP_RF_nominal_p05']=mapok & pmap.lt(.05)
    z['MAP_RF_BH_selected_p05']=mapok & as_num(z.MAP_RF_q_BH_selected_MAP_FDR_pairs).lt(.05)
    z['ADNI_RF_nominal_p05']=adniok & padni.lt(.05)
    z['dual_cohort_reduced_form_support']=z.MAP_RF_nominal_p05 & z.ADNI_RF_nominal_p05
    primary=valid_bool(z.PRIMARY_replicated_same_lag_direction)
    # A: replicated + discovery RF BH + ADNI RF nominal.
    # B: replicated + both RF nominal, but discovery RF BH not met.
    # C: replicated, but no two-cohort RF statistical support or missing fit.
    # D: MAP BH discovery without ADNI same-lag direction-concordant replication.
    # None of these tiers depend on first-stage F, partial R2, or any strength filter.
    z['evidence_tier']=np.select([
        primary & z.MAP_RF_BH_selected_p05 & z.ADNI_RF_nominal_p05,
        primary & z.dual_cohort_reduced_form_support,
        primary,
    ],[
        'A_replicated_MAP_RF_BH_ADNI_RF_nominal',
        'B_replicated_dual_RF_nominal',
        'C_replicated_RF_not_dually_supported',
    ],default='D_MAP_FDR_not_ADNI_replicated')
    z['causal_identification']='NOT_ESTABLISHED_pleiotropy_and_colocalization_not_tested'
    z['RF_assessment_status']=np.select([
        mapok & adniok,
        mapok | adniok,
    ],['both_cohorts_fullfit_verified','one_cohort_fullfit_verified'],default='RF_not_fully_evaluated')
    z['RF_interpretation']='cluster-robust SNP_x_followup_reduced_form; NOT classical Anderson-Rubin'
    z['MMSE_direction_note']='coef_sum is aggregate visit-lag conditional association; positive sums associate with higher MMSE; not long-term causal effect'
    return z

def gene_aggregate(z):
    rows=[]
    for gene,sub in z.groupby('gene',sort=True):
        rep=sub.loc[sub.PRIMARY_replicated_same_lag_direction]
        signs=np.sign(as_num(rep.MAP_coef_lag1).to_numpy(float))
        rows.append({'gene':gene,'n_MAP_FDR_IV_gene_pairs':len(sub),'n_ADNI_primary_replicated_IV_gene_pairs':len(rep),
                     'n_independent_replicated_IVs':rep.lead_snp.nunique(),
                     'n_A_tier_IVs':int(rep.evidence_tier.str.startswith('A_').sum()),
                     'n_B_tier_IVs':int(rep.evidence_tier.str.startswith('B_').sum()),
                     'n_C_tier_IVs':int(rep.evidence_tier.str.startswith('C_').sum()),
                     'replicated_lags':';'.join(map(str,sorted(rep.selected_lag_order.dropna().astype(int).unique()))),
                     'MAP_lag1_direction_agreement_across_replicated_IVs':
                       bool(np.all(signs==signs[0])) if len(signs)>=2 else np.nan,
                     'multiple_IVs_formal_MR_or_heterogeneity':'NOT_TESTED; no independent covariance/coloc test'})
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    p.add_argument('--analysis-name',default='ad_mmse')
    p.add_argument('--source-dir',default='');p.add_argument('--out-dir',default='')
    p.add_argument('--candidate-file',default='');p.add_argument('--phenotype-cov',default='')
    p.add_argument('--pc-file',default='');p.add_argument('--pc-manifest',default='')
    p.add_argument('--component-dir',default='')
    p.add_argument('--n-jobs',type=int,default=12);p.add_argument('--chunksize',type=int,default=6)
    p.add_argument('--resume-robustness',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--no-robustness',action='store_true',help='Only reconstruct selection, mark RF as not evaluated')
    args=p.parse_args();core=Path(__file__).resolve().parents[1];sys.path.insert(0,str(core))
    base=Path(args.root)/'single_iv_causal'/args.analysis_name
    src=Path(args.source_dir) if args.source_dir else base/'temporal_causal_agecubic_full_followup_fast'
    out=Path(args.out_dir) if args.out_dir else base/'standard_final'
    args.candidate_file=args.candidate_file or str(base/'prep'/'AD_MMSE_shared_frozen_candidates.csv')
    args.phenotype_cov=args.phenotype_cov or str(base/'prep'/'AD_MMSE_phenotype_cov.csv')
    args.pc_file=args.pc_file or str(base/'genotype_pcs'/'single_iv_genotype_pcs.csv')
    args.pc_manifest=args.pc_manifest or str(base/'genotype_pcs'/'pc_selection.json')
    args.component_dir=args.component_dir or str(base/'single_iv_genetic_component')
    out.mkdir(parents=True,exist_ok=True)
    discfile=src/'discovery_all_IV_gene_lag_models.csv.gz'
    repfile=src/'replication_ADNI_all_lags_for_discovery_significant_IV_gene_pairs.csv.gz'
    candfile=Path(args.candidate_file)
    # Fail fast before expensive fits and verify the finished run describes MAP/ADNI full follow-up.
    conf=src/'requested_run_config.json'
    if not conf.is_file():raise FileNotFoundError(f'No completed discovery configuration: {conf}')
    config=json.loads(conf.read_text())
    if config.get('discovery_cohort')!='MAP' or config.get('outcome_col')!='mmse':
        raise ValueError('Source is not MAP MMSE discovery')
    if config.get('require_stage1_followup_support') is not False:
        raise ValueError('Source was time-support filtered; this AD MMSE analysis requires all follow-up')
    if config.get('adjust_current_age_cubic') is not True or config.get('use_genotype_pcs') is not True:
        raise ValueError('Source did not use age cubic and genotype PCs')
    print('[FINAL] loading already-completed MAP discovery / ADNI replication; Stage2 screening will NOT rerun',flush=True)
    disc=check_rows(read_csv(discfile),'MAP discovery','MAP')
    rep=check_rows(read_csv(repfile),'ADNI replication','ADNI')
    cand=read_csv(candfile)
    d,r,m,primary=finalize_from_existing(disc,rep,cand)
    m.attrs['min_participants']=int(config.get('min_participants',50))
    print(f'[FINAL] MAP FDR pairs={len(m)}, ADNI same-lag+direction replicated={len(primary)}, nominal-only={int(m.replicated_same_lag_nominal_only.sum())}',flush=True)
    d.to_csv(out/'AD_MMSE_SUPPLEMENT_MAP_all_lag_models.csv.gz',index=False,compression='gzip')
    r.to_csv(out/'AD_MMSE_SUPPLEMENT_ADNI_all_lags_annotated.csv.gz',index=False,compression='gzip')
    robust=None
    if not args.no_robustness:
        for f in [args.pc_file,args.pc_manifest,args.phenotype_cov]:
            if not Path(f).is_file():raise FileNotFoundError(f)
        robust=fit_selected(m,args,out,core)
    full=add_robustness(m,robust);full=evidence(full)
    final=full.loc[full.PRIMARY_replicated_same_lag_direction].copy()
    final=final.sort_values(['evidence_tier','ADNI_pvalue','MAP_q_bh_global','gene'],kind='mergesort')
    full.sort_values(['PRIMARY_replicated_same_lag_direction','MAP_q_bh_global','gene'],ascending=[False,True,True],inplace=True)
    # Artifact names deliberately feature selection rule, not generic 'nominal' or legacy PPMI.
    full.to_csv(out/'AD_MMSE_SUPPLEMENT_all_MAP_FDR_IV_gene_pairs.csv',index=False)
    final.to_csv(out/'AD_MMSE_FINAL_replicated_same_lag_direction.csv',index=False)
    final.to_csv(out/'AD_MMSE_FINAL_evidence_graded.csv',index=False)
    gene_aggregate(full).to_csv(out/'AD_MMSE_FINAL_gene_summary.csv',index=False)
    if robust is not None:
        bad=robust.loc[~valid_bool(robust.FAST_fullfit_numeric_match)]
        bad.to_csv(out/'AD_MMSE_FULLFIT_CHECK_ERRORS.csv',index=False)
    else:bad=pd.DataFrame()
    counts=full.evidence_tier.value_counts().to_dict()
    facts={'status':'COMPLETE' if bad.empty and robust is not None else ('SELECTION_ONLY' if args.no_robustness else 'REVIEW_REQUIRED'),
           'analysis':'AD MMSE','stage1_instrument_source':'PPMI-PDBP validated temporal tQTL',
           'stage2_discovery':'MAP','stage2_replication':'ADNI',
           'primary_selection':'MAP q_BH_global<0.05 then ADNI p<0.05 at same MAP-significant lag AND same nonzero coef_sum direction; choose min ADNI BIC among eligible lags',
           'fdr_family':'all successful MAP IV-gene-lag models',
           'n_MAP_tested_models':int(len(d)),'n_MAP_FDR_lag_models':int(d.MAP_discovery_FDR.sum()),
           'n_MAP_FDR_IV_gene_pairs':int(len(full)),'n_ADNI_same_lag_nominal_pairs':int(full.replicated_same_lag_nominal_only.sum()),
           'n_ADNI_primary_same_lag_direction_pairs':int(len(final)),
           'n_ADNI_primary_genes':int(final.gene.nunique()),'evidence_tier_counts':counts,
           'AR_type_test':'participant-cluster-robust reduced-form F/Wald on SNP x cubic followup basis; NOT a classical Anderson-Rubin inversion nor a proof of IV validity',
           'true_causal_identification':'NOT ESTABLISHED; exclusion restriction, pleiotropy, colocalization untested; first-stage strength is NOT evaluated or used',
           'no_new_IV_QC':True,'no_new_stage2_high_throughput_run':True,'full_followup':True,
           'first_stage_F_used_in_grading':False,'first_stage_F_computed_in_finalization':False,
           'RF_BH_family':'RF for selected lag of ALL MAP FDR-discovered IV-gene pairs only; post-selection descriptive, not discovery FDR',
           'evidence_tier_definitions':{
               'A':'ADNI same-lag direction-concordant replicated, MAP selected-lag RF BH q<0.05 and ADNI RF p<0.05',
               'B':'ADNI same-lag direction-concordant replicated, both RF nominal p<0.05, MAP RF BH q not <0.05',
               'C':'ADNI same-lag direction-concordant replicated, but dual RF criteria unmet/unavailable',
               'D':'MAP BH discovery only, without ADNI same-lag coef_sum-concordant replication'},
           'source_files_sha256':{str(f):sha_file(f) for f in [discfile,repfile,candfile,conf]},
           'min_participants':int(config.get('min_participants',50)),
           'n_full_refit_mismatch_or_failed':int(len(bad))}
    (out/'AD_MMSE_FINAL_summary.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2))
    if not bad.empty:
        print(f'[ERROR] {len(bad)} full-fit/FAST mismatches. FINAL requires review; see AD_MMSE_FULLFIT_CHECK_ERRORS.csv',flush=True)
        raise RuntimeError('Full-detail numerical audit failed; do NOT label results validated')
    print(json.dumps(facts,indent=2,ensure_ascii=False),flush=True)
    print('[DONE] FINAL: ',out/'AD_MMSE_FINAL_replicated_same_lag_direction.csv',flush=True)

if __name__=='__main__':main()
