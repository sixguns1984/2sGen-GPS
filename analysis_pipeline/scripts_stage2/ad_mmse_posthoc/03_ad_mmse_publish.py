#!/usr/bin/env python3
"""AD MMSE publication with the EXACT live PD evidence tier function.

Frozen MAP/ADNI Stage-II and original AR tests; Power, multi-IV and LD are posthoc.
One publication grade (PD descriptive_evidence_tier); never use first-stage F.
"""
from __future__ import annotations
import argparse,json,hashlib
from pd_tier_bridge import grade_ad_with_pd_function
from pathlib import Path
import numpy as np
import pandas as pd

KEY=['gene','lead_snp','artifact_id']
PAIR=KEY[:2]


def yes(s):return s.astype('string').fillna('false').str.lower().isin(('true','1','yes'))
def num(s):return pd.to_numeric(s,errors='coerce')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for piece in iter(lambda:f.read(1024*1024),b''):h.update(piece)
    return h.hexdigest()

def load_final(path):
    d=pd.read_csv(path,low_memory=False,dtype={k:str for k in KEY})
    if not set(KEY).issubset(d):raise ValueError(f'Missing keys from AD MMSE final: {set(KEY)-set(d)}')
    if 'PRIMARY_replicated_same_lag_direction' in d.columns:
        if not yes(d.PRIMARY_replicated_same_lag_direction).all():raise ValueError('Final file has non-replicated rows')
    if d.duplicated(KEY).any() or d.duplicated(PAIR).any():raise ValueError('AD final contains duplicate IV-gene pairs')
    return d

def attach_power(d,power,min_valid):
    x=d.copy()
    for co in ('PP','PD'):
        x[f'bootstrap_power_{co}']=np.nan
        x[f'bootstrap_valid_{co}']=np.nan
    if power is not None:
        p=pd.read_csv(power,low_memory=False,dtype={'gene':str,'variant':str})
        required={'gene','variant','cohort','bootstrap_power','bootstrap_valid'}
        if not required.issubset(p):raise ValueError(f'Power file missing {required-set(p)}')
        if not p.cohort.isin(('PP','PD')).all():raise ValueError('Unexpected power cohort; AD MMSE power is Stage-I PP/PD, NOT MAP/ADNI')
        p['lead_snp']=p.variant.astype(str)
        if p.duplicated(PAIR+['cohort']).any():raise ValueError('Duplicate IV-gene-cohort power records')
        for co in ('PP','PD'):
            z=p.loc[p.cohort.eq(co),PAIR+['bootstrap_power','bootstrap_valid']].copy()
            z=z.rename(columns={'bootstrap_power':f'__power_{co}','bootstrap_valid':f'__valid_{co}'})
            x=x.merge(z,on=PAIR,how='left',validate='one_to_one')
            x[f'bootstrap_power_{co}']=num(x.pop(f'__power_{co}'))
            x[f'bootstrap_valid_{co}']=num(x.pop(f'__valid_{co}'))
    evalmask=(x.bootstrap_power_PP.notna()&x.bootstrap_power_PD.notna()&
              x.bootstrap_valid_PP.ge(min_valid).fillna(False)&
              x.bootstrap_valid_PD.ge(min_valid).fillna(False)).fillna(False).astype(bool)
    passmask=(evalmask&x.bootstrap_power_PP.ge(.8).fillna(False)&
              x.bootstrap_power_PD.ge(.8).fillna(False)).fillna(False).astype(bool)
    x['tqtl_power_both_ge80']=passmask
    x['power_screen_status']=np.select([passmask,evalmask],['pass_both_80pct','below_80pct'],default='not_evaluable')
    x['power_method']='Stage-I PP/PD observed-donor overall-ctAR 8df bootstrap redetection; conditional post-selection'
    return x

def attach_multiiv(d,path):
    x=d.copy()
    wanted=['n_independent_stage1_IVs','n_finite_commonlag_estimates','multiIV_n','multiIV_beta',
            'multiIV_se','multiIV_p','multiIV_q_BH','multiIV_Q','multiIV_Q_p','multiIV_Q_q_BH',
            'multiIV_I2','multiIV_direction_concordance_fraction',
            'LOO_n','LOO_all_same_direction','LOO_all_nominal_p05','LOO_max_p','reference_lag']
    for co in ('MAP','ADNI'):
        for col in wanted:x[f'{co}_{col}']=np.nan
    if not path or not Path(path).is_file():return x
    m=pd.read_csv(path,low_memory=False,dtype={'gene':str})
    req={'gene','cohort'}
    if not req.issubset(m):raise ValueError(f'Multi-IV table missing {req-set(m)}')
    if m.duplicated(['gene','cohort']).any():raise ValueError('Duplicate gene/cohort multi-IV summary')
    for co in ('MAP','ADNI'):
        z=m.loc[m.cohort.eq(co)].copy()
        keep=['gene']+[col for col in wanted if col in z.columns]
        z=z[keep].rename(columns={c:f'__{co}_{c}' for c in keep if c!='gene'})
        x=x.merge(z,on='gene',how='left',validate='many_to_one')
        for col in keep:
            if col!='gene':x[f'{co}_{col}']=x.pop(f'__{co}_{col}')
    return x

def attach_ld(d,lddir,threshold=0.10):
    x=d.copy()
    x['cross_gene_max_r2']=np.nan
    x['cross_gene_ld_flag']=pd.Series(pd.NA,index=x.index,dtype='boolean')
    x['cross_gene_ld_status']='not_evaluated'
    if not lddir:return x
    lddir=Path(lddir)
    f=lddir/'lead_iv_positions.csv';pairs=lddir/'lead_iv_ld_pairs.csv'
    if not f.is_file() or not pairs.is_file():return x
    positions=pd.read_csv(f,low_memory=False,dtype={'SNP':str,'lead_snp':str})
    if 'lead_snp' not in positions:raise ValueError(f'LD position file missing lead_snp: {f}')
    available=set(positions.loc[positions.BP.notna(),'lead_snp'].astype(str))
    # Only genuine cross-gene SNP pairs count. SNPs for the SAME gene are excluded.
    mapgenes=x.groupby('lead_snp').gene.apply(lambda s:set(s.astype(str))).to_dict()
    maxes={}
    try:ld=pd.read_csv(pairs,low_memory=False,dtype={'SNP_A':str,'SNP_B':str})
    except pd.errors.EmptyDataError:ld=pd.DataFrame()
    for r in ld.itertuples(index=False):
        a=str(getattr(r,'SNP_A',''));b=str(getattr(r,'SNP_B',''))
        r2=num(pd.Series([getattr(r,'R2',np.nan)])).iloc[0]
        if not np.isfinite(r2):continue
        for ga in mapgenes.get(a,set()):
            for gb in mapgenes.get(b,set()):
                if ga==gb:continue
                maxes[(ga,a)]=max(float(r2),maxes.get((ga,a),0))
                maxes[(gb,b)]=max(float(r2),maxes.get((gb,b),0))
    ok=x.lead_snp.astype(str).isin(available)
    x.loc[ok,'cross_gene_ld_status']='evaluated'
    vals=pd.Series([maxes.get((str(g),str(s)),np.nan) for g,s in zip(x.gene,x.lead_snp)],index=x.index)
    x.loc[ok,'cross_gene_max_r2']=vals.loc[ok]
    x.loc[ok,'cross_gene_ld_flag']=(vals.loc[ok].ge(threshold).fillna(False)).to_numpy(bool)
    x.loc[~ok,'cross_gene_ld_status']='lead_snp_missing_in_LD_reference'
    return x

def attach_original_lag_coefficients(d,src,cohort):
    x=d.copy()
    for j in range(1,5):
        target=f'{cohort}_coef_x_lag{j}'
        if target not in x:x[target]=np.nan
    if not Path(src).is_file():return x
    raw=pd.read_csv(src,low_memory=False,dtype={k:str for k in KEY})
    if not set(KEY+['lag_order']).issubset(raw):return x
    coefs=[f'coef_x_lag{j}' for j in range(1,5)]
    available=[c for c in coefs if c in raw]
    if not available:return x
    raw['lag_order']=num(raw.lag_order)
    if 'cohort' in raw:raw=raw.loc[raw.cohort.astype(str).eq(cohort)]
    raw=raw[KEY+['lag_order']+available].copy()
    if raw.duplicated(KEY+['lag_order']).any():raise ValueError(f'Duplicate original model at same lag: {src}')
    raw=raw.rename(columns={'lag_order':'selected_lag_order',**{c:f'__{cohort}_{c}' for c in available}})
    x['selected_lag_order']=num(x.selected_lag_order)
    x=x.merge(raw,on=KEY+['selected_lag_order'],how='left',validate='one_to_one')
    for c in available:
        target=f'{cohort}_{c}';source=f'__{target}'
        a=num(x[target]);b=num(x.pop(source));
        if a.notna().any() and b.notna().any():
            chk=a.notna()&b.notna()
            if not np.allclose(a.loc[chk],b.loc[chk],rtol=1e-6,atol=1e-8):
                raise ValueError(f'{cohort}: original lag coefficient conflicts with final master {target}')
        x[target]=a.combine_first(b)
    # Consistency check of lag1 against original FINAL selected coefficient.
    lag1=f'{cohort}_coef_x_lag1';stored=f'{cohort}_coef_lag1'
    if lag1 in x and stored in x:
        a=num(x[lag1]);b=num(x[stored]);mask=a.notna()&b.notna()
        if mask.any() and not np.allclose(a[mask],b[mask],rtol=1e-6,atol=1e-8):
            raise ValueError(f'{cohort} lag1 differs from saved FINAL model; refusing publication regeneration')
    return x

def report_columns(x):
    lead=['gene','lead_snp','artifact_id','selected_lag_order',
          'MAP_pvalue','MAP_q_bh_global','MAP_n_participants',
          'ADNI_pvalue','ADNI_n_participants','selected_coef_sum_direction_concordant',
          'descriptive_evidence_tier']
    for co in ('MAP','ADNI'):
        lead += [f'{co}_coef_x_lag{i}' for i in range(1,5)] + [f'{co}_coef_sum',
                f'{co}_RF_cluster_pvalue',f'{co}_RF_cluster_F']
    for co in ('MAP','ADNI'):
        lead += [f'{co}_multiIV_n',f'{co}_multiIV_beta',f'{co}_multiIV_p']
        if co=='ADNI':lead += [f'{co}_multiIV_q_BH']
        lead += [
                 f'{co}_multiIV_Q_p',f'{co}_multiIV_I2',
                 f'{co}_LOO_n',f'{co}_LOO_all_nominal_p05']
    lead += ['AR_discoveryFDR_repNominal_support','MAP_RF_q_BH_PD_candidate_family',
             'multiIV_MAP_BH_support','MAP_multiIV_q_BH_PD_candidate_family',
             'MAP_multiIV_Q_q_BH_PD_candidate_family','multiIV_FDR_support',
             'multiIV_heterogeneity_FDR_flag','multiIV_LOO_stable','cross_gene_LD_clean',
             'multiIV_ADNI_nominal_support','multiIV_both_nominal_support',
             'cross_gene_max_r2','cross_gene_ld_flag','cross_gene_ld_status',
             'bootstrap_power_PP','bootstrap_valid_PP','bootstrap_power_PD','bootstrap_valid_PD',
             'power_screen_status','tqtl_power_both_ge80']
    return [c for c in lead if c in x.columns]

def publish(root,final_path,power,extra,lddir,out,min_valid,allow_missing=False):
    final=load_final(final_path)
    if power and not Path(power).is_file() and not allow_missing:raise FileNotFoundError(f'Power not yet available: {power}')
    if not power and not allow_missing:raise FileNotFoundError('Provide completed PP/PD power results; or --allow-missing-power for preliminary diagnostic tables')
    x=attach_power(final,power if power and Path(power).is_file() else None,min_valid)
    x=attach_multiiv(x,extra)
    x=attach_ld(x,lddir)
    # Exact same tiering logic as PD, loaded from the currently installed
    # scripts_stage2/10_assemble_final_master.py. No independent AD tier formula.
    graded,pd_tier_src=grade_ad_with_pd_function(x)
    if 'descriptive_evidence_tier' in x.columns:
        x=x.drop(columns=['descriptive_evidence_tier'])
    for c in graded.columns:
        if c in x.columns: x=x.drop(columns=[c])
    x=pd.concat([x,graded],axis=1)
    # The legacy AD-only tier is an input-stage historical artifact: remove it
    # from all NEW FULL and reviewer outputs (original input file untouched).
    x=x.drop(columns=['evidence_tier'],errors='ignore')
    x['multiIV_MAP_BH_support']=x.multiIV_FDR_support.astype(bool)
    x['multiIV_ADNI_nominal_support']=num(x['ADNI_multiIV_p']).lt(.05).fillna(False).astype(bool)
    x['multiIV_both_nominal_support']=(num(x['MAP_multiIV_p']).lt(.05)&num(x['ADNI_multiIV_p']).lt(.05)).fillna(False).astype(bool)
    fast=root/'single_iv_causal/ad_mmse/temporal_causal_agecubic_full_followup_fast'
    x=attach_original_lag_coefficients(x,fast/'discovery_all_IV_gene_lag_models.csv.gz','MAP')
    x=attach_original_lag_coefficients(x,fast/'replication_ADNI_all_lags_for_discovery_significant_IV_gene_pairs.csv.gz','ADNI')
    x['evidence_tier_is_selection_gate']=False
    if len(x)!=len(final) or x[KEY].duplicated().any():raise AssertionError('Publication changed selected MMSE pairs')
    out.mkdir(parents=True,exist_ok=True)
    full=out/'Supplement_AD_MMSE_IV_gene_all_statistics_with_power_robustness.csv'
    x.to_csv(full,index=False)
    # This is the ONE canonical AD-MMSE final evidence master for downstream
    # reporting; the original 20_finalize result is preserved only as raw input.
    canonical=out/'AD_MMSE_FINAL_evidence_graded.csv'
    x.to_csv(canonical,index=False)
    reviewer=out/'Supplement_AD_MMSE_IV_gene_publication_ALL.csv'
    cols=report_columns(x)
    x[cols].to_csv(reviewer,index=False)
    qualified=x.loc[x.tqtl_power_both_ge80].copy()
    power80=out/'Supplement_AD_MMSE_IV_gene_publication_POWER80.csv'
    qualified[cols].to_csv(power80,index=False)
    audit=out/'AD_MMSE_IV_gene_power_screen_audit.csv'
    x[KEY+['descriptive_evidence_tier','bootstrap_power_PP','bootstrap_valid_PP','bootstrap_power_PD','bootstrap_valid_PD','power_screen_status']].to_csv(audit,index=False)
    gene=x.groupby('gene',as_index=False).agg(n_replicated_IVs=('lead_snp','nunique'),n_power80_IVs=('tqtl_power_both_ge80','sum'))
    gene.to_csv(out/'AD_MMSE_gene_summary_power80.csv',index=False)
    manifest={'input_final':str(final_path),'final_sha256':sha(final_path),
              'stage1_ctar_power_results':str(power) if power else None,
              'multi_iv_details':str(extra) if extra else None,'LD_source':str(lddir) if lddir else None,
              'n_final_replicated_IV_gene_pairs':len(x),'n_final_genes':x.gene.nunique(),
              'n_power80_both_PP_PD':len(qualified),
              'n_power_below80':int(x.power_screen_status.eq('below_80pct').sum()),
              'n_power_not_evaluable':int(x.power_screen_status.eq('not_evaluable').sum()),
              'evidence_tiers_PD_standard':x.descriptive_evidence_tier.value_counts().to_dict(),
              'PD_tier_source':str(pd_tier_src),
              'PD_tier_function':'add_evidence_annotations',
              'tiering_definition':'Exact current PD descriptive tier. MAP RF -> PPMI AR; ADNI RF -> selected replication AR; MAP multi-IV -> PD multi-IV. Stage-I Power is NOT a tier input. First-stage F not used.',
              'selection':'AD MAP discovery q<0.05 and ADNI same-lag p<0.05 with coef_sum sign concordance, frozen',
              'power':'PPMI(PP)/PDBP(PD) observed-donor ctAR bootstrap redetection for Stage-I temporal tQTL; NOT MAP/ADNI MMSE outcome power',
              'AR':'Original MAP/ADNI cluster robust reduced-form F/Wald (not classical Anderson-Rubin), no recomputation',
              'multiIV':'IVW + Q/I2 + LOO descriptive within MAP and ADNI using existing AD-matched Stage-I eligible frozen IVs; no re-selection',
              'LD':'cross-gene local LD on MAP discovery cohort matching genotype reference, descriptive only',
              'nonevaluable':'Non-estimable tests remain NA; PD rule treats absent heterogeneity q as not flagged and absent LD flag as not flagged. Inspect LD audit before interpreting A tiers.',
              'outputs':{'harmonized_final_master':str(canonical),'reviewer_all':str(reviewer),'reviewer_power80':str(power80),'audit_full':str(full),'power_screen':str(audit)}}
    (out/'AD_MMSE_posthoc_evidence_manifest.json').write_text(json.dumps(manifest,indent=2,default=str,ensure_ascii=False))
    print(json.dumps(manifest,indent=2,default=str,ensure_ascii=False))
    return x,manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--final-file',default='')
    p.add_argument('--power-results',default='')
    p.add_argument('--multiiv-file',default='')
    p.add_argument('--ld-dir',default='')
    p.add_argument('--out-dir',default='')
    p.add_argument('--min-valid-bootstrap',type=int,default=800)
    p.add_argument('--allow-missing-power',action='store_true')
    a=p.parse_args();root=Path(a.root);base=root/'single_iv_causal/ad_mmse/standard_final'
    power=a.power_results or root/'manuscript_figures/tqtl_benchmark_v4/AD_MMSE_ctar_power_v1/locus_bootstrap_results.csv'
    extra=a.multiiv_file or base/'posthoc_evidence/multi_iv_gene_consistency_summary.csv'
    ld=a.ld_dir or base/'posthoc_evidence/ld_audit'
    final=Path(a.final_file) if a.final_file else base/'AD_MMSE_FINAL_evidence_graded.csv'
    out=Path(a.out_dir) if a.out_dir else base/'posthoc_evidence'
    publish(root,final,power,extra,ld,out,a.min_valid_bootstrap,a.allow_missing_power)
if __name__=='__main__':main()
