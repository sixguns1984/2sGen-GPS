#!/usr/bin/env python3
"""Conservative MRI DBM publication integration; NO new statistical estimation.

Preserves frozen MRI hit membership and primary estimates. Joins only diagnostics
actually calculated by the existing AR-type / multi-IV / LOO / LD scripts.
Uncomputable or missing diagnostics stay NA, not 0, and are NEVER re-estimated
using a different test. Stage-I power and 3-UPDRS provenance join on exact IDs.

The canonical publication file is backed up once before atomic replacement.
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

KEY = ['gene','lead_snp','artifact_id','roi']
PAIR = KEY[:3]
GENE_ROI = ['gene','roi']
PHENOS = ('updrs1','updrs2','updrs3')
CLINICAL = ['significant_updrs1','significant_updrs2','significant_updrs3',
            'phenotypes_significant','n_significant_phenotypes',
            'has_pooled_PD_SU_replication']
POWER = ['bootstrap_power_PP','bootstrap_power_PD','bootstrap_valid_PP','bootstrap_valid_PD',
         'bootstrap_mc_ci_low_PP','bootstrap_mc_ci_high_PP',
         'bootstrap_mc_ci_low_PD','bootstrap_mc_ci_high_PD','power_screen_status']
AR = ['iv_ar_test_type','robust_status','robust_n_obs','robust_n_people',
      'iv_reduced_form_wald_chi2','iv_reduced_form_df','iv_reduced_form_p',
      'iv_reduced_form_q_BH','iv_reduced_form_partial_r2']
MULTI = ['n_eligible_stage1_IVs','n_estimable_independent_IVs','multiIV_n',
         'multiIV_beta','multiIV_se','multiIV_p','multiIV_p_q_BH',
         'multiIV_Q','multiIV_Q_df','multiIV_Q_p','multiIV_Q_p_q_BH','multiIV_I2',
         'multiIV_direction_concordance_fraction','LOO_n','LOO_all_same_direction',
         'LOO_max_p','LOO_all_nominal_p05']
LD = ['cross_gene_max_r2','cross_gene_ld_flag',
      'n_cross_gene_ld_partners_r2ge_threshold','LD_cluster_id']
GRADES = ['MRI_descriptive_evidence_tier']
META = ['MRI_replication_available','MRI_evidence_calculation_status']
PRIMARY = ['n_obs','n_people','beta_grex_lag1','se_cluster','z_cluster',
           'p_cluster','q_global_BH','beta_DBM_lag1','mean_MRI_gap_years']
REVIEWER = KEY + ['phenotypes_significant','n_significant_phenotypes'] + [f'significant_{p}' for p in PHENOS] + [
    'n_people','grex_lag_order','beta_grex_lag1','grex_coef_sum','beta_DBM_lag1',
    'se_cluster','p_cluster','q_global_BH',
    'robust_status','iv_reduced_form_p','iv_reduced_form_q_BH','iv_reduced_form_df',
    'multiIV_n','multiIV_p_q_BH','multiIV_Q_p_q_BH','multiIV_I2',
    'LOO_n','LOO_all_nominal_p05','cross_gene_max_r2','cross_gene_ld_flag',
    'bootstrap_power_PP','bootstrap_power_PD','power_screen_status',
    'MRI_descriptive_evidence_tier']


def read(path, required=()):
    path=Path(path)
    if not path.exists(): raise FileNotFoundError(str(path))
    d=pd.read_csv(path,low_memory=False,dtype={c:'string' for c in KEY})
    missing=set(required)-set(d.columns)
    if missing: raise ValueError(f'{path}: missing columns {sorted(missing)}')
    return d


def optional(path, required=()):
    if path is None or not Path(path).is_file(): return None
    return read(path,required)


def uniq(d,by,label):
    if d.duplicated(by).any():
        raise ValueError(f'{label}: duplicate keys {d.loc[d.duplicated(by,keep=False),by].head(8)}')


def numerically_identical(old,new,col):
    a=pd.to_numeric(old,errors='coerce');b=pd.to_numeric(new,errors='coerce')
    return np.allclose(a,b,rtol=1e-8,atol=1e-10,equal_nan=True)


def merge_evidence(original,evidence):
    if evidence is None:return original.copy(),False
    uniq(evidence,KEY,'MRI evidence master')
    matches=original[KEY].merge(evidence[KEY],on=KEY,how='outer',indicator=True,validate='one_to_one')
    if not matches['_merge'].eq('both').all():
        raise ValueError(f'FDR MRI membership mismatch: {matches._merge.value_counts().to_dict()}')
    # Absolutely no changes to original MRI association coefficients, P or global BH.
    for c in ['p_cluster','q_global_BH','beta_grex_lag1','se_cluster']:
        if c in original and c in evidence:
            v=original[KEY+[c]].merge(evidence[KEY+[c]],on=KEY,how='inner',validate='one_to_one',suffixes=('_old','_new'))
            if not numerically_identical(v[c+'_old'],v[c+'_new'],c):
                raise ValueError(f'Primary MRI statistic {c} differs from evidence file; refusing to overwrite')
    # Retain original primary columns, import only auxiliary diagnostics and grades.
    allowed=set(AR+MULTI+LD+GRADES+META)
    incoming=[c for c in evidence if c in allowed and c not in KEY]
    old=original.drop(columns=[c for c in incoming if c in original],errors='ignore')
    return old.merge(evidence[KEY+incoming],on=KEY,how='left',validate='one_to_one'),True


def apply_optional(d,tab,keys,cols,label):
    if tab is None:return d,False
    uniq(tab,keys,label)
    use=[c for c in cols if c in tab.columns and c not in keys]
    # Skip columns already present (evidence master takes priority; no overwrites)
    use=[c for c in use if c not in d]
    return d.merge(tab[keys+use],on=keys,how='left',validate='many_to_one'),True


def normalize_clinical(d,clinical):
    if clinical is None: return d,False
    if not set(PAIR).issubset(clinical.columns):
        # A valid wide table may not contain artifact_id, but the strict frozen
        # Stage-I clinical candidate is preferred and expected for real publication.
        raise ValueError('Clinical candidate source must have gene, lead_snp, artifact_id')
    uniq(clinical,PAIR,'Power-qualified three-UPDRS clinical source')
    use=[c for c in CLINICAL+POWER if c in clinical]
    z=d.drop(columns=[c for c in use if c in d],errors='ignore').merge(
        clinical[PAIR+use],on=PAIR,how='left',validate='many_to_one',indicator='_clinmatch')
    if not z['_clinmatch'].eq('both').all():
        misses=z.loc[z['_clinmatch'].ne('both'),PAIR].drop_duplicates()
        raise ValueError(f'Cannot join MRI FDR hits to three-UPDRS clinical IVs: {misses.head(8)}')
    z=z.drop(columns='_clinmatch')
    # Rebuild only the phenotype annotation from frozen 3-UPDRS flags; never
    # infer association from missing flags.
    flags=[f'significant_{p}' for p in PHENOS]
    if set(flags).issubset(z):
        for c in flags:
            if z[c].isna().any():raise ValueError(f'Missing {c} in clinical candidate list')
            z[c]=z[c].astype('string').str.lower().isin(['true','1','yes']).astype(bool)
        expected=z.apply(lambda r:';'.join(p for p in PHENOS if r['significant_'+p]),axis=1)
        z['phenotypes_significant']=expected
        z['n_significant_phenotypes']=z[flags].sum(axis=1)
    return z,True


def integrate(original,clinical=None,evidence=None,ar=None,multi=None,ld=None):
    uniq(original,KEY,'Original MRI publication')
    d,has_full=merge_evidence(original,evidence)
    d,_=apply_optional(d,ar,KEY,AR,'Original MRI AR diagnostics')
    d,_=apply_optional(d,multi,GENE_ROI,MULTI,'MRI gene-ROI multi-IV summary')
    d,_=apply_optional(d,ld,PAIR,LD,'MRI cross-gene LD annotations')
    d,has_clin=normalize_clinical(d,clinical)
    # Consistent schema: not estimated = NA. Numerical NA is not a failed test.
    for c in dict.fromkeys(PRIMARY+AR+MULTI+LD+CLINICAL+POWER+GRADES+META):
        if c not in d.columns:d[c]=pd.NA
    # The source version can provide robust_status even if AR p is not computed.
    # No rank-reduced alternative AR is attempted and no zero P value is invented.
    if 'robust_status' in d:
        status=d['robust_status'].astype('string')
        missing_ar=pd.to_numeric(d['iv_reduced_form_p'],errors='coerce').isna()
        d.loc[missing_ar & status.eq('singular_reduced_form').fillna(False),'iv_reduced_form_p']=np.nan
    if not has_full:
        # We do not manufacture A/B/C tiers from partial evidence.
        if evidence is None and 'MRI_descriptive_evidence_tier' in d:
            d['MRI_descriptive_evidence_tier']=pd.NA
    # Clinical lag coefficients are not MRI coefficients. This MRI model has
    # exactly one GReX_previous exposure term. The GReX coefficient sum for an
    # L=1 model is exactly the already-reported beta_grex_lag1 (an identity,
    # NOT a new estimate or pooled coefficient). Never claim this for unknown
    # or multi-lag model formulas.
    if 'model' in d.columns:
        source_model = d['model'].astype('string').str.strip()
        single_lag = source_model.eq(
            'DBM_current ~ GReX_previous + DBM_previous + age_cubic + sex + genotype_PCs + IQR'
        ).fillna(False)
    else:
        single_lag = pd.Series(False, index=d.index)
    d['grex_lag_order'] = pd.Series(pd.NA,index=d.index,dtype='Int64')
    d.loc[single_lag,'grex_lag_order'] = 1
    d['grex_coef_sum'] = pd.to_numeric(d['beta_grex_lag1'], errors='coerce').where(single_lag)
    # Source values (including missing data) are preserved; do not invent
    # other MRI GReX lags, infer a coef_sum from model title alone otherwise,
    # or alter original robust/AR results.
    d['MRI_replication_available']=False
    d['MRI_evidence_calculation_status']=('complete_evidence_source' if has_full else
                                          'partial_or_no_evidence_source')
    return d


def atomic_csv(d,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=str(path.parent),suffix='.csv',delete=False) as f:
        temp=Path(f.name)
        d.to_csv(f,index=False)
    os.replace(temp,path)


def publish(root,component=1,canonical_file='',clinical_file='',evidence_file='',ar_file='',multi_file='',ld_file=''):
    root=Path(root)
    base=root/'single_iv_causal'/'mri_dbm_lag1'/f'pc{component}'
    pub=base/'publication_robustness';pub.mkdir(parents=True,exist_ok=True)
    canonical=Path(canonical_file) if canonical_file else pub/'Supplement_MRI_DBM_significant_IV_gene_ROI_integrated.csv'
    if not canonical.exists():raise FileNotFoundError(canonical)
    backup=pub/'Supplement_MRI_DBM_significant_IV_gene_ROI_integrated_before_evidence_v2.csv'
    original=read(backup if backup.exists() else canonical,KEY)
    clinical_path=Path(clinical_file) if clinical_file else root/'single_iv_causal'/'cross_phenotype_summary_lag1'/'PD_validated_IV_gene_stage1_candidates.csv'
    clinical=read(clinical_path,PAIR)  # Required: prevents accidental stale phenotype/power mapping.
    full_path=Path(evidence_file) if evidence_file else base/'evidence_v1'/'Supplement_MRI_DBM_significant_IV_gene_ROI_evidence_FULL.csv'
    ar_path=Path(ar_file) if ar_file else pub/'MRI_DBM_IV_reduced_form_AR_type_diagnostics.csv'
    multi_path=Path(multi_file) if multi_file else base/'evidence_v1'/'MRI_multiIV_gene_ROI_consistency_summary.csv'
    ld_path=Path(ld_file) if ld_file else None
    full=optional(full_path,KEY)
    ar=optional(ar_path,KEY)
    multi=optional(multi_path,GENE_ROI)
    ld=optional(ld_path,PAIR)
    combined=integrate(original,clinical,full,ar,multi,ld)
    for c in ['n_obs','n_people','beta_grex_lag1','se_cluster','p_cluster','q_global_BH']:
        if c in original:
            z=original[KEY+[c]].merge(combined[KEY+[c]],on=KEY,suffixes=('_old','_new'),validate='one_to_one')
            if not numerically_identical(z[c+'_old'],z[c+'_new'],c):
                raise ValueError(f'Frozen primary MRI value {c} changed; refusing to publish')
    if not backup.exists():shutil.copy2(canonical,backup)
    detail=pub/'Supplement_MRI_DBM_significant_IV_gene_ROI_evidence_FULL.csv'
    reviewer=pub/'Supplement_MRI_DBM_significant_IV_gene_ROI_reviewer.csv'
    # If re-running, the frozen backup is always the base, not last run's wide file.
    atomic_csv(combined,detail)
    atomic_csv(combined[[c for c in REVIEWER if c in combined]],reviewer)
    atomic_csv(combined,canonical)
    report={
        'n_FDR_hits':int(len(combined)),
        'n_genes':int(combined.gene.nunique()),
        'AR_test_definition':'Existing original reduced-form cluster Wald; NOT classical Anderson-Rubin. No alternative rank-aware re-fitting.',
        'lag_coefficient_rule':'Only explicit frozen L=1 MRI formula: grex_coef_sum = beta_grex_lag1. No unavailable lag estimated or filled.',
        'n_AR_evaluable':int(pd.to_numeric(combined.iv_reduced_form_p,errors='coerce').notna().sum()),
        'n_AR_unavailable':int(pd.to_numeric(combined.iv_reduced_form_p,errors='coerce').isna().sum()),
        'n_multiIV_n_ge_2':int((pd.to_numeric(combined.multiIV_n,errors='coerce')>=2).sum()),
        'n_LOO_n_ge_3':int((pd.to_numeric(combined.LOO_n,errors='coerce')>=3).sum()),
        'n_power80':int(combined.power_screen_status.astype('string').eq('pass_both_80pct').sum()),
        'original_mri_file':str(backup),
        'clinical_source':str(clinical_path),
        'evidence_source':str(full_path) if full is not None else None,
        'ar_source':str(ar_path) if ar is not None else None,
        'multiIV_source':str(multi_path) if multi is not None else None,
        'ld_source':str(ld_path) if ld is not None else None,
        'note':'Missing/failed diagnostics are NA. Original MRI P, BH and FDR hit membership unchanged. Evidence tiers are imported, not recomputed.',
    }
    (pub/'MRI_evidence_conservative_v2_manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False))
    print('[UPDATED]',canonical)
    print('[REVIEWER]',reviewer)
    print('[FULL]',detail)
    return combined


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--component',type=int,default=1)
    p.add_argument('--canonical-file',default='')
    p.add_argument('--clinical-file',default='')
    p.add_argument('--evidence-file',default='')
    p.add_argument('--ar-file',default='')
    p.add_argument('--multi-file',default='')
    p.add_argument('--ld-file',default='')
    a=p.parse_args()
    publish(a.root,a.component,a.canonical_file,a.clinical_file,a.evidence_file,a.ar_file,a.multi_file,a.ld_file)

if __name__=='__main__':main()
