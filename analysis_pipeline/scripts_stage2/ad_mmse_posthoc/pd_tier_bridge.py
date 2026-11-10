#!/usr/bin/env python3
"""AD MMSE -> the exact live PD evidence-tier function.

The grading rules are NOT copied here. The existing PD 10_assemble_final_master.py
add_evidence_annotations() is loaded and applied to the MAP/ADNI analogous fields.
This is post-selection evidence characterization, not an IV selection/causal proof.
"""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import numpy as np
import pandas as pd


def _n(x, col):
    if col not in x.columns:
        return pd.Series(np.nan,index=x.index,dtype=float)
    return pd.to_numeric(x[col],errors='coerce')


def _truth(s):
    return s.astype('string').fillna('false').str.lower().str.strip().isin(['true','1','yes'])


def _source():
    p=Path(os.environ.get('PD_TIER_SOURCE', Path(__file__).resolve().parents[1]/'10_assemble_final_master.py'))
    if not p.is_file():
        raise FileNotFoundError(f'Cannot load current PD tier function from {p}. Set PD_TIER_SOURCE to existing 10_assemble_final_master.py; do NOT use a historic formula.')
    spec=importlib.util.spec_from_file_location('live_pd_10_assemble_final_master',p)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module,'add_evidence_annotations',None)):
        raise ValueError(f'PD source no longer exposes add_evidence_annotations: {p}')
    return p,module.add_evidence_annotations


def grade_ad_with_pd_function(ad):
    """Return only PD-defined grading and diagnostic columns, plus the source path.

    MAP (discovery) -> PPMI; ADNI (replication) -> PD_PLUS_SU;
    MAP gene-level multi-IV -> existing PD multi-IV; MAP SNP x time RF -> PD AR-type.
    AR q and gene-level multi-IV/q are recomputed by the *live PD code* over the
    final replicated candidate pair/gene families, as PD does.

    The ADNI multi-IV statistic is separately retained as an independent-cohort
    diagnostic, but does not add a new, AD-only tier requirement.
    """
    req=['gene','lead_snp','MAP_RF_cluster_pvalue','ADNI_RF_cluster_pvalue']
    missing=[c for c in req if c not in ad]
    if missing:raise ValueError(f'AD FINAL missing mandatory PD-mapped evidence: {missing}')
    x=pd.DataFrame(index=ad.index)
    x['gene']=ad['gene'].astype(str)
    x['lead_snp']=ad['lead_snp'].astype(str)
    x['replication_cohort_for_primary']='PD+SU'
    x['robust_PPMI_ar_pvalue']=_n(ad,'MAP_RF_cluster_pvalue')
    x['robust_PD_PLUS_SU_ar_pvalue']=_n(ad,'ADNI_RF_cluster_pvalue')
    x['coef_sum_direction_concordant']=_truth(ad['selected_coef_sum_direction_concordant']) if 'selected_coef_sum_direction_concordant' in ad else pd.Series(False,index=ad.index)
    x['multiIV_n']=_n(ad,'MAP_multiIV_n')
    x['multiIV_p']=_n(ad,'MAP_multiIV_p')
    x['multiIV_Q_p']=_n(ad,'MAP_multiIV_Q_p')
    x['LOO_all_same_direction']=_truth(ad['MAP_LOO_all_same_direction']) if 'MAP_LOO_all_same_direction' in ad else pd.Series(False,index=ad.index)
    x['LOO_all_nominal_p05']=_truth(ad['MAP_LOO_all_nominal_p05']) if 'MAP_LOO_all_nominal_p05' in ad else pd.Series(False,index=ad.index)
    # Follow the same exact PD handling of cross-gene LD flag, even for NA.
    x['cross_gene_ld_flag']=ad['cross_gene_ld_flag'] if 'cross_gene_ld_flag' in ad else pd.Series(pd.NA,index=ad.index,dtype='boolean')
    src,pd_annotate=_source()
    z=pd_annotate(x)
    columns=['descriptive_evidence_tier','AR_discoveryFDR_repNominal_support',
             'robust_PPMI_ar_q_BH','robust_PD_PLUS_SU_ar_q_BH',
             'multiIV_FDR_support','multiIV_p_q_BH','multiIV_Q_p_q_BH',
             'multiIV_heterogeneity_FDR_flag','multiIV_LOO_stable',
             'cross_gene_LD_clean','both_replication_directions_concordant',
             'evidence_tier_is_selection_gate']
    out=z[columns].copy()
    out=out.rename(columns={
        'robust_PPMI_ar_q_BH':'MAP_RF_q_BH_PD_candidate_family',
        'robust_PD_PLUS_SU_ar_q_BH':'ADNI_RF_q_BH_PD_candidate_family',
        'multiIV_p_q_BH':'MAP_multiIV_q_BH_PD_candidate_family',
        'multiIV_Q_p_q_BH':'MAP_multiIV_Q_q_BH_PD_candidate_family'})
    if not out.index.equals(ad.index):
        raise AssertionError('Live PD evidence function reordered AD candidates')
    return out,src
