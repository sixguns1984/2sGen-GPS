from __future__ import annotations
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

def _num(x):return pd.to_numeric(x,errors="coerce")
def _prefix(df,p,keys):return df.rename(columns={c:f"{p}{c}" for c in df.columns if c not in keys})



def _bh_qvalues(values):
    x=_num(values)
    out=pd.Series(np.nan,index=x.index,dtype=float)
    ok=x.notna() & np.isfinite(x)
    if ok.any():out.loc[ok]=multipletests(x.loc[ok].to_numpy(float),method="fdr_bh")[1]
    return out

def add_evidence_annotations(master):
    """Add post-selection robustness FDRs and descriptive evidence profiles.

    IMPORTANT: none of these annotations changes primary candidate membership.
    BH families are the frozen downstream candidate pairs (AR/coef-sum contrasts)
    or unique candidate genes with an estimable multi-IV statistic (multi-IV/Q).
    """
    x=master.copy()
    # Pair-level post-selection sensitivity tests.
    for cohort in ["PPMI","PD","SU","PD_PLUS_SU"]:
        pcol=f"robust_{cohort}_ar_pvalue"
        if pcol in x.columns:
            qcol=f"robust_{cohort}_ar_q_BH"
            x[qcol]=_bh_qvalues(x[pcol])
            x[f"robust_{cohort}_ar_FDR_support"]=x[qcol]<0.05
        ccol=f"robust_{cohort}_coef_sum_p_cluster"
        if ccol in x.columns:
            x[f"robust_{cohort}_coef_sum_q_BH"]=_bh_qvalues(x[ccol])
    # AR robustness follows the same discovery/replication logic as the primary analysis.
    # PPMI is the discovery cohort (BH-FDR), whereas PD and SU are independent replications (nominal P<0.05).
    # The more stringent all-three-FDR quantity is retained only as a descriptive annotation.
    ppmi_ar_p=_num(x.get("robust_PPMI_ar_pvalue",np.nan));pd_ar_p=_num(x.get("robust_PD_ar_pvalue",np.nan));su_ar_p=_num(x.get("robust_SU_ar_pvalue",np.nan));ppmi_ar_q=_num(x.get("robust_PPMI_ar_q_BH",np.nan));pd_ar_q=_num(x.get("robust_PD_ar_q_BH",np.nan));su_ar_q=_num(x.get("robust_SU_ar_q_BH",np.nan))
    x["AR_all3_nominal_support"]=np.nan # obsolete under pooled/PD primary replication
    selected_ar_p=_num(x.get("robust_PD_ar_pvalue",np.nan)) if "replication_cohort_for_primary" in x and x["replication_cohort_for_primary"].eq("PD").all() else _num(x.get("robust_PD_PLUS_SU_ar_pvalue",np.nan))
    x["AR_discoveryFDR_repNominal_support"]=(ppmi_ar_q<0.05)&(selected_ar_p<0.05)
    x["AR_replication_design"]="PD" if "replication_cohort_for_primary" in x and x["replication_cohort_for_primary"].eq("PD").all() else "PD+SU"
    x["AR_all3_FDR_stringent_annotation"]=np.nan
    # Multi-IV statistics are gene-level: BH once per unique gene, then merge back.
    g=x.sort_values(["gene","lead_snp"],kind="mergesort").drop_duplicates("gene").copy()
    for pcol,qcol in [("multiIV_p","multiIV_p_q_BH"),("multiIV_Q_p","multiIV_Q_p_q_BH")]:
        if pcol in g.columns:g[qcol]=_bh_qvalues(g[pcol])
    keep=[c for c in ["gene","multiIV_p_q_BH","multiIV_Q_p_q_BH"] if c in g.columns]
    if len(keep)>1:
        x=x.drop(columns=[c for c in keep[1:] if c in x.columns],errors="ignore").merge(g[keep],on="gene",how="left",validate="many_to_one")
    x["multiIV_FDR_support"]=_num(x.get("multiIV_p_q_BH",np.nan))<0.05
    x["multiIV_heterogeneity_FDR_flag"]=_num(x.get("multiIV_Q_p_q_BH",np.nan))<0.05
    x["cross_gene_LD_clean"]=~x.get("cross_gene_ld_flag",False).fillna(False).astype(bool) if isinstance(x.get("cross_gene_ld_flag",False),pd.Series) else True
    x["both_replication_directions_concordant"]=x["coef_sum_direction_concordant"].fillna(False).astype(bool) if "coef_sum_direction_concordant" in x else False
    n=_num(x.get("multiIV_n",np.nan))
    loo_dir=x.get("LOO_all_same_direction",pd.Series(np.nan,index=x.index)).eq(True)
    loo_sig=x.get("LOO_all_nominal_p05",pd.Series(np.nan,index=x.index)).eq(True)
    # For 2-IV genes LOO is undefined; for >=3 require stable sign + all leave-one-out pooled P<0.05.
    x["multiIV_LOO_stable"]=(n==2) | ((n>=3)&loo_dir&loo_sig)
    qhet=_num(x.get("multiIV_Q_p_q_BH",np.nan))
    no_hetero=(qhet.isna() | (qhet>=0.05))
    # Descriptive tiers are NOT selection rules.
    multi_stable=x["multiIV_FDR_support"] & no_hetero & x["multiIV_LOO_stable"] & x["cross_gene_LD_clean"]
    ar_stable=x["AR_discoveryFDR_repNominal_support"] & x["both_replication_directions_concordant"] & no_hetero & x["cross_gene_LD_clean"]
    tier=np.full(len(x),"C_primary_replicated",dtype=object)
    tier[x["AR_discoveryFDR_repNominal_support"].to_numpy(bool)]="B_AR_discoveryFDR_repNominal"
    tier[x["multiIV_FDR_support"].to_numpy(bool)]="B_multiIV_FDR"
    tier[ar_stable.to_numpy(bool)]="A_crosscohort_AR_supported"
    tier[multi_stable.to_numpy(bool)]="A_multiIV_stable"
    tier[(ar_stable & multi_stable).to_numpy(bool)]="A_both_AR_and_multiIV_supported"
    x["descriptive_evidence_tier"]=tier
    x["evidence_tier_is_selection_gate"]=False
    x["stage1_both_F_ge10"]=np.nan # historical scalar F not valid weak-IV gate for full temporal tQTL
    return x

