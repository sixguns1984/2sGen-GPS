#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
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

def ld_annotations(ldfile,master,threshold=0.10):
    keys=["gene","lead_snp"];out=master[keys].copy();out["cross_gene_max_r2"]=0.0;out["n_cross_gene_ld_partners_r2ge_threshold"]=0;out["cross_gene_ld_flag"]=False;out["LD_cluster_id"]=""
    p=Path(ldfile)
    if not p.exists() or p.stat().st_size==0:return out
    ld=pd.read_csv(p,low_memory=False)
    if ld.empty:return out
    ld["R2"]=_num(ld.R2);gm=master.groupby("lead_snp").gene.apply(lambda s:sorted(set(s.astype(str)))).to_dict();edges=[];pair_stats={}
    for r in ld.itertuples(index=False):
        ga=gm.get(str(r.SNP_A),[]);gb=gm.get(str(r.SNP_B),[]);r2=float(r.R2) if np.isfinite(r.R2) else np.nan
        for a in ga:
            for b in gb:
                if a==b or not np.isfinite(r2):continue
                for snp,g,other in [(str(r.SNP_A),a,b),(str(r.SNP_B),b,a)]:
                    k=(g,snp);d=pair_stats.setdefault(k,{"max":0.0,"partners":set()});d["max"]=max(d["max"],r2)
                    if r2>=threshold:d["partners"].add(other)
                if r2>=threshold:edges.append((a,b))
    # Connected components at threshold.
    adj={g:set() for g in master.gene.astype(str).unique()}
    for a,b in edges:adj.setdefault(a,set()).add(b);adj.setdefault(b,set()).add(a)
    seen=set();cluster={};cid=0
    for g in sorted(adj):
        if g in seen or not adj[g]:continue
        cid+=1;stack=[g];comp=[];seen.add(g)
        while stack:
            u=stack.pop();comp.append(u)
            for v in adj.get(u,[]):
                if v not in seen:seen.add(v);stack.append(v)
        lab=f"LD{cid}:"+";".join(sorted(comp))
        for u in comp:cluster[u]=lab
    vals=[]
    for r in out.itertuples(index=False):
        d=pair_stats.get((str(r.gene),str(r.lead_snp)),{"max":0.0,"partners":set()});vals.append((d["max"],len(d["partners"]),len(d["partners"])>0,cluster.get(str(r.gene),"")))
    out[["cross_gene_max_r2","n_cross_gene_ld_partners_r2ge_threshold","cross_gene_ld_flag","LD_cluster_id"]]=pd.DataFrame(vals,index=out.index)
    return out

def main():
    ap=argparse.ArgumentParser(description="Assemble the frozen exact-replication candidate set and all post-selection causal robustness annotations into one final IV-gene master table.");ap.add_argument("--root",required=True);ap.add_argument("--analysis-name",required=True);ap.add_argument("--standard-dir",default="");ap.add_argument("--robustness-dir",default="");ap.add_argument("--ld-dir",default="");ap.add_argument("--ld-threshold",type=float,default=0.10);a=ap.parse_args()
    root=Path(a.root);std=Path(a.standard_dir) if a.standard_dir else root/"single_iv_causal"/a.analysis_name/"standard_final";rob=Path(a.robustness_dir) if a.robustness_dir else std/"robustness";lddir=Path(a.ld_dir) if a.ld_dir else std/"ld_audit";master=pd.read_csv(std/f"{a.analysis_name}_downstream_IV_gene_master.csv",low_memory=False);keys=["gene","lead_snp"]
    exact=pd.read_csv(rob/"exact_candidate_selectedlag_robustness.csv",low_memory=False)
    for cohort,prefix in [("PP","robust_PPMI_"),("PD","robust_PD_"),("SU","robust_SU_"),("PD+SU","robust_PD_PLUS_SU_")]:
        z=exact[exact.cohort.astype(str).eq(cohort)].drop(columns="cohort").drop_duplicates(keys)
        master=master.merge(_prefix(z,prefix,keys),on=keys,how="left")
    miv=pd.read_csv(rob/"multi_iv_gene_consistency_summary.csv",low_memory=False);miv=miv.drop_duplicates("gene");master=master.merge(miv,on="gene",how="left",suffixes=("","_multiIV"))
    ldann=ld_annotations(lddir/"lead_iv_ld_pairs.csv",master,a.ld_threshold);master=master.merge(ldann,on=keys,how="left")
    master=add_evidence_annotations(master)
    # Final concise columns, while preserving a full wide table for traceability.
    core=["gene","lead_snp","primary_downstream_candidate","pooled_PD_PLUS_SU_support","selection_rule","n_replicated_IV_gene_pairs"]
    primary=["PPMI_lag_order","PPMI_pvalue","PPMI_q_bh_global","PPMI_joint_F","PPMI_partial_r2","PPMI_coef_sum","PD_lag_order","PD_pvalue","PD_coef_sum","SU_lag_order","SU_pvalue","SU_coef_sum","PD_PLUS_SU_lag_order","PD_PLUS_SU_pvalue","PD_PLUS_SU_coef_sum","PPMI_PD_direction_concordant","PPMI_SU_direction_concordant","both_replications_direction_concordant","PPMI_PD_PLUS_SU_direction_concordant","coef_sum_direction_concordant","lag1_direction_concordant"]
    strength=["strength_ppmi_first_stage_F","strength_ppmi_first_stage_partial_R2","strength_pdbp_first_stage_F","strength_pdbp_score_partial_R2","stage1_pdbp_direction_value","stage1_ld_prune_max_r2_to_kept"]
    robust=[]
    for p in ["robust_PPMI_","robust_PD_","robust_SU_","robust_PD_PLUS_SU_"]:
        robust += [p+c for c in ["coef_sum_se_cluster","coef_sum_p_cluster","ar_pvalue","ar_F","ar_df1","ar_df2","exposure_condition","ar_condition","n_obs","n_participants"]]
    multi=["reference_candidate_snp","reference_lag","n_independent_stage1_IVs","n_finite_commonlag_estimates","multiIV_n","multiIV_beta","multiIV_se","multiIV_p","multiIV_Q","multiIV_Q_df","multiIV_Q_p","multiIV_I2","multiIV_direction_concordance_fraction","LOO_n","LOO_all_same_direction","LOO_max_p","LOO_all_nominal_p05"]
    fdrcols=[]
    for p in ["robust_PPMI_","robust_PD_","robust_SU_","robust_PD_PLUS_SU_"]:
        fdrcols += [p+"ar_q_BH",p+"ar_FDR_support",p+"coef_sum_q_BH"]
    fdrcols += ["AR_all3_nominal_support","AR_discoveryFDR_repNominal_support","AR_all3_FDR_stringent_annotation","multiIV_p_q_BH","multiIV_Q_p_q_BH","multiIV_FDR_support","multiIV_heterogeneity_FDR_flag","multiIV_LOO_stable","stage1_both_F_ge10"]
    ldcols=["cross_gene_max_r2","n_cross_gene_ld_partners_r2ge_threshold","cross_gene_ld_flag","cross_gene_LD_clean","LD_cluster_id","both_replication_directions_concordant","descriptive_evidence_tier","evidence_tier_is_selection_gate"]
    cols=[]
    for c in core+primary+strength+robust+multi+fdrcols+ldcols:
        if c in master.columns and c not in cols:cols.append(c)
    concise=master[cols].copy();concise["robustness_statistics_are_selection_gates"]=False;concise["standard_pipeline_excludes_MR_Egger_and_coloc"]=True
    master.to_csv(std/f"{a.analysis_name}_FINAL_causal_IV_gene_master_FULL.csv",index=False);concise.to_csv(std/f"{a.analysis_name}_FINAL_causal_IV_gene_master.csv",index=False)
    # Gene-level final summary.
    # Gene-level summary with the best descriptive tier across final IV-gene pairs.
    tier_rank={"A_both_AR_and_multiIV_supported":1,"A_multiIV_stable":2,"A_crosscohort_AR_supported":3,"B_multiIV_FDR":4,"B_AR_discoveryFDR_repNominal":5,"C_primary_replicated":6}
    cg=concise.copy();cg["_tier_rank"]=cg["descriptive_evidence_tier"].map(tier_rank).fillna(99)
    g=cg.groupby("gene",as_index=False).agg(n_final_IV_gene_pairs=("lead_snp","nunique"),n_pooled_supported_pairs=("pooled_PD_PLUS_SU_support","sum"),any_cross_gene_ld_flag=("cross_gene_ld_flag","max"),any_AR_discoveryFDR_repNominal_support=("AR_discoveryFDR_repNominal_support","max"),any_AR_all3_FDR_stringent_annotation=("AR_all3_FDR_stringent_annotation","max"),any_multiIV_FDR_support=("multiIV_FDR_support","max"),best_tier_rank=("_tier_rank","min"))
    inv={v:k for k,v in tier_rank.items()};g["best_descriptive_evidence_tier"]=g.best_tier_rank.map(inv);g=g.drop(columns="best_tier_rank")
    gm=miv.drop_duplicates("gene");g=g.merge(gm,on="gene",how="left")
    fq=master.sort_values(["gene","lead_snp"]).drop_duplicates("gene")[[c for c in ["gene","multiIV_p_q_BH","multiIV_Q_p_q_BH"] if c in master.columns]]
    if len(fq.columns)>1:g=g.merge(fq,on="gene",how="left")
    g.to_csv(std/f"{a.analysis_name}_FINAL_causal_gene_summary.csv",index=False)
    concise.descriptive_evidence_tier.value_counts(dropna=False).rename_axis("descriptive_evidence_tier").reset_index(name="n_IV_gene_pairs").to_csv(std/f"{a.analysis_name}_evidence_tier_counts.csv",index=False)
    audit=pd.read_csv(rob/"fast_vs_full_detail_numeric_audit.csv") if (rob/"fast_vs_full_detail_numeric_audit.csv").exists() else pd.DataFrame();manifest={"analysis_name":a.analysis_name,"final_selection_rule":"PPMI global BH, PD+SU (UPDRS) or PD (MoCA) nominal significance at PPMI significant lag; selected replication best BIC and coef_sum direction concordance","pooled_PD_PLUS_SU":"primary for UPDRS only","n_final_pairs":int(len(concise)),"n_final_genes":int(concise.gene.nunique()),"post_selection_robustness_not_filters":["first-stage F/partial R2","cluster-robust coefficient SE/P and BH annotation","AR-type reduced-form test; discovery PPMI BH-FDR plus nominal selected PD+SU or PD replication (as appropriate), supportive only","old all-three AR BH-FDR metric deprecated (NA)","multiple-independent-IV fixed-effect consistency and BH annotation","Cochran Q/I2 and heterogeneity BH annotation","leave-one-IV-out","cross-gene LD audit","descriptive evidence tier"],"not_in_standard_pipeline":["MR-Egger","scalarized MR","colocalization"],"fast_full_detail_numeric_audit_allclose":bool(audit["allclose_1e-8"].fillna(False).all()) if len(audit) and "allclose_1e-8" in audit.columns else None,"files":{"final_master":str(std/f"{a.analysis_name}_FINAL_causal_IV_gene_master.csv"),"full_master":str(std/f"{a.analysis_name}_FINAL_causal_IV_gene_master_FULL.csv"),"gene_summary":str(std/f"{a.analysis_name}_FINAL_causal_gene_summary.csv"),"funnel":str(std/f"{a.analysis_name}_selection_funnel.csv"),"robustness_manifest":str(rob/"robustness_manifest.json")}};(std/f"{a.analysis_name}_FINAL_causal_manifest.json").write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
