#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ID_GENE=("gene","Ensembl_ID","gene_id")
ID_SNP=("lead_snp","Variant_ID","snp","snpid")


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _normalize_ids(df: pd.DataFrame, label: str) -> pd.DataFrame:
    x=df.copy()
    g=next((c for c in ID_GENE if c in x.columns),None)
    s=next((c for c in ID_SNP if c in x.columns),None)
    if g is None or s is None:
        raise KeyError(f"{label}: cannot identify gene/SNP columns; columns={x.columns.tolist()}")
    ren={}
    if g!="gene":ren[g]="gene"
    if s!="lead_snp":ren[s]="lead_snp"
    x=x.rename(columns=ren)
    x["gene"]=x["gene"].astype(str)
    x["lead_snp"]=x["lead_snp"].astype(str)
    return x


def _dedup_pair(df: pd.DataFrame, label: str) -> pd.DataFrame:
    x=_normalize_ids(df,label)
    dup=x.duplicated(["gene","lead_snp"],keep=False)
    if dup.any():
        # Replication best-BIC files should already be one row/pair. If not, choose deterministically.
        sort=[]
        for c in ["q_bh_global","pvalue","p","bic","BIC","lag_order"]:
            if c in x.columns:
                x[c]=pd.to_numeric(x[c],errors="coerce")
                sort.append(c)
        if sort:
            x=x.sort_values(sort+["gene","lead_snp"],na_position="last")
        else:
            x=x.sort_values(["gene","lead_snp"])
        x=x.drop_duplicates(["gene","lead_snp"],keep="first")
    return x.reset_index(drop=True)


def _prefix_payload(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    x=df.copy()
    ren={c:f"{prefix}_{c}" for c in x.columns if c not in {"gene","lead_snp"}}
    return x.rename(columns=ren)


def _effect_col(df: pd.DataFrame, prefix: str):
    preferred=[
        f"{prefix}_coef_sum",f"{prefix}_beta_sum",f"{prefix}_effect",
        f"{prefix}_beta",f"{prefix}_coef",f"{prefix}_estimate"
    ]
    for c in preferred:
        if c in df.columns:return c
    # Avoid lag-specific coefficients as a default direction measure unless nothing else exists.
    for c in df.columns:
        lc=c.lower()
        if c.startswith(prefix+"_") and any(k in lc for k in ["coef_sum","effect_sum","beta_sum"]):return c
    return None


def _find_file(base: Path, names) -> Path:
    for n in names:
        p=base/n
        if p.exists():return p
    raise FileNotFoundError(f"None of the expected files exist under {base}: {names}")


def _count_row(step, df):
    return {"step":step,"n_IV_gene_pairs":int(len(df)),"n_genes":int(df.gene.nunique()) if len(df) else 0,"n_IVs":int(df.lead_snp.nunique()) if len(df) else 0}


def _bh(p):
    p=pd.to_numeric(pd.Series(p),errors="coerce").to_numpy(float)
    out=np.full(len(p),np.nan)
    ok=np.isfinite(p)
    if not ok.any():return out
    vals=p[ok];order=np.argsort(vals);ranked=vals[order];m=len(ranked)
    q=ranked*m/np.arange(1,m+1);q=np.minimum.accumulate(q[::-1])[::-1];q=np.clip(q,0,1)
    tmp=np.empty(m);tmp[order]=q;out[np.where(ok)[0]]=tmp
    return out


def main():
    ap=argparse.ArgumentParser(description="Standard finalization of FAST temporal-causal results. Primary downstream selection = PPMI discovery BH and pooled PD+SU (UPDRS) or PD (MoCA) nominal p<0.05 at PPMI significant lag, best BIC and coef_sum sign concordance. AR, multiIV and LD are annotations.")
    ap.add_argument("--root",required=True)
    ap.add_argument("--analysis-name",required=True)
    ap.add_argument("--stage13-dir",default=None,help="FAST full-universe Stage13 output directory")
    ap.add_argument("--out-dir",default=None)
    ap.add_argument("--stage1-eligible",default=None)
    ap.add_argument("--strength-file",default=None)
    ap.add_argument("--ar-detail-file",default=None,help="Optional targeted/full-detail lag-model file containing AR annotations; never used for selection")
    ap.add_argument("--multi-iv-summary",default=None,help="Optional gene-level Q/I2/LOO summary; never used for selection")
    ap.add_argument("--ld-audit-file",default=None,help="Optional pair/gene LD-audit table; never used for selection")
    # Compatibility with the existing run_stage2_postprocess_existing_fast.sh runner.
    ap.add_argument("--replication-cohorts",default="PD,SU")
    ap.add_argument("--replication-pooled-groups",default="PD+SU")
    a=ap.parse_args()
    if a.analysis_name.lower()!="moca" and "PD+SU" not in a.replication_pooled_groups.replace(" ",""):
        raise ValueError("UPDRS requires an existing jointly fitted PD+SU replication model")

    root=Path(a.root)
    base=Path(a.stage13_dir) if a.stage13_dir else root/"single_iv_causal"/a.analysis_name/"temporal_causal_agecubic_full_followup_fast"
    out=Path(a.out_dir) if a.out_dir else root/"single_iv_causal"/a.analysis_name/"standard_final"
    out.mkdir(parents=True,exist_ok=True)

    stage1_path=Path(a.stage1_eligible) if a.stage1_eligible else root/"single_iv_causal"/"iv_validation"/"single_iv_primary_eligible.csv"
    strength_path=Path(a.strength_file) if a.strength_file else root/"single_iv_causal"/"iv_strength_all"/"all_IV_first_stage_strength.csv"

    discovery_path=_find_file(base,["final_IV_pair_summary.csv","discovery_selected_IV_gene_models.csv","discovery_selected_models.csv"])
    pd_path=_find_file(base,["replication_PD_PRIMARY_same_discovery_significant_lag_nominal_p05_bestBIC.csv","replication_PD_nominal_p05_pass_bestBIC.csv"])
    su_path=(_find_file(base,["replication_SU_PRIMARY_same_discovery_significant_lag_nominal_p05_bestBIC.csv","replication_SU_nominal_p05_pass_bestBIC.csv"]) if a.analysis_name.lower()!="moca" else None)
    pool_path=(_find_file(base,["replication_PD_PLUS_SU_PRIMARY_same_discovery_significant_lag_nominal_p05_bestBIC.csv","replication_PD_PLUS_SU_nominal_p05_pass_bestBIC.csv"]) if a.analysis_name.lower()!="moca" else None)

    stage1=_dedup_pair(_read_table(stage1_path),"stage1 eligible")
    discovery=_dedup_pair(_read_table(discovery_path),"discovery")
    pdrep=_dedup_pair(_read_table(pd_path),"PD replication")
    surep=_dedup_pair(_read_table(su_path),"SU replication") if su_path else pd.DataFrame(columns=["gene","lead_snp"])
    pool=_dedup_pair(_read_table(pool_path),"PD+SU pooled") if pool_path else pd.DataFrame(columns=["gene","lead_snp"])

    # PPMI significant lag + true pooled PD+SU for UPDRS; PD alone for MoCA.
    # Always pick best replication BIC BEFORE coef_sum sign concordance.
    keys=["gene","lead_snp"]
    rep_cohort="PD" if a.analysis_name.lower()=="moca" else "PD+SU"
    import importlib.util
    helper=Path(__file__).with_name("12_summarize_pd_four_phenotypes.py")
    spec=importlib.util.spec_from_file_location("cross_phenotype_lag1_selector",helper)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    selected=mod.select_joint_replication(base,rep_cohort)
    if selected.duplicated(keys).any():raise ValueError("Duplicated selected IV-gene")
    # Replace legacy lag1-prefiltered PD/SU/pooled manifests for status/funnel
    # with the actual newly selected-lag nominal replication support.
    if rep_cohort=="PD+SU":
        for cohort in ("PD","SU"):
            rawfile=next((p for p in [base/f"replication_{cohort}_all_lags_annotated.csv.gz",
                                          base/f"replication_{cohort}_all_lags_annotated.csv"] if p.exists()),None)
            if rawfile is None:raise FileNotFoundError(f"Missing {cohort} all-lags replication results")
            raw=_read_table(rawfile)
            if raw.duplicated(keys+["lag_order"]).any():raise ValueError(f"Duplicate {cohort} raw lag models")
            raw=raw.merge(selected[keys+["lag_order"]],on=keys+["lag_order"],how="inner",validate="one_to_one")
            raw=raw.loc[pd.to_numeric(raw.pvalue,errors="coerce").lt(.05)].copy()
            if cohort=="PD":pdrep=_dedup_pair(raw,"PD nominal same selected lag")
            else:surep=_dedup_pair(raw,"SU nominal same selected lag")
        pool=_dedup_pair(selected,"pooled PD+SU nominal same selected lag")
    candidate_keys=selected[keys].drop_duplicates(keys).copy()
    # The older precomputed PRIMARY pooled CSV may already be lag1-filtered.
    # For UPDRS the new selected pairs came from the actual FULL pooled all-lags
    # p<0.05 model, so pooled support must be TRUE even for newly restored pairs.
    pool_keys=(selected[keys].drop_duplicates(keys).assign(pooled_PD_PLUS_SU_support=True)
               if rep_cohort=="PD+SU" else
               pool[keys].drop_duplicates(keys).assign(pooled_PD_PLUS_SU_support=True))
    candidate_keys=candidate_keys.merge(pool_keys,on=keys,how="left")
    candidate_keys["pooled_PD_PLUS_SU_support"]=candidate_keys["pooled_PD_PLUS_SU_support"].eq(True)
    candidate_keys["primary_downstream_candidate"]=True
    candidate_keys["selection_rule"]=f"PPMI global BH FDR<0.05; {rep_cohort} nominal p<0.05 at PPMI significant lag, replication BIC best, PPMI vs replication coef_sum same sign"

    # Build an all-discovery status table first.
    status=discovery.copy()
    status=status.merge(pdrep[keys].assign(PD_replication=True),on=keys,how="left")
    status=status.merge(surep[keys].assign(SU_replication=True),on=keys,how="left")
    status=status.merge(pool[keys].assign(PD_PLUS_SU_pooled_support=True),on=keys,how="left")
    for c in ["PD_replication","SU_replication","PD_PLUS_SU_pooled_support"]:
        status[c]=status[c].eq(True)
    status["both_independent_replications"]=status.PD_replication & status.SU_replication
    # Pooled UPDRS support is established by the newly selected FULL replication result,
    # not the legacy PRIMARY CSV annotated under lag1 direction.
    if rep_cohort=="PD+SU":
        status["PD_PLUS_SU_pooled_support"]=status[keys].merge(
            candidate_keys[keys].assign(_present=True),on=keys,how="left")["_present"].eq(True).to_numpy(dtype=bool)
    status["primary_downstream_candidate"]=status[keys].merge(candidate_keys[keys].assign(_selected=True),on=keys,how="left")["_selected"].eq(True).to_numpy(dtype=bool)
    status.to_csv(out/f"{a.analysis_name}_all_discovery_pairs_replication_status.csv",index=False)

    # Master table: candidate keys + full discovery/replication payloads + Stage-I + strength annotations.
    master=candidate_keys.copy()
    for label,df in [("PPMI",discovery),("PD",pdrep),("SU",surep),("PD_PLUS_SU",pool)]:
        master=master.merge(_prefix_payload(df,label),on=keys,how="left")
    master=master.merge(_prefix_payload(stage1,"stage1"),on=keys,how="left")

    if strength_path.exists():
        strength=_dedup_pair(_read_table(strength_path),"strength")
        master=master.merge(_prefix_payload(strength,"strength"),on=keys,how="left")

    # Aggregate coef_sum direction is the primary selection gate. Lag-1 is audit-only.
    pcol="PPMI_coef_sum"
    dcol="PD_coef_sum"
    scol="SU_coef_sum"
    poolcol="PD_PLUS_SU_coef_sum"
    if not all(c in master.columns for c in [pcol,(dcol if rep_cohort=="PD" else poolcol)]):raise ValueError("coef_sum columns unavailable")
    if pcol in master.columns and dcol in master.columns:
        master["PPMI_PD_direction_concordant"]=np.sign(pd.to_numeric(master[pcol],errors="coerce"))==np.sign(pd.to_numeric(master[dcol],errors="coerce"))
    if pcol in master.columns and scol in master.columns:
        master["PPMI_SU_direction_concordant"]=np.sign(pd.to_numeric(master[pcol],errors="coerce"))==np.sign(pd.to_numeric(master[scol],errors="coerce"))
    if pcol in master.columns and dcol in master.columns and scol in master.columns:
        master["both_replications_direction_concordant"]=(master["PPMI_PD_direction_concordant"] & master["PPMI_SU_direction_concordant"])
    if pcol in master.columns and poolcol in master.columns:
        master["PPMI_PD_PLUS_SU_direction_concordant"]=np.sign(pd.to_numeric(master[pcol],errors="coerce"))==np.sign(pd.to_numeric(master[poolcol],errors="coerce"))

    # Optional AR annotations. Merge only; never use subset-run q-values for selection.
    if a.ar_detail_file and Path(a.ar_detail_file).exists():
        ar=_normalize_ids(_read_table(Path(a.ar_detail_file)),"AR detail")
        # If multiple lags exist, prefer the original discovery-selected lag when available.
        lag_master=next((c for c in ["PPMI_lag_order","PPMI_best_lag","PPMI_lag"] if c in master.columns),None)
        lag_ar=next((c for c in ["lag_order","best_lag","lag"] if c in ar.columns),None)
        if lag_master and lag_ar:
            tmp=master[keys+[lag_master]].rename(columns={lag_master:"_orig_lag"})
            ar=ar.merge(tmp,on=keys,how="inner")
            ar=ar[pd.to_numeric(ar[lag_ar],errors="coerce")==pd.to_numeric(ar["_orig_lag"],errors="coerce")].copy()
        ar=_dedup_pair(ar,"AR detail selected lag")
        master=master.merge(_prefix_payload(ar,"AR"),on=keys,how="left")

    # Optional gene-level multi-IV annotations.
    if a.multi_iv_summary and Path(a.multi_iv_summary).exists():
        miv=_read_table(Path(a.multi_iv_summary))
        g=next((c for c in ID_GENE if c in miv.columns),None)
        if g:
            if g!="gene":miv=miv.rename(columns={g:"gene"})
            miv["gene"]=miv["gene"].astype(str)
            miv=miv.drop_duplicates("gene")
            miv=miv.rename(columns={c:f"multiIV_{c}" for c in miv.columns if c!="gene"})
            master=master.merge(miv,on="gene",how="left")

    # Optional LD annotation. Gene-level or pair-level accepted.
    if a.ld_audit_file and Path(a.ld_audit_file).exists():
        ld=_read_table(Path(a.ld_audit_file))
        g=next((c for c in ID_GENE if c in ld.columns),None)
        s=next((c for c in ID_SNP if c in ld.columns),None)
        if g:
            if g!="gene":ld=ld.rename(columns={g:"gene"})
            ld["gene"]=ld["gene"].astype(str)
            if s:
                if s!="lead_snp":ld=ld.rename(columns={s:"lead_snp"})
                ld["lead_snp"]=ld["lead_snp"].astype(str)
                ld=ld.drop_duplicates(keys)
                ld=ld.rename(columns={c:f"LD_{c}" for c in ld.columns if c not in keys})
                master=master.merge(ld,on=keys,how="left")
            else:
                ld=ld.drop_duplicates("gene")
                ld=ld.rename(columns={c:f"LD_{c}" for c in ld.columns if c!="gene"})
                master=master.merge(ld,on="gene",how="left")

    # Critically: selected lag may differ from discovery pair-best lag. Synchronize
    # all primary discovery and replication model payloads to the SAME selected lag.
    # This ensures 08 robustness evaluates and audits the selected model correctly.
    modelkey=keys+["lag_order"]
    disc_lag=_read_table(base/"discovery_significant_IV_gene_lag_models.csv")
    if disc_lag.duplicated(modelkey).any():raise ValueError("Duplicated discovery pair x lag")
    selected_lag=selected[keys+["lag_order"]].rename(columns={"lag_order":"selected_lag_order"})
    master=master.merge(selected_lag,on=keys,how="left",validate="one_to_one")
    for col in ["PPMI_lag_order","PD_lag_order","SU_lag_order","PD_PLUS_SU_lag_order"]:
        # Do not let 08 use unrelated, best-lag models from other cohorts.
        if col in master:master[col]=master.selected_lag_order if col==("PD_lag_order" if rep_cohort=="PD" else "PD_PLUS_SU_lag_order") or col=="PPMI_lag_order" else np.nan
    # Correct PPMI payload at the selected lag, because discovery best lag can differ.
    payload=disc_lag.merge(selected[keys+["lag_order"]],on=modelkey,how="inner",validate="one_to_one")
    rename={c:"PPMI_"+c for c in payload if c not in keys}
    for c in rename.values():
        if c in master:master=master.drop(columns=c)
    master=master.merge(payload.rename(columns=rename),on=keys,how="left",validate="one_to_one")
    # Import raw PD and SU replication models at the EXACT newly selected lag.
    # Old PRIMARY files are lag1-gated and would have missing coefficients for
    # newly restored coef_sum-only candidates. This does NOT change selection.
    for co in ("PD", "SU") if rep_cohort=="PD+SU" else ():
        options=[base/f"replication_{co}_all_lags_annotated.csv.gz",
                 base/f"replication_{co}_all_lags_annotated.csv"]
        rawfile=next((f for f in options if f.is_file()),None)
        if rawfile is None:
            raise FileNotFoundError(f"{co}: raw all-lags file needed to report selected-lag replication coefficients: {options}")
        raw=_read_table(rawfile)
        modelkey=keys+["lag_order"]
        if raw.duplicated(modelkey).any():raise ValueError(f"{rawfile}: duplicate pair x lag rows")
        raw=raw.merge(selected[modelkey],on=modelkey,how="inner",validate="one_to_one")
        raw=raw.rename(columns={c:f"{co}_{c}" for c in raw if c not in keys})
        master=master.drop(columns=[c for c in master if c.startswith(f"{co}_")],errors="ignore")
        master=master.merge(raw,on=keys,how="left",validate="one_to_one")
        master[f"{co}_lag_order"]=master["selected_lag_order"]
    # Rejoin exact replication best-BIC row and update selected-cohort payload.
    rep_prefix="PD_" if rep_cohort=="PD" else "PD_PLUS_SU_"
    payload=selected.drop(columns=["discovery_coef_lag1","lag1_direction_concordant","discovery_coef_sum","coef_sum_direction_concordant"],errors="ignore")
    payload=payload.rename(columns={c:rep_prefix+c for c in payload if c not in keys})
    for c in payload.columns:
        if c in master and c not in keys:master=master.drop(columns=c)
    master=master.merge(payload,on=keys,how="left",validate="one_to_one")
    master["lag1_direction_concordant"]=(pd.to_numeric(master["PPMI_coef_lag1"],errors="coerce") * pd.to_numeric(master[("PD_coef_lag1" if rep_cohort=="PD" else "PD_PLUS_SU_coef_lag1")],errors="coerce")>0)
    master["coef_sum_direction_concordant"]=(pd.to_numeric(master["PPMI_coef_sum"],errors="coerce") * pd.to_numeric(master[("PD_coef_sum" if rep_cohort=="PD" else "PD_PLUS_SU_coef_sum")],errors="coerce")>0)
    if not master.coef_sum_direction_concordant.all():raise ValueError("Selected pair coef_sum reconciliation failed")
    # Reconcile every displayed direction flag to the SAME selected-lag models.
    for co in ("PD", "SU", "PD_PLUS_SU"):
        cc=f"{co}_coef_sum"
        if cc in master:
            master[f"PPMI_{co}_direction_concordant"]=(
                pd.to_numeric(master["PPMI_coef_sum"],errors="coerce") *
                pd.to_numeric(master[cc],errors="coerce")>0)
    master["replication_design"]="single_PD" if rep_cohort=="PD" else "pooled_PD_PLUS_SU"
    master["replication_cohort_for_primary"]=rep_cohort
    master["both_replications_direction_concordant"]=master["coef_sum_direction_concordant"]
    # Count how many independently replicated pairs support each gene.
    gc=master.groupby("gene",as_index=False).agg(n_replicated_IV_gene_pairs=("lead_snp","nunique"))
    master=master.merge(gc,on="gene",how="left")
    master=master.sort_values(["gene","lead_snp"]).reset_index(drop=True)
    master.to_csv(out/f"{a.analysis_name}_downstream_IV_gene_master.csv",index=False)

    # Clean Stage-I inputs for optional targeted AR/full-detail rerun: exact candidate pairs only.
    clean_exact=stage1.merge(candidate_keys[keys],on=keys,how="inner")
    clean_exact.to_csv(out/f"{a.analysis_name}_candidate_exact_pairs_clean_stage1_input.csv",index=False)

    # All Stage-I eligible IVs for genes in the candidate set: for multi-IV consistency only.
    genes=sorted(master.gene.unique())
    all_gene_ivs=stage1[stage1.gene.isin(genes)].copy().sort_values(["gene","lead_snp"])
    all_gene_ivs.to_csv(out/f"{a.analysis_name}_candidate_genes_all_stage1_eligible_IVs.csv",index=False)

    # Gene summary.
    gene_summary=master.groupby("gene",as_index=False).agg(
        n_replicated_IV_gene_pairs=("lead_snp","nunique"),
        n_pooled_supported_pairs=("pooled_PD_PLUS_SU_support","sum")
    )
    if "both_replications_direction_concordant" in master.columns:
        z=master.groupby("gene")["both_replications_direction_concordant"].agg(["sum","count"]).reset_index()
        z=z.rename(columns={"sum":"n_direction_concordant_pairs","count":"n_pairs_with_direction_evaluable"})
        gene_summary=gene_summary.merge(z,on="gene",how="left")
    gene_summary=gene_summary.sort_values(["n_replicated_IV_gene_pairs","gene"],ascending=[False,True])
    gene_summary.to_csv(out/f"{a.analysis_name}_downstream_gene_summary.csv",index=False)

    # Optional one-IV-per-gene file for plotting only; never the primary downstream candidate set.
    rank=master.copy()
    qcols=[c for c in rank.columns if c in {"PPMI_q_bh_global","PPMI_discovery_q_bh_global","PPMI_best_q_bh_global"}]
    pcols=[c for c in rank.columns if c in {"PPMI_pvalue","PPMI_discovery_pvalue","PPMI_best_pvalue"}]
    bcols=[c for c in rank.columns if c in {"PPMI_bic","PPMI_BIC","PPMI_discovery_bic","PPMI_best_bic"}]
    sort=["gene"];asc=[True]
    for cols in [qcols,pcols,bcols]:
        if cols:
            c=cols[0];rank[c]=pd.to_numeric(rank[c],errors="coerce");sort.append(c);asc.append(True)
    sort.append("lead_snp");asc.append(True)
    representative=rank.sort_values(sort,ascending=asc,na_position="last").drop_duplicates("gene",keep="first")
    representative["representative_only_not_primary_filter"]=True
    representative.to_csv(out/f"{a.analysis_name}_OPTIONAL_one_representative_IV_per_gene.csv",index=False)

    # Funnel.
    funnel=[]
    funnel.append(_count_row("Stage-I eligible replicated/direction-consistent/LD-pruned temporal IV-gene pairs",stage1))
    funnel.append(_count_row("PPMI discovery global-BH significant IV-gene pairs",discovery))
    funnel.append(_count_row("PD nominal primary replication",pdrep))
    funnel.append(_count_row("SU nominal primary replication",surep))
    funnel.append(_count_row(f"{rep_cohort} nominal at PPMI significant lag and coef_sum sign: PRIMARY DOWNSTREAM SET",master))
    pooled=master[master.pooled_PD_PLUS_SU_support]
    funnel.append(_count_row("Primary downstream set also supported in pooled PD+SU (annotation only)",pooled))
    funnel_df=pd.DataFrame(funnel)
    funnel_df.to_csv(out/f"{a.analysis_name}_selection_funnel.csv",index=False)

    manifest={
        "analysis_name":a.analysis_name,
        "stage13_dir":str(base),
        "selection_rule":f"PPMI global BH discovery plus {rep_cohort} at same significant lag, best replication BIC and coef_sum sign concordance.",
        "pooled_PD_PLUS_SU":"PRIMARY replication for UPDRS; MoCA primary replication is PD.",
        "not_selection_gates":["first-stage F","partial R2","Anderson-Rubin-type P","Cochran Q","I2","leave-one-IV-out","cross-gene LD annotation","pooled PD+SU P"],
        "n_primary_downstream_pairs":int(len(master)),
        "n_primary_downstream_genes":int(master.gene.nunique()),
        "n_primary_pairs_with_pooled_support":int(master.pooled_PD_PLUS_SU_support.sum()),
        "n_candidate_genes_with_multiple_replicated_IVs":int((gc.n_replicated_IV_gene_pairs>1).sum()),
        "n_all_stage1_eligible_IVs_for_candidate_genes":int(len(all_gene_ivs)),
        "files":{
            "master":str(out/f"{a.analysis_name}_downstream_IV_gene_master.csv"),
            "gene_summary":str(out/f"{a.analysis_name}_downstream_gene_summary.csv"),
            "funnel":str(out/f"{a.analysis_name}_selection_funnel.csv"),
            "clean_exact_candidate_input":str(out/f"{a.analysis_name}_candidate_exact_pairs_clean_stage1_input.csv"),
            "all_stage1_IVs_candidate_genes":str(out/f"{a.analysis_name}_candidate_genes_all_stage1_eligible_IVs.csv"),
            "optional_representative":str(out/f"{a.analysis_name}_OPTIONAL_one_representative_IV_per_gene.csv")
        }
    }
    with open(out/f"{a.analysis_name}_standard_final_manifest.json","w") as f:json.dump(manifest,f,indent=2)

    print("="*90)
    print("STANDARD TEMPORAL-CAUSAL FINALIZATION")
    print("="*90)
    print(funnel_df.to_string(index=False))
    print("\nPrimary downstream candidate set:")
    print(f"  {len(master)} IV-gene pairs; {master.gene.nunique()} genes")
    print(f"  pooled PD+SU support: {int(master.pooled_PD_PLUS_SU_support.sum())}/{len(master)} pairs (annotation only)")
    print(f"  genes with >1 independently replicated IV: {int((gc.n_replicated_IV_gene_pairs>1).sum())}")
    print(f"  all Stage-I eligible IVs across candidate genes for multi-IV sensitivity: {len(all_gene_ivs)}")
    print(f"\nSaved to: {out}")

if __name__=="__main__":main()
