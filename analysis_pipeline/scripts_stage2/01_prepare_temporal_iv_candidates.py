#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gengps.amp_pd.genotype import make_genotype_source
from gengps.amp_pd.single_iv_causal import normalize_qtl_table,find_first_existing

def _artifact_id(gene,snp):return hashlib.sha1(f"{gene}|{snp}".encode()).hexdigest()[:16]
def _read_fam_ids(prefix):
    fam=pd.read_csv(str(prefix)+".fam",sep=r"\s+",header=None,dtype=str)
    return fam.iloc[:,1].astype(str).tolist()
def _default_pp_bfile(root):
    for p in [root/"data/genotype/PP/amp_pd_PP_qc",root/"data/genotype/PP/amp_pd_PP",root/"data/stage1_manuscript/genotype/PP/amp_pd_PP_qc"]:
        if Path(str(p)+".bed").exists():return str(p)
    return ""
def find_pdbp_qtl(root,label,user=""):
    if user:
        p=Path(user)
        if not p.exists():raise FileNotFoundError(p)
        return p
    c=[]
    for cohort_dir in ["PD","PDBP","REP"]:
        base=root/"qtl"/cohort_dir/label
        c += [base/"final"/"all_results.parquet",base/"all_results.parquet",base/"results"/"all_results.parquet"]
    return find_first_existing(c)
def _find_ppmi_tqtl(root,label):
    base=root/"qtl"/"PP"/label/"final"
    p=find_first_existing([base/"significant_tqtl.parquet",base/"significant_temporal.parquet",base/"significant_union.parquet"])
    if p is None:raise FileNotFoundError(f"No finalized PPMI significant temporal-QTL table under {base}")
    return p
def _read_ppmi_tqtl(path):
    raw=pd.read_parquet(path);x=normalize_qtl_table(raw);x=x[np.isfinite(x.qtl_p)].copy();x=x.rename(columns={"qtl_p":"ppmi_tqtl_p"})
    fdr_col=next((c for c in ["FDR","Temporal_FDR","temporal_fdr","q_value","qvalue"] if c in x.columns),None)
    if fdr_col is not None:
        x["ppmi_temporal_fdr"]=pd.to_numeric(x[fdr_col],errors="coerce")
        bad=np.isfinite(x.ppmi_temporal_fdr)&(x.ppmi_temporal_fdr>=0.05)
        if bad.any():raise RuntimeError(f"{path} contains {int(bad.sum())} rows with temporal FDR >= 0.05; expected a finalized significant_tqtl table")
    else:x["ppmi_temporal_fdr"]=np.nan
    intp=next((c for c in ["Interaction_Pvalue","interaction_pvalue","interaction_p","P_interaction","p_interaction"] if c in x.columns),None)
    intf=next((c for c in ["Interaction_FDR","interaction_fdr","interaction_q","Interaction_qvalue"] if c in x.columns),None)
    x["ppmi_interaction_p"]=pd.to_numeric(x[intp],errors="coerce") if intp else np.nan
    x["ppmi_interaction_fdr"]=pd.to_numeric(x[intf],errors="coerce") if intf else np.nan
    keep=["gene","lead_snp","ppmi_tqtl_p","ppmi_temporal_fdr","ppmi_interaction_p","ppmi_interaction_fdr"]
    for c in ["basis_json"]+[c for c in x.columns if str(c).startswith("beta_g_")]:
        if c in x.columns and c not in keep:keep.append(c)
    y=x[keep].copy()
    if y.duplicated(["gene","lead_snp"]).any():raise RuntimeError(f"Duplicate SNP-gene pairs in {path}")
    return y
def _merge_pdbp(cand,pq):
    raw=pd.read_parquet(pq);px=normalize_qtl_table(raw);px=px.rename(columns={"qtl_p":"pdbp_tqtl_p"})
    intp=next((c for c in ["Interaction_Pvalue","interaction_pvalue","interaction_p","P_interaction","p_interaction"] if c in px.columns),None)
    px["pdbp_interaction_p"]=pd.to_numeric(px[intp],errors="coerce") if intp else np.nan
    keep=["gene","lead_snp","pdbp_tqtl_p","pdbp_interaction_p"]
    for c in [c for c in px.columns if str(c).startswith("beta_g_")]:
        px=px.rename(columns={c:f"pdbp_{c}"});keep.append(f"pdbp_{c}")
    px=px[keep].drop_duplicates(["gene","lead_snp"])
    return cand.merge(px,on=["gene","lead_snp"],how="left",validate="one_to_one")
def _merge_direction(cand,dfile):
    dr=pd.read_parquet(dfile).copy()
    rename={"Variant_ID":"lead_snp","Ensembl_ID":"gene","trajectory_r":"pdbp_direction_value","trajectory_direction_concordant":"pdbp_direction_concordant"}
    dr=dr.rename(columns=rename)
    req={"gene","lead_snp","pdbp_direction_value"}
    if not req.issubset(dr.columns):raise ValueError(f"{dfile} missing {sorted(req-set(dr.columns))}")
    if "pdbp_direction_concordant" not in dr.columns:dr["pdbp_direction_concordant"]=pd.to_numeric(dr.pdbp_direction_value,errors="coerce")>0
    dr=dr[["gene","lead_snp","pdbp_direction_value","pdbp_direction_concordant"]].drop_duplicates(["gene","lead_snp"])
    out=cand.merge(dr,on=["gene","lead_snp"],how="left",validate="one_to_one")
    out["pdbp_direction_metric"]="allele_harmonized_trajectory_r"
    return out
def _greedy_ld_prune(x,bfile,plink,r2thr):
    if x.empty:return x.assign(ld_independent=False,ld_prune_rank=np.nan,ld_prune_max_r2_to_kept=np.nan)
    gsrc=make_genotype_source(str(bfile),plink,prefer_bed_reader=True);ids=_read_fam_ids(bfile);out=[]
    for gene,z in x.groupby("gene",sort=False):
        z=z.sort_values(["ppmi_tqtl_p","lead_snp"],kind="mergesort").copy();snps=z.lead_snp.astype(str).tolist()
        if len(snps)==1:
            row=z.iloc[0].to_dict();row.update(ld_independent=True,ld_prune_rank=1,ld_prune_max_r2_to_kept=0.0);out.append(row);continue
        try:
            G=np.asarray(gsrc.read(ids,snps,mean_impute=True,dtype=np.float64),float)
            if G.shape!=(len(ids),len(snps)):raise ValueError(f"genotype shape {G.shape} != {(len(ids),len(snps))}")
            C=np.corrcoef(G,rowvar=False);R2=np.square(np.nan_to_num(C,nan=0.0,posinf=0.0,neginf=0.0));np.fill_diagonal(R2,1.0);kept=[]
            for j in range(len(snps)):
                maxr=max([float(R2[j,k]) for k in kept],default=0.0);row=z.iloc[j].to_dict();row["ld_prune_max_r2_to_kept"]=maxr
                if maxr<float(r2thr):kept.append(j);row["ld_independent"]=True;row["ld_prune_rank"]=len(kept)
                else:row["ld_independent"]=False;row["ld_prune_rank"]=np.nan
                out.append(row)
        except Exception as e:raise RuntimeError(f"LD pruning failed for {gene}: {e}") from e
    return pd.DataFrame(out)
def main():
    ap=argparse.ArgumentParser(description="Unified temporal-tQTL IV selection: PPMI temporal FDR<0.05 -> same SNP-gene PDBP temporal P<0.05 -> trajectory_r>0 -> PPMI LD pruning. Genotype-by-time interaction is annotation only.")
    ap.add_argument("--root",default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt");ap.add_argument("--qtl-label",default="submitted_followup_ctar_mptga");ap.add_argument("--pdbp-qtl-table",default="");ap.add_argument("--ppmi-bfile",default="");ap.add_argument("--plink",default="/usr/bin/plink1.9");ap.add_argument("--ld-r2-threshold",type=float,default=0.1);ap.add_argument("--trajectory-direction-file",default="");ap.add_argument("--stage1-replication-summary",default="");ap.add_argument("--out-dir",default="")
    a=ap.parse_args();root=Path(a.root);out=Path(a.out_dir) if a.out_dir else root/"single_iv_causal"/"iv_validation";out.mkdir(parents=True,exist_ok=True)
    ppfile=_find_ppmi_tqtl(root,a.qtl_label);cand=_read_ppmi_tqtl(ppfile);cand["artifact_id"]=[_artifact_id(g,s) for g,s in zip(cand.gene.astype(str),cand.lead_snp.astype(str))];cand.to_csv(out/"single_iv_01_ppmi_significant_tqtl.csv",index=False)
    pq=find_pdbp_qtl(root,a.qtl_label,a.pdbp_qtl_table)
    if pq is None:raise FileNotFoundError("Could not find PDBP all_results.parquet; pass --pdbp-qtl-table")
    cand=_merge_pdbp(cand,pq);cand["pdbp_tqtl_rep_nominal"]=pd.to_numeric(cand.pdbp_tqtl_p,errors="coerce")<0.05;cand["pdbp_interaction_p_lt_0_05_annotation"]=pd.to_numeric(cand.pdbp_interaction_p,errors="coerce")<0.05
    rep_summary=Path(a.stage1_replication_summary) if a.stage1_replication_summary else root/"qtl"/"replication_summary.csv"
    stage1_expected=np.nan
    if rep_summary.exists():
        rs=pd.read_csv(rep_summary);keycol="result_label" if "result_label" in rs.columns else ("method" if "method" in rs.columns else None)
        if keycol is not None:
            rr=rs[rs[keycol].astype(str).eq(str(a.qtl_label))]
            if rr.empty and "method" in rs.columns:rr=rs[rs["method"].astype(str).eq(str(a.qtl_label))]
            if len(rr)==1 and "n_tqtl_rep_nominal" in rr.columns and pd.notna(rr.iloc[0]["n_tqtl_rep_nominal"]):
                stage1_expected=int(rr.iloc[0]["n_tqtl_rep_nominal"]);got=int(cand.pdbp_tqtl_rep_nominal.fillna(False).sum())
                if got!=stage1_expected:raise RuntimeError(f"Stage-I temporal-tQTL replication count mismatch: candidate script={got}, replication_summary.csv={stage1_expected}. Check the PDBP all_results path before LD pruning.")
    dfile=Path(a.trajectory_direction_file) if a.trajectory_direction_file else root/"qtl"/"trajectory_direction_consistency"/a.qtl_label/"trajectory_direction_pairs.parquet"
    if not dfile.exists():raise FileNotFoundError(f"Trajectory direction file not found: {dfile}. Run scripts_stage2/00_trajectory_direction.py first.")
    cand=_merge_direction(cand,dfile);cand["pdbp_direction_concordant"]=cand.pdbp_direction_concordant.fillna(False).astype(bool);cand["iv_pre_ld_eligible"]=cand.pdbp_tqtl_rep_nominal.fillna(False).astype(bool)&cand.pdbp_direction_concordant
    cand["iv_pre_ld_reason"]=np.where(cand.iv_pre_ld_eligible,"PPMI_temporal_FDR<0.05+PDBP_temporal_P<0.05+trajectory_r>0","failed_replication_or_direction")
    cand.to_csv(out/"single_iv_02_replication_direction_audit.csv",index=False)
    valid=cand[cand.iv_pre_ld_eligible].copy();valid.to_csv(out/"single_iv_03_replication_direction_pass.csv",index=False)
    bfile=a.ppmi_bfile or _default_pp_bfile(root)
    if not bfile:raise FileNotFoundError("Could not locate PPMI PLINK bfile; pass --ppmi-bfile")
    pruned=_greedy_ld_prune(valid,bfile,a.plink,a.ld_r2_threshold);pruned.to_csv(out/"single_iv_04_ld_pruning_all_rows.csv",index=False);final=pruned[pruned.ld_independent.fillna(False).astype(bool)].copy();final["iv_primary_eligible"]=True;final["iv_eligibility_reason"]="PPMI_temporal_FDR<0.05+PDBP_temporal_P<0.05+trajectory_r>0+PPMI_LD_independent";final=final.sort_values(["gene","ppmi_tqtl_p","lead_snp"]);final.to_csv(out/"single_iv_all_independent_candidates.csv",index=False);final.to_csv(out/"single_iv_primary_eligible.csv",index=False)
    repmask=cand.pdbp_tqtl_rep_nominal.fillna(False);dirmask=cand.pdbp_direction_concordant.fillna(False);summary=pd.DataFrame([{"n_ppmi_significant_tqtl":int(len(cand)),"n_pdbp_matched_with_finite_p":int(np.isfinite(pd.to_numeric(cand.pdbp_tqtl_p,errors="coerce")).sum()),"n_pdbp_temporal_p_lt_0_05":int(repmask.sum()),"stage1_expected_n_tqtl_rep_nominal":stage1_expected,"n_direction_concordant_among_pdbp_rep":int((repmask&dirmask).sum()),"direction_concordance_rate_among_pdbp_rep":float(cand.loc[repmask,"pdbp_direction_concordant"].mean()) if repmask.any() else np.nan,"n_rep_direction_pass_before_ld":int(len(valid)),"n_after_ld":int(len(final)),"n_genes_after_ld":int(final.gene.nunique()),"ld_r2_threshold":float(a.ld_r2_threshold)}]);summary.to_csv(out/"single_iv_candidate_filter_summary.csv",index=False)
    manifest={"qtl_label":a.qtl_label,"architecture":"one unified temporal-tQTL instrument family; no dynamic/additive causal arms; every LD-independent IV-gene pair is analyzed separately; no multi-SNP expression predictor","selection_order":["PPMI finalized significant temporal cis-eQTL (global temporal BH-FDR<0.05)","same SNP-gene PDBP temporal P<0.05","allele-harmonized PPMI-PDBP trajectory_r>0","PPMI LD pruning within gene among replicated+direction-concordant pairs"],"interaction_role":"PPMI/PDBP genotype-by-time interaction statistics are retained only as annotation and never determine causal-IV eligibility","ld_pruning":{"cohort":"PPMI","r2_threshold":a.ld_r2_threshold,"selection_order":"ascending PPMI temporal P within gene after replication and direction filtering"},"ppmi_tqtl_file":str(ppfile),"pdbp_qtl_file":str(pq),"stage1_replication_summary":str(rep_summary) if rep_summary.exists() else "","stage1_expected_n_tqtl_rep_nominal":None if not np.isfinite(stage1_expected) else int(stage1_expected),"trajectory_direction_file":str(dfile),"strength":"Stage-II step 02 computes F/R2/rho diagnostics only and never changes primary eligibility","n_final_IV_pairs":int(len(final)),"n_final_genes":int(final.gene.nunique())};(out/"single_iv_manifest.json").write_text(json.dumps(manifest,indent=2));print("\n=== UNIFIED TEMPORAL-tQTL FILTER SUMMARY ===");print(summary.to_string(index=False));print("\n=== MANIFEST ===");print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
