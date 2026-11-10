#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd


def bim(path):
    x=pd.read_csv(str(path)+".bim",sep=r"\s+",header=None,names=["CHR","SNP","CM","BP","A1","A2"],dtype={"CHR":str,"SNP":str,"BP":int})
    return x


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt");ap.add_argument("--analysis-name",default="updrs3");ap.add_argument("--candidate-file",default="");ap.add_argument("--raw-bfile",default="/public/labdata/public_data/AMP_PD_WGS/plink/all_chrs_merged");ap.add_argument("--plink",default="/usr/bin/plink1.9");ap.add_argument("--window-kb",type=int,default=2000);ap.add_argument("--report-r2",type=float,default=0.1);ap.add_argument("--flag-r2",type=float,default=0.2);ap.add_argument("--out-dir",default="");a=ap.parse_args()
    root=Path(a.root);out=Path(a.out_dir) if a.out_dir else root/"single_iv_causal"/a.analysis_name/"locus_validation";out.mkdir(parents=True,exist_ok=True)
    cf=Path(a.candidate_file) if a.candidate_file else root/"single_iv_causal"/a.analysis_name/"temporal_causal"/"final_IV_pair_summary.csv"
    if not cf.exists():cf=root/"single_iv_causal"/"iv_validation"/"single_iv_primary_eligible.csv"
    x=pd.read_csv(cf,low_memory=False)
    if not {"gene","lead_snp"}.issubset(x.columns):raise ValueError(f"{cf} must contain gene and lead_snp")
    x=x[["gene","lead_snp"]].dropna().astype(str).drop_duplicates();b=bim(a.raw_bfile);m=x.merge(b.rename(columns={"SNP":"lead_snp"}),on="lead_snp",how="left")
    m.to_csv(out/"lead_iv_positions.csv",index=False);missing=m[m.BP.isna()].copy();missing.to_csv(out/"lead_iv_missing_from_raw_bim.csv",index=False)
    present=m.dropna(subset=["BP"]);snps=sorted(present.lead_snp.unique());sl=out/"lead_iv_snps.txt";sl.write_text("\n".join(snps)+"\n")
    if len(snps)>=2:
        pref=out/"lead_iv_pairwise_ld";cmd=[a.plink,"--bfile",a.raw_bfile,"--extract",sl,"--r2","--ld-window",999999,"--ld-window-kb",a.window_kb,"--ld-window-r2",a.report_r2,"--out",pref];print("+"," ".join(map(str,cmd)),flush=True);subprocess.run(list(map(str,cmd)),check=True)
        ldfile=Path(str(pref)+".ld")
        ld=pd.read_csv(ldfile,sep=r"\s+",dtype={"SNP_A":str,"SNP_B":str,"CHR_A":str,"CHR_B":str}) if ldfile.exists() and ldfile.stat().st_size else pd.DataFrame()
    else:ld=pd.DataFrame()
    if not ld.empty:
        gm=x.groupby("lead_snp").gene.apply(lambda s:";".join(sorted(set(s.astype(str))))).to_dict();ld["genes_A"]=ld.SNP_A.map(gm);ld["genes_B"]=ld.SNP_B.map(gm);ld["high_ld_flag"]=pd.to_numeric(ld.R2,errors="coerce")>=a.flag_r2;ld.to_csv(out/"lead_iv_ld_pairs.csv",index=False)
        high=ld[ld.high_ld_flag].copy();genes=set()
        for c in ["genes_A","genes_B"]:
            for z in high[c].dropna():genes.update(str(z).split(";"))
        flags=x.copy();flags["lead_iv_high_ld_with_other_candidate"]=flags.gene.isin(genes);flags.to_csv(out/"lead_iv_gene_ld_flags.csv",index=False)
    else:
        pd.DataFrame(columns=["CHR_A","BP_A","SNP_A","CHR_B","BP_B","SNP_B","R2","genes_A","genes_B","high_ld_flag"]).to_csv(out/"lead_iv_ld_pairs.csv",index=False);flags=x.copy();flags["lead_iv_high_ld_with_other_candidate"]=False;flags.to_csv(out/"lead_iv_gene_ld_flags.csv",index=False)
    manifest={"candidate_file":str(cf),"purpose":"Reviewer-3 locus audit: identify candidate genes whose lead single-IV SNPs are in local LD with lead IVs of other prioritized genes. This does not replace formal colocalization.","window_kb":a.window_kb,"report_r2":a.report_r2,"flag_r2":a.flag_r2,"n_genes":int(x.gene.nunique()),"n_lead_snps":len(snps),"n_high_ld_pairs":int(flags.lead_iv_high_ld_with_other_candidate.sum()) if "lead_iv_high_ld_with_other_candidate" in flags else 0};(out/"ld_audit_manifest.json").write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))

if __name__=="__main__":main()
