#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,subprocess
from pathlib import Path
import pandas as pd

def read_cov(path):
    x=pd.read_csv(path,low_memory=False);x=x.drop(columns=[c for c in x.columns if str(c).startswith("Unnamed:")],errors="ignore")
    for c in ["cohort","participant_id","time"]:
        if c not in x.columns:raise ValueError(f"Missing phenotype column {c}")
    x["cohort"]=x.cohort.astype(str);x["participant_id"]=x.participant_id.astype(str);return x

def read_fam(prefix):return pd.read_csv(str(prefix)+".fam",sep=r"\s+",header=None,names=["FID","IID","PAT","MAT","SEX","PHENO"],dtype=str)

def match(cov,fam):
    fi={str(x):i for i,x in enumerate(fam.FID)};ii={str(x):i for i,x in enumerate(fam.IID)};rows=[]
    for pid in cov.participant_id.drop_duplicates().astype(str):
        if pid in ii:r=fam.iloc[ii[pid]];how="IID"
        elif pid in fi:r=fam.iloc[fi[pid]];how="FID"
        else:rows.append({"participant_id":pid,"genotype_match":"missing","FID":"","IID":""});continue
        rows.append({"participant_id":pid,"genotype_match":how,"FID":str(r.FID),"IID":str(r.IID)})
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--root",default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt");ap.add_argument("--analysis-name",default="updrs3");ap.add_argument("--phenotype-cov",required=True);ap.add_argument("--candidate-file",default="");ap.add_argument("--raw-bfile",default="/public/labdata/public_data/AMP_PD_WGS/plink/all_chrs_merged");ap.add_argument("--plink",default="/usr/bin/plink1.9");ap.add_argument("--extract-genotype",action=argparse.BooleanOptionalAction,default=True);a=ap.parse_args()
    root=Path(a.root);out=root/"single_iv_causal"/a.analysis_name;gdir=out/"genotype";gdir.mkdir(parents=True,exist_ok=True)
    cf=Path(a.candidate_file) if a.candidate_file else root/"single_iv_causal"/"iv_validation"/"single_iv_primary_eligible.csv"
    if not cf.exists():raise FileNotFoundError(cf)
    cand=pd.read_csv(cf,low_memory=False)
    if not {"gene","lead_snp"}.issubset(cand.columns):raise ValueError("candidate file must contain gene, lead_snp")
    cov=read_cov(a.phenotype_cov);fam=read_fam(a.raw_bfile);m=match(cov,fam);cc=cov[["cohort","participant_id"]].drop_duplicates().merge(m,on="participant_id",how="left");cc.to_csv(out/"participant_genotype_match.csv",index=False)
    summary=cc.groupby("cohort").apply(lambda z:pd.Series({"participants":z.participant_id.nunique(),"genotype_matched":z.loc[z.genotype_match.ne("missing"),"participant_id"].nunique(),"genotype_missing":z.loc[z.genotype_match.eq("missing"),"participant_id"].nunique()}),include_groups=False).reset_index();summary.to_csv(out/"participant_overlap_summary.csv",index=False)
    snps=sorted(cand.lead_snp.dropna().astype(str).unique());(gdir/"needed_single_iv_snps.txt").write_text("\n".join(snps)+"\n")
    keep=cc[cc.genotype_match.ne("missing")][["FID","IID"]].drop_duplicates();keep.to_csv(gdir/"keep_single_iv_stage2.txt",sep="\t",header=False,index=False)
    manifest={"analysis_name":a.analysis_name,"phenotype_cov":a.phenotype_cov,"candidate_file":str(cf),"raw_bfile":a.raw_bfile,"n_candidate_genes":int(cand.gene.nunique()),"n_lead_snps":len(snps)}
    if a.extract_genotype:
        pref=gdir/"single_iv_snps";cmd=[a.plink,"--bfile",a.raw_bfile,"--keep",str(gdir/"keep_single_iv_stage2.txt"),"--extract",str(gdir/"needed_single_iv_snps.txt"),"--make-bed","--out",str(pref)];subprocess.run(cmd,check=True);manifest["single_iv_bfile"]=str(pref);manifest["extracted_participants"]=int(len(read_fam(pref)));bim=pd.read_csv(str(pref)+".bim",sep=r"\s+",header=None,dtype=str);manifest["extracted_snps"]=int(len(bim));manifest["missing_requested_snps"]=int(len(snps)-len(bim))
    (out/"single_iv_input_audit.json").write_text(json.dumps(manifest,indent=2));print(summary.to_string(index=False));print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
