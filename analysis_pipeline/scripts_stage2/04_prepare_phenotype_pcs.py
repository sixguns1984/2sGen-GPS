#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
import json
import math
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

def csv_list(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]

def run(cmd,log_file=None):
    print("+"," ".join(map(str,cmd)),flush=True)
    if log_file is None:
        subprocess.run(list(map(str,cmd)),check=True)
    else:
        p=subprocess.run(list(map(str,cmd)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        Path(log_file).write_text(p.stdout or "")
        if p.returncode!=0:
            raise RuntimeError(f"Command failed ({p.returncode}); see {log_file}: {' '.join(map(str,cmd))}")

def read_match(path):
    x=pd.read_csv(path,dtype=str)
    req=["cohort","participant_id","genotype_match","FID","IID"]
    miss=[c for c in req if c not in x.columns]
    if miss: raise ValueError(f"{path} missing columns {miss}")
    x=x[x["genotype_match"].ne("missing")].copy()
    x["cohort"]=x["cohort"].astype(str)
    x["participant_id"]=x["participant_id"].astype(str)
    x["FID"]=x["FID"].astype(str);x["IID"]=x["IID"].astype(str)
    return x.drop_duplicates(["cohort","participant_id"])

def fingerprint(df,raw_bfile,alpha,max_pcs,qc,prune):
    z=df[["cohort","participant_id","FID","IID"]].sort_values(["cohort","participant_id"]).to_csv(index=False)
    payload={"participants":z,"raw_bfile":str(raw_bfile),"alpha":alpha,"max_pcs":max_pcs,"qc":qc,"prune":prune}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()

def parse_twstats(path,max_pcs):
    rows=[]
    with open(path,"r",errors="replace") as fh:
        for line in fh:
            s=line.strip()
            if not s or s.startswith("#"): continue
            tok=s.split()
            if len(tok)<5: continue
            try:
                pc=int(float(tok[0]))
            except Exception:
                continue
            if pc<1 or pc>int(max_pcs): continue
            vals=[]
            for t in tok[1:]:
                try: vals.append(float(t))
                except Exception: vals.append(np.nan)
            # Standard EIGENSOFT twstats output:
            # N eigenvalue difference twstat P-value effect.n
            eig=vals[0] if len(vals)>0 else np.nan
            diff=vals[1] if len(vals)>1 else np.nan
            tw=vals[2] if len(vals)>2 else np.nan
            pv=vals[3] if len(vals)>3 else np.nan
            eff=vals[4] if len(vals)>4 else np.nan
            rows.append({"PC":pc,"eigenvalue":eig,"difference":diff,"twstat":tw,"p_value":pv,"effective_n":eff})
    if not rows:
        raise ValueError(f"Could not parse twstats output: {path}")
    out=pd.DataFrame(rows).sort_values("PC").drop_duplicates("PC",keep="first")
    return out

def select_leading_consecutive(tw,alpha):
    pmap=dict(zip(tw.PC.astype(int),tw.p_value.astype(float)))
    selected=[]
    pc=1
    while pc in pmap and np.isfinite(pmap[pc]) and pmap[pc]<float(alpha):
        selected.append(pc);pc+=1
    nominal=tw.loc[np.isfinite(tw.p_value)&(tw.p_value<float(alpha)),"PC"].astype(int).tolist()
    later=[x for x in nominal if x not in selected]
    return selected,nominal,later

def read_eigenvec(path,n_pcs):
    names=["FID","IID"]+[f"gPC{i}" for i in range(1,int(n_pcs)+1)]
    x=pd.read_csv(path,sep=r"\s+",header=None,names=names,dtype={"FID":str,"IID":str})
    for c in names[2:]: x[c]=pd.to_numeric(x[c],errors="coerce")
    return x

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt")
    ap.add_argument("--analysis-name",required=True)
    ap.add_argument("--raw-bfile",default="/public/labdata/public_data/AMP_PD_WGS/plink/all_chrs_merged")
    ap.add_argument("--plink",default="/usr/bin/plink1.9")
    ap.add_argument("--twstats",default="/usr/bin/twstats")
    ap.add_argument("--twtable",default="twtable")
    ap.add_argument("--cohorts",default="PP,PD,SU")
    ap.add_argument("--max-pcs",type=int,default=20)
    ap.add_argument("--alpha",type=float,default=0.05)
    ap.add_argument("--mind",type=float,default=0.05)
    ap.add_argument("--geno",type=float,default=0.05)
    ap.add_argument("--maf",type=float,default=0.05)
    ap.add_argument("--hwe",type=float,default=1e-6)
    ap.add_argument("--ld-prune",action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument("--prune-window",type=int,default=50)
    ap.add_argument("--prune-step",type=int,default=5)
    ap.add_argument("--prune-r2",type=float,default=0.2)
    ap.add_argument("--force",action="store_true")
    a=ap.parse_args()
    root=Path(a.root);adir=root/"single_iv_causal"/a.analysis_name
    match_file=adir/"participant_genotype_match.csv"
    if not match_file.exists(): raise FileNotFoundError(f"{match_file}; run scripts_stage2/03_prepare_phenotype_genotype.py first")
    # Accept both executable paths and binary names available on PATH.
    def resolve_executable(name):
        from shutil import which
        candidate=which(str(name))
        if candidate is None:
            raise FileNotFoundError(f"Executable not found on PATH or at specified path: {name}")
        return str(Path(candidate).resolve())
    a.plink=resolve_executable(a.plink)
    a.twstats=resolve_executable(a.twstats)
    print(f"[TOOLS] PLINK={a.plink} twstats={a.twstats}",flush=True)
    x=read_match(match_file)
    cohorts=csv_list(a.cohorts)
    x=x[x.cohort.isin(cohorts)].copy()
    out=adir/"genotype_pcs";out.mkdir(parents=True,exist_ok=True)
    qc={"mind":a.mind,"geno":a.geno,"maf":a.maf,"hwe":a.hwe}
    prune={"enabled":bool(a.ld_prune),"window":a.prune_window,"step":a.prune_step,"r2":a.prune_r2}
    fp=fingerprint(x,a.raw_bfile,a.alpha,a.max_pcs,qc,prune)
    manifest_file=out/"pc_selection.json"
    combined_file=out/"single_iv_genotype_pcs.csv"
    if not a.force and manifest_file.exists() and combined_file.exists():
        old=json.loads(manifest_file.read_text())
        if old.get("input_fingerprint")==fp:
            print(f"Inputs unchanged; reusing {manifest_file} and {combined_file}")
            print(json.dumps(old,indent=2))
            return
    cohort_results={};combined=[]
    for cohort in cohorts:
        cc=x[x.cohort.eq(cohort)].copy()
        if cc.empty:
            print(f"[WARN] cohort {cohort}: no genotype-matched participants; skipped",flush=True)
            continue
        cdir=out/cohort;cdir.mkdir(exist_ok=True)
        keep=cdir/"keep.txt";cc[["FID","IID"]].drop_duplicates().to_csv(keep,sep="\t",header=False,index=False)
        qc_prefix=cdir/f"{cohort}.qc"
        cmd=[a.plink,"--bfile",a.raw_bfile,"--keep",keep,"--mind",a.mind,"--geno",a.geno,"--maf",a.maf,"--hwe",a.hwe,"--make-bed","--out",qc_prefix]
        run(cmd,cdir/"01_qc.log")
        fam=pd.read_csv(str(qc_prefix)+".fam",sep=r"\s+",header=None,dtype=str)
        n=int(len(fam))
        if n<4: raise RuntimeError(f"{cohort}: only {n} participants remain after QC; PCA/twstats not reliable")
        pca_input=qc_prefix
        n_pruned=np.nan
        if a.ld_prune:
            prune_prefix=cdir/f"{cohort}.ld"
            run([a.plink,"--bfile",qc_prefix,"--indep-pairwise",a.prune_window,a.prune_step,a.prune_r2,"--out",prune_prefix],cdir/"02_ld_prune.log")
            prune_in=Path(str(prune_prefix)+".prune.in")
            if not prune_in.exists() or prune_in.stat().st_size==0: raise RuntimeError(f"{cohort}: no variants in {prune_in}")
            n_pruned=sum(1 for _ in open(prune_in))
            pca_prefix=cdir/f"{cohort}.pca"
            npcs=min(int(a.max_pcs),n-1)
            run([a.plink,"--bfile",qc_prefix,"--extract",prune_in,"--pca",npcs,"--out",pca_prefix],cdir/"03_pca.log")
        else:
            pca_prefix=cdir/f"{cohort}.pca";npcs=min(int(a.max_pcs),n-1)
            run([a.plink,"--bfile",qc_prefix,"--pca",npcs,"--out",pca_prefix],cdir/"03_pca.log")
        eigenval=Path(str(pca_prefix)+".eigenval");eigenvec=Path(str(pca_prefix)+".eigenvec")
        twout=cdir/f"{cohort}.twstats.out"
        run([a.twstats,"-t",a.twtable,"-i",eigenval,"-o",twout],cdir/"04_twstats.log")
        tw=parse_twstats(twout,npcs);selected,nominal,later=select_leading_consecutive(tw,a.alpha)
        tw["significant"]=tw.p_value<a.alpha;tw["selected_leading"]=tw.PC.isin(selected);tw.to_csv(cdir/f"{cohort}.twstats.parsed.csv",index=False)
        ev=read_eigenvec(eigenvec,npcs)
        merged=cc[["cohort","participant_id","FID","IID"]].merge(ev,on=["FID","IID"],how="inner",validate="one_to_one")
        missing=sorted(set(cc.participant_id)-set(merged.participant_id))
        if missing: print(f"[WARN] {cohort}: {len(missing)} phenotype participants absent after PCA QC; first={missing[:5]}",flush=True)
        merged.to_csv(cdir/f"{cohort}.all_pcs.csv",index=False)
        combined.append(merged)
        cohort_results[cohort]={
            "n_input_participants":int(cc.participant_id.nunique()),
            "n_after_qc":n,
            "n_pca_participants":int(merged.participant_id.nunique()),
            "n_pca_components_calculated":npcs,
            "selected_n_pcs":len(selected),
            "selected_pc_indices":selected,
            "selected_columns":[f"gPC{i}" for i in selected],
            "selection_rule":"leading_consecutive_twstats_p_lt_alpha",
            "all_nominal_significant_pc_indices":nominal,
            "later_isolated_significant_pc_indices":later,
            "nonconsecutive_significance_observed":bool(later),
            "alpha":a.alpha,
            "n_ld_pruned_variants":None if not np.isfinite(n_pruned) else int(n_pruned),
            "twstats_parsed":str(cdir/f"{cohort}.twstats.parsed.csv"),
            "eigenvec":str(eigenvec),
            "warning":"Later isolated Tracy-Widom-significant PCs are diagnostic only and are not included." if later else ""
        }
        print(f"{cohort}: n={n}; selected PCs={selected}; nominal TW-significant={nominal}",flush=True)
    if not combined: raise RuntimeError("No cohort PCA results produced")
    comb=pd.concat(combined,ignore_index=True,sort=False)
    comb.to_csv(combined_file,index=False)
    manifest={
        "analysis_name":a.analysis_name,
        "input_fingerprint":fp,
        "raw_bfile":a.raw_bfile,
        "plink":a.plink,
        "twstats":a.twstats,
        "twtable":a.twtable,
        "pc_number_selection":"Stage-I production rule: PLINK PCA followed by EIGENSOFT twstats; retain leading consecutive PCs with Tracy-Widom P < alpha",
        "alpha":a.alpha,
        "max_pcs":a.max_pcs,
        "genotype_qc":qc,
        "ld_pruning":prune,
        "cohorts":cohort_results,
        "combined_pc_file":str(combined_file)
    }
    manifest_file.write_text(json.dumps(manifest,indent=2))
    print("\n=== Single-IV Stage-II genotype PC selection ===")
    for c,r in cohort_results.items(): print(f"{c}: selected_n_pcs={r['selected_n_pcs']} selected={r['selected_pc_indices']}")
    print(f"manifest: {manifest_file}")
    print(f"PC covariates: {combined_file}")

if __name__=="__main__":
    main()
