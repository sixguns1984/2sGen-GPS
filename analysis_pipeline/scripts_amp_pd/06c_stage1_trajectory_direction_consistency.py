#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

METHOD="submitted_followup_ctar_mptga"
BETA_COLS=[f"beta_g_{i}" for i in range(8)]


def _numeric_beta_cols(path: Path):
    import pyarrow.parquet as pq
    names=pq.ParquetFile(path).schema_arrow.names
    cols=[]
    for c in names:
        if str(c).startswith("beta_g_"):
            try:cols.append((int(str(c).split("beta_g_")[-1]),str(c)))
            except Exception:pass
    return [c for _,c in sorted(cols)]


def _read_minimal(path: Path, extra_cols):
    cols=["Variant_ID","Ensembl_ID"]+list(extra_cols)
    return pd.read_parquet(path,columns=cols)


def _complement_allele(a: str):
    a=str(a).upper()
    mp={"A":"T","T":"A","C":"G","G":"C"}
    return "".join(mp.get(x,x) for x in a)


def _allele_alignment(pp_bim: Path, rep_bim: Path, variants):
    pp=pd.read_csv(pp_bim,sep=r"\s+",header=None,usecols=[1,4,5],names=["snpid","a1","a2"],dtype=str)
    rp=pd.read_csv(rep_bim,sep=r"\s+",header=None,usecols=[1,4,5],names=["snpid","a1","a2"],dtype=str)
    pp=pp.drop_duplicates("snpid").set_index("snpid")
    rp=rp.drop_duplicates("snpid").set_index("snpid")
    variants=pd.Index(pd.unique(pd.Series(variants,dtype=str)),dtype=str)
    out={}
    for s in variants:
        if s not in pp.index or s not in rp.index:
            out[s]="missing_bim"
            continue
        p1,p2=str(pp.at[s,"a1"]).upper(),str(pp.at[s,"a2"]).upper()
        r1,r2=str(rp.at[s,"a1"]).upper(),str(rp.at[s,"a2"]).upper()
        if p1==r1 and p2==r2:out[s]="same"
        elif p1==r2 and p2==r1:out[s]="flip"
        elif p1==_complement_allele(r1) and p2==_complement_allele(r2):out[s]="same_complement"
        elif p1==_complement_allele(r2) and p2==_complement_allele(r1):out[s]="flip_complement"
        else:out[s]="unresolved"
    return out


def _cohort_followup_support(code_root: Path, root: Path, cohort: str):
    if str(code_root) not in sys.path:sys.path.insert(0,str(code_root))
    from gengps.amp_pd.config import AmpPdPaths
    from gengps.amp_pd.real_qtl import load_cohort_metadata,_augment_fixed_genotype_pcs,_prepare_irregular_followup_rows,method_covariates,_drop_missing_covariates
    paths=AmpPdPaths(root=str(root))
    meta=load_cohort_metadata(paths,cohort)
    meta=_augment_fixed_genotype_pcs(paths,cohort,meta,n_pcs=8)
    empty=pd.DataFrame(index=meta.index)
    meta,empty,_=_prepare_irregular_followup_rows(meta,empty)
    covars=method_covariates(METHOD,cohort,meta,cell_adjust=True)
    meta,empty=_drop_missing_covariates(meta,empty,covars)
    t=pd.to_numeric(meta["followup_time"],errors="coerce").to_numpy(float)
    t=t[np.isfinite(t)]
    if len(t)==0:raise RuntimeError(f"No finite follow-up times remain for cohort={cohort}")
    return {"cohort":cohort,"time_min":float(np.min(t)),"time_max":float(np.max(t)),"n_obs":int(len(meta)),"n_participants":int(meta["participant_id"].astype(str).nunique()),"n_covariates":int(len(covars)),"covariates":list(covars)}


def _raw_cubic_basis(t):
    t=np.asarray(t,float)
    return np.column_stack([np.ones_like(t),t,t*t,t*t*t])


def _trajectory_r_batch(beta_dis,beta_rep,flip_mask,basis):
    beta_dis=np.asarray(beta_dis,float)
    beta_rep=np.asarray(beta_rep,float)
    flip_mask=np.asarray(flip_mask,bool)
    B=np.asarray(basis,float)
    if beta_dis.ndim!=2 or beta_dis.shape[1]!=8:raise ValueError(f"Expected discovery beta shape (n,8), got {beta_dis.shape}")
    if beta_rep.ndim!=2 or beta_rep.shape[1]!=8:raise ValueError(f"Expected replication beta shape (n,8), got {beta_rep.shape}")
    d2=beta_dis[:,0:4]@B.T
    d1=beta_dis[:,4:8]@B.T
    r2=beta_rep[:,0:4]@B.T
    r1=beta_rep[:,4:8]@B.T
    if np.any(flip_mask):
        old_r2=r2[flip_mask].copy()
        old_r1=r1[flip_mask].copy()
        r1[flip_mask]=old_r1-old_r2
        r2[flip_mask]=-old_r2
    nvec=3*B.shape[0]
    sx=np.sum(d1,axis=1)+np.sum(d2,axis=1)
    sy=np.sum(r1,axis=1)+np.sum(r2,axis=1)
    sxx=np.sum(d1*d1,axis=1)+np.sum(d2*d2,axis=1)
    syy=np.sum(r1*r1,axis=1)+np.sum(r2*r2,axis=1)
    sxy=np.sum(d1*r1,axis=1)+np.sum(d2*r2,axis=1)
    vx=sxx-(sx*sx)/nvec
    vy=syy-(sy*sy)/nvec
    cov=sxy-(sx*sy)/nvec
    den=np.sqrt(np.maximum(vx,0)*np.maximum(vy,0))
    r=np.full(len(beta_dis),np.nan,float)
    ok=np.isfinite(cov)&np.isfinite(den)&(den>1e-14)
    r[ok]=np.clip(cov[ok]/den[ok],-1.0,1.0)
    return r


def _summary_row(df: pd.DataFrame,name: str,mask):
    z=df.loc[np.asarray(mask,bool),"trajectory_r"]
    z=pd.to_numeric(z,errors="coerce")
    z=z[np.isfinite(z)]
    return {"subset":name,"n_pairs":int(np.asarray(mask,bool).sum()),"n_with_finite_trajectory_r":int(len(z)),"n_positive_trajectory_r":int((z>0).sum()),"fraction_positive_trajectory_r":float((z>0).mean()) if len(z) else np.nan,"median_trajectory_r":float(z.median()) if len(z) else np.nan,"q25_trajectory_r":float(z.quantile(.25)) if len(z) else np.nan,"q75_trajectory_r":float(z.quantile(.75)) if len(z) else np.nan}


def _self_test():
    t=np.linspace(0,2,51)
    B=_raw_cubic_basis(t)
    d=np.array([[0.3,0.2,-0.05,0.01,0.15,0.1,-0.02,0.005]],float)
    r_same=_trajectory_r_batch(d,d,np.array([False]),B)[0]
    r_opp=_trajectory_r_batch(d,-d,np.array([False]),B)[0]
    b=d.copy()
    d2=b[:,0:4].copy();d1=b[:,4:8].copy()
    rep=np.empty_like(b)
    rep[:,0:4]=-d2
    rep[:,4:8]=d1-d2
    r_flip=_trajectory_r_batch(d,rep,np.array([True]),B)[0]
    assert r_same>0.999999
    assert r_opp<-0.999999
    assert r_flip>0.999999
    print("self-test passed")


def main():
    ap=argparse.ArgumentParser(description="PPMI-PDBP trajectory-level direction consistency for revised CTAR Stage-I tQTLs")
    ap.add_argument("--root",default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt")
    ap.add_argument("--code-root",default=None)
    ap.add_argument("--n-grid",type=int,default=101)
    ap.add_argument("--batch-size",type=int,default=25000)
    ap.add_argument("--out-dir",default=None)
    ap.add_argument("--self-test",action="store_true")
    a=ap.parse_args()
    if a.self_test:
        _self_test()
        return
    root=Path(a.root)
    code_root=Path(a.code_root) if a.code_root else Path(__file__).resolve().parents[1]
    out=Path(a.out_dir) if a.out_dir else root/"qtl"/"trajectory_direction_consistency"/METHOD
    out.mkdir(parents=True,exist_ok=True)
    pp=root/"qtl"/"PP"/METHOD/"final"/"significant_tqtl.parquet"
    if not pp.exists():pp=root/"qtl"/"PP"/METHOD/"final"/"significant_temporal.parquet"
    pdp=root/"qtl"/"PD"/METHOD/"final"/"all_results.parquet"
    if not pp.exists():raise FileNotFoundError(pp)
    if not pdp.exists():raise FileNotFoundError(pdp)
    pp_beta=_numeric_beta_cols(pp)
    pd_beta=_numeric_beta_cols(pdp)
    if pp_beta!=BETA_COLS:raise RuntimeError(f"PPMI expected beta columns {BETA_COLS}, found {pp_beta}")
    if pd_beta!=BETA_COLS:raise RuntimeError(f"PDBP expected beta columns {BETA_COLS}, found {pd_beta}")
    d=_read_minimal(pp,["Pvalue","Interaction_Pvalue"]+BETA_COLS)
    r=_read_minimal(pdp,["Pvalue","Interaction_Pvalue"]+BETA_COLS)
    if d.duplicated(["Variant_ID","Ensembl_ID"]).any():raise RuntimeError("Duplicate PPMI Variant_ID-Ensembl_ID pairs")
    if r.duplicated(["Variant_ID","Ensembl_ID"]).any():raise RuntimeError("Duplicate PDBP Variant_ID-Ensembl_ID pairs")
    m=d.merge(r,on=["Variant_ID","Ensembl_ID"],how="inner",suffixes=("_dis","_rep"),validate="one_to_one")
    print(f"PPMI discovery tQTLs={len(d):,}; PDBP replication rows={len(r):,}; matched={len(m):,}",flush=True)
    pp_support=_cohort_followup_support(code_root,root,"PP")
    pd_support=_cohort_followup_support(code_root,root,"PD")
    lo=max(pp_support["time_min"],pd_support["time_min"])
    hi=min(pp_support["time_max"],pd_support["time_max"])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi<=lo:raise RuntimeError(f"No positive common follow-up support: PPMI={pp_support}, PDBP={pd_support}")
    grid=np.linspace(lo,hi,int(a.n_grid))
    basis=_raw_cubic_basis(grid)
    pp_bim=root/"data"/"genotype"/"PP"/"amp_pd_PP_qc.bim"
    rep_bim=root/"data"/"genotype"/"REP"/"amp_pd_REP_qc.bim"
    amap=_allele_alignment(pp_bim,rep_bim,m["Variant_ID"].astype(str))
    m["allele_alignment"]=m["Variant_ID"].astype(str).map(amap)
    valid=m["allele_alignment"].isin(["same","same_complement","flip","flip_complement"])
    m["trajectory_r"]=np.nan
    bs=max(1,int(a.batch_size))
    idx=np.flatnonzero(valid.to_numpy())
    for st in range(0,len(idx),bs):
        ii=idx[st:st+bs]
        bd=m.iloc[ii][[f"{c}_dis" for c in BETA_COLS]].to_numpy(float)
        br=m.iloc[ii][[f"{c}_rep" for c in BETA_COLS]].to_numpy(float)
        flip=m.iloc[ii]["allele_alignment"].isin(["flip","flip_complement"]).to_numpy()
        rr=_trajectory_r_batch(bd,br,flip,basis)
        m.iloc[ii,m.columns.get_loc("trajectory_r")]=rr
        print(f"trajectory concordance {min(st+bs,len(idx)):,}/{len(idx):,}",flush=True)
    m["trajectory_direction_concordant"]=pd.array(np.where(np.isfinite(m["trajectory_r"]),m["trajectory_r"]>0,None),dtype="boolean")
    m["pdbp_temporal_rep_p_lt_0_05"]=pd.to_numeric(m["Pvalue_rep"],errors="coerce")<0.05
    summaries=[]
    finite_align=m["allele_alignment"].isin(["same","same_complement","flip","flip_complement"])
    summaries.append(_summary_row(m,"all_matched_discovery_tqtl",finite_align))
    summaries.append(_summary_row(m,"pdbp_nominal_temporal_replication",finite_align&m["pdbp_temporal_rep_p_lt_0_05"]))
    summary=pd.DataFrame(summaries)
    keep=["Variant_ID","Ensembl_ID","Pvalue_dis","Interaction_Pvalue_dis","Pvalue_rep","Interaction_Pvalue_rep","allele_alignment","trajectory_r","trajectory_direction_concordant","pdbp_temporal_rep_p_lt_0_05"]
    m[keep].to_parquet(out/"trajectory_direction_pairs.parquet",index=False,compression="zstd")
    summary.to_csv(out/"trajectory_direction_summary.csv",index=False)
    align_summary=m["allele_alignment"].value_counts(dropna=False).rename_axis("allele_alignment").reset_index(name="n_pairs")
    align_summary.to_csv(out/"allele_alignment_summary.csv",index=False)
    metadata={"method":METHOD,"definition":"Pearson correlation between allele-harmonized PPMI and PDBP genotype-effect trajectories evaluated on the common follow-up interval. The trajectory vector concatenates the reference-genotype zero curve and the G=1 and G=2 effect curves, preserving the exact two-channel binary hard-call CTAR parameterization.","direction_concordant":"trajectory_r > 0","n_grid":int(a.n_grid),"common_followup_min":float(lo),"common_followup_max":float(hi),"ppmi_support":pp_support,"pdbp_support":pd_support,"ppmi_file":str(pp),"pdbp_file":str(pdp),"ppmi_bim":str(pp_bim),"pdbp_bim":str(rep_bim),"n_discovery":int(len(d)),"n_matched":int(len(m)),"n_allele_harmonized":int(valid.sum())}
    (out/"trajectory_direction_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print("\n=== trajectory-level direction consistency ===")
    print(summary.to_string(index=False))
    print("\n=== allele alignment ===")
    print(align_summary.to_string(index=False))
    print(f"\nOutputs: {out}")


if __name__=="__main__":
    main()
