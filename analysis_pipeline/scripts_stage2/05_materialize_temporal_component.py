#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gengps.amp_pd.genotype import make_genotype_source
from gengps.amp_pd.grex import legacy_single_iv_from_qtl_row
_WORKER={}

def clean_cov(path):
    x=pd.read_csv(path,low_memory=False);x=x.drop(columns=[c for c in x.columns if str(c).startswith("Unnamed:")],errors="ignore")
    for c in ["cohort","participant_id","time"]:
        if c not in x.columns:raise ValueError(f"Missing required phenotype column {c}")
    x["cohort"]=x.cohort.astype(str);x["participant_id"]=x.participant_id.astype(str);x["time"]=pd.to_numeric(x.time,errors="coerce");x=x.dropna(subset=["participant_id","time"]).sort_values(["cohort","participant_id","time"]).drop_duplicates(["cohort","participant_id","time"],keep="first").reset_index(drop=True);x["single_iv_row_id"]=np.arange(len(x));return x

def followup_support_mask(rows,tmin,tmax,tol=1e-10):
    t=pd.to_numeric(rows["time"],errors="coerce").to_numpy(float);return np.isfinite(t)&(t>=float(tmin)-float(tol))&(t<=float(tmax)+float(tol))

def read_genotype_frame(gsrc,rows,snp):
    pids=pd.Index(pd.unique(rows.participant_id.astype(str)),dtype=str);G=np.asarray(gsrc.read(pids.tolist(),[str(snp)],mean_impute=True,dtype=np.float64),float)
    if G.shape!=(len(pids),1):raise ValueError(f"genotype shape {G.shape}, expected {(len(pids),1)}")
    return pd.Series(G[:,0],index=pids,name="iv_dosage")

def meta_for_model(rows):
    m=rows.copy();m["participant_id"]=m.participant_id.astype(str);m["followup_time"]=pd.to_numeric(m.time,errors="coerce")
    if "age" in m.columns:m["current_age"]=pd.to_numeric(m.age,errors="coerce")+m.followup_time
    else:m["current_age"]=m.followup_time
    if "sample_id" not in m.columns:m["sample_id"]=m.participant_id+"__singleIV__"+m.followup_time.astype(str)
    return m

def direct_component(model,gsrc,rows,snp):
    m=meta_for_model(rows);dose=read_genotype_frame(gsrc,m,snp);donorG=pd.DataFrame({str(snp):dose.values},index=dose.index.astype(str));pred=np.asarray(model.predict_grex(donorG,m),float)
    if pred.shape!=(len(m),):
        G=pd.DataFrame({str(snp):m.participant_id.map(dose).to_numpy(float)},index=m.index);pred=np.asarray(model.predict_grex(G,m),float)
    if pred.shape!=(len(m),):raise ValueError(f"component shape {pred.shape}, expected {(len(m),)}")
    return pred,m.participant_id.map(dose).to_numpy(float)

def _support_columns(z,tmin,tmax):
    support=followup_support_mask(z,tmin,tmax);z["stage1_followup_min"]=float(tmin);z["stage1_followup_max"]=float(tmax);z["within_stage1_followup_support"]=support;z["within_stage1_time_support"]=support;return z,support

def _summary(aid,gene,snp,z,support,status="ok"):
    t=pd.to_numeric(z.time,errors="coerce") if len(z) else pd.Series(dtype=float);return {"artifact_id":aid,"gene":gene,"lead_snp":snp,"status":status,"n_rows":int(len(z)),"n_participants":int(z.participant_id.nunique()) if "participant_id" in z.columns else 0,"phenotype_followup_min":float(t.min()) if len(t) else np.nan,"phenotype_followup_max":float(t.max()) if len(t) else np.nan,"stage1_followup_min":float(_WORKER["stage1_followup_min"]),"stage1_followup_max":float(_WORKER["stage1_followup_max"]),"n_rows_within_stage1_followup_support":int(np.sum(support)),"n_rows_outside_stage1_followup_support":int(np.sum(~support)),"support_fraction":float(np.mean(support)) if len(support) else np.nan}

def _init(cfg):
    global _WORKER
    _WORKER=dict(cfg);_WORKER["cov"]=pd.read_parquet(cfg["cov_cache"])
    if not cfg.get("rewrite_support_only",False):_WORKER["gsrc"]=make_genotype_source(cfg["bfile"],cfg["plink"],prefer_bed_reader=True)

def _one(row):
    try:
        gene=str(row["gene"]);snp=str(row["lead_snp"]);aid=str(row["artifact_id"]);pf=Path(_WORKER["out"])/"single_iv_genetic_component"/f"{aid}.parquet"
        if _WORKER.get("rewrite_support_only",False):
            if not pf.exists():raise FileNotFoundError(f"Existing component required for --rewrite-support-only: {pf}")
            z=pd.read_parquet(pf);z,support=_support_columns(z,_WORKER["stage1_followup_min"],_WORKER["stage1_followup_max"]);z["support_definition"]="explicit_PPMI_StageI_followup_domain";z.to_parquet(pf,index=False,compression="zstd");return _summary(aid,gene,snp,z,support,"support_rewritten"),None
        payload=dict(row);payload["Variant_ID"]=snp;payload["Ensembl_ID"]=gene;payload["Pvalue"]=float(pd.to_numeric(pd.Series([row.get("ppmi_tqtl_p",np.nan)]),errors="coerce").iloc[0]);model=legacy_single_iv_from_qtl_row(payload,qtl_method=_WORKER["qtl_label"]);cov=_WORKER["cov"].copy();pred,dose=direct_component(model,_WORKER["gsrc"],cov,snp);z=cov.copy();z["gene"]=gene;z["lead_snp"]=snp;z["artifact_id"]=aid;z["grex_single_iv"]=pred;z["iv_dosage"]=dose;z["component_source"]="full_frozen_PPMI_StageI_temporal_single_IV_row";z,support=_support_columns(z,_WORKER["stage1_followup_min"],_WORKER["stage1_followup_max"]);z["support_definition"]="explicit_PPMI_StageI_followup_domain";keep=[c for c in ["single_iv_row_id","cohort","participant_id","time","gene","lead_snp","artifact_id","grex_single_iv","iv_dosage","stage1_followup_min","stage1_followup_max","within_stage1_followup_support","within_stage1_time_support","support_definition","component_source"] if c in z.columns];z[keep].to_parquet(pf,index=False,compression="zstd");return _summary(aid,gene,snp,z,support),None
    except Exception as e:return {"artifact_id":str(row.get("artifact_id","")),"gene":str(row.get("gene","")),"lead_snp":str(row.get("lead_snp","")),"status":"error","n_rows":0},{"artifact_id":str(row.get("artifact_id","")),"gene":str(row.get("gene","")),"lead_snp":str(row.get("lead_snp","")),"error":repr(e),"traceback":traceback.format_exc()}

def main():
    ap=argparse.ArgumentParser(description="Materialize or update the frozen PPMI Stage-I temporal single-IV genetic component and explicitly encode its follow-up-time support domain.");ap.add_argument("--root",default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt");ap.add_argument("--analysis-name",default="updrs3");ap.add_argument("--phenotype-cov",required=True);ap.add_argument("--candidate-file",default="");ap.add_argument("--qtl-label",default="submitted_followup_ctar_mptga");ap.add_argument("--single-iv-bfile",default="");ap.add_argument("--plink",default="plink");ap.add_argument("--stage1-followup-min",type=float,default=0.0);ap.add_argument("--stage1-followup-max",type=float,required=True);ap.add_argument("--rewrite-support-only",action="store_true");ap.add_argument("--n-jobs",type=int,default=8);a=ap.parse_args()
    if not np.isfinite(a.stage1_followup_min) or not np.isfinite(a.stage1_followup_max) or a.stage1_followup_max<a.stage1_followup_min:raise ValueError("Require finite --stage1-followup-min/max with max >= min")
    root=Path(a.root);out=root/"single_iv_causal"/a.analysis_name;(out/"single_iv_genetic_component").mkdir(parents=True,exist_ok=True);cf=Path(a.candidate_file) if a.candidate_file else root/"single_iv_causal"/"iv_validation"/"single_iv_primary_eligible.csv";cand=pd.read_csv(cf,low_memory=False);bfile=a.single_iv_bfile or str(out/"genotype"/"single_iv_snps");need={"gene","lead_snp","artifact_id"};missing=need-set(cand.columns)
    if not a.rewrite_support_only:need|={"basis_json"};missing=need-set(cand.columns)
    if missing:raise ValueError(f"candidate file missing {sorted(missing)}")
    if not a.rewrite_support_only:
        for ext in [".bed",".bim",".fam"]:
            if not Path(bfile+ext).exists():raise FileNotFoundError(bfile+ext)
    cov=clean_cov(a.phenotype_cov);cache=out/"phenotype_clean.parquet";cov.to_parquet(cache,index=False,compression="zstd");rows=cand.to_dict("records");cfg={"cov_cache":str(cache),"bfile":bfile,"plink":a.plink,"out":str(out),"qtl_label":a.qtl_label,"stage1_followup_min":float(a.stage1_followup_min),"stage1_followup_max":float(a.stage1_followup_max),"rewrite_support_only":bool(a.rewrite_support_only)};summ=[];errs=[]
    if a.n_jobs<=1:
        _init(cfg)
        for r in rows:s,e=_one(r);summ.append(s);errs.extend([e] if e else [])
    else:
        with ProcessPoolExecutor(max_workers=a.n_jobs,initializer=_init,initargs=(cfg,)) as ex:
            futs=[ex.submit(_one,r) for r in rows]
            for i,f in enumerate(as_completed(futs),1):
                s,e=f.result();summ.append(s);errs.extend([e] if e else [])
                if i%50==0 or i==len(futs):print(f"components/support {i}/{len(futs)}",flush=True)
    sdf=pd.DataFrame(summ).sort_values(["gene","lead_snp"]);sdf.to_csv(out/"single_iv_genetic_component_summary.csv",index=False);pd.DataFrame(errs).to_csv(out/"single_iv_genetic_component_errors.csv",index=False)
    nout=int(pd.to_numeric(sdf.get("n_rows_outside_stage1_followup_support",0),errors="coerce").fillna(0).sum()) if len(sdf) else 0;manifest={"analysis_name":a.analysis_name,"candidate_file":str(cf),"conceptual_role":"one full frozen PPMI Stage-I temporal genetic-expression component per LD-independent replicated temporal tQTL IV-gene pair; no dynamic/additive branches and no multi-SNP predictor","temporal_component":"complete frozen PPMI Stage-I MPTGA genetic function evaluated at each observed outcome follow-up time","stage1_time_axis":"followup_time","stage1_model_cohort":"PP","stage1_followup_min":float(a.stage1_followup_min),"stage1_followup_max":float(a.stage1_followup_max),"support_definition":"stage1_followup_min <= phenotype time <= stage1_followup_max; explicit PPMI Stage-I follow-up domain, independent of age/current_age","rewrite_support_only":bool(a.rewrite_support_only),"n_candidate_pairs":int(len(cand)),"n_candidate_genes":int(cand.gene.nunique()),"n_component_errors":int(len(errs)),"total_component_rows_outside_stage1_followup_support":nout,"bfile":bfile if not a.rewrite_support_only else "not_used_support_only"};(out/"single_iv_genetic_component_manifest.json").write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
    if errs:print(f"[WARN] {len(errs)} component/support jobs failed; inspect single_iv_genetic_component_errors.csv",flush=True)
    if not errs and cov.time.max()>a.stage1_followup_max and nout==0:raise RuntimeError("Phenotype contains visits beyond --stage1-followup-max but no component rows were marked outside support; support encoding failed")
if __name__=="__main__":main()
