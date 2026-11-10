#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gengps.amp_pd.single_iv_causal import bh_adjust,fit_all_lags

_WORKER_COV=None
_WORKER_COHORT_CACHE={}
_WORKER_SELECTED_PCS={}

def _init_worker(cov_cache,selected_pcs):
    global _WORKER_COV,_WORKER_COHORT_CACHE,_WORKER_SELECTED_PCS
    _WORKER_COV=pd.read_parquet(cov_cache);_WORKER_COHORT_CACHE={};_WORKER_SELECTED_PCS=selected_pcs or {}

def _worker_cov(cohort):
    global _WORKER_COV,_WORKER_COHORT_CACHE
    key=str(cohort)
    if key not in _WORKER_COHORT_CACHE:
        members=cohort_members(key);_WORKER_COHORT_CACHE[key]=_WORKER_COV[_WORKER_COV.cohort.astype(str).isin(members)].copy()
    return _WORKER_COHORT_CACHE[key]

def csv_list(x):return [z.strip() for z in str(x).split(",") if z.strip()]
def semi_list(x):return [z.strip() for z in str(x).split(";") if z.strip()]
def cohort_members(x):return [z.strip() for z in str(x).split("+") if z.strip()]
def label(x):return str(x).replace("+","_PLUS_").replace("/","_").replace(" ","")
def read_cov(path,outcome_col):
    x=pd.read_csv(path,low_memory=False);x=x.drop(columns=[c for c in x.columns if str(c).startswith("Unnamed:")],errors="ignore")
    for c in ["cohort","participant_id","time",outcome_col]:
        if c not in x.columns:raise ValueError(f"Phenotype file missing {c}")
    x["cohort"]=x.cohort.astype(str);x["participant_id"]=x.participant_id.astype(str);x["time"]=pd.to_numeric(x.time,errors="coerce");x[outcome_col]=pd.to_numeric(x[outcome_col],errors="coerce");x=x.dropna(subset=["participant_id","time",outcome_col]).sort_values(["cohort","participant_id","time"],kind="mergesort").drop_duplicates(["cohort","participant_id","time"],keep="first").reset_index(drop=True);x["single_iv_row_id"]=np.arange(len(x));return x
def add_pcs(cov,pc_file,pc_manifest):
    if not Path(pc_file).exists():raise FileNotFoundError(pc_file)
    pc=pd.read_csv(pc_file,low_memory=False);pc["cohort"]=pc.cohort.astype(str);pc["participant_id"]=pc.participant_id.astype(str);manifest=json.loads(Path(pc_manifest).read_text()) if Path(pc_manifest).exists() else {};selected={}
    for c,r in manifest.get("cohorts",{}).items():selected[str(c)]=[str(z) for z in r.get("selected_columns",[])]
    allcols=sorted({z for v in selected.values() for z in v},key=lambda s:int(s.replace("gPC","")) if s.startswith("gPC") and s.replace("gPC","").isdigit() else s);z=pc[[c for c in ["cohort","participant_id"]+allcols if c in pc.columns]].drop_duplicates(["cohort","participant_id"]);return cov.merge(z,on=["cohort","participant_id"],how="left",validate="many_to_one"),selected
def load_iv(pred_file,cov,cohort,require_support=False,cov_is_subset=False):
    wanted=["cohort","participant_id","time","gene","lead_snp","artifact_id","grex_single_iv","iv_dosage","stage1_followup_min","stage1_followup_max","within_stage1_followup_support","within_stage1_time_support"]
    try:p=pd.read_parquet(pred_file,columns=wanted)
    except Exception:
        p=pd.read_parquet(pred_file);p=p[[c for c in wanted if c in p.columns]].copy()
    members=cohort_members(cohort);c=cov.copy() if cov_is_subset else cov[cov.cohort.astype(str).isin(members)].copy();p=p[p.cohort.astype(str).isin(members)].copy();keys=["cohort","participant_id","time"]
    missing_c=[x for x in keys if x not in c.columns];missing_p=[x for x in keys if x not in p.columns]
    if missing_c or missing_p:raise ValueError(f"Stable visit-key merge unavailable: phenotype_missing={missing_c}, component_missing={missing_p}")
    for z in (c,p):z["cohort"]=z.cohort.astype(str);z["participant_id"]=z.participant_id.astype(str);z["_single_iv_time_key"]=pd.to_numeric(z.time,errors="coerce").round(10)
    mkeys=["cohort","participant_id","_single_iv_time_key"]
    if c.duplicated(mkeys).any():raise ValueError("Phenotype contains duplicate cohort-participant-time visit keys")
    if p.duplicated(mkeys).any():raise ValueError(f"Single-IV component contains duplicate cohort-participant-time visit keys: {pred_file}")
    value_cols=[x for x in ["gene","lead_snp","artifact_id","grex_single_iv","iv_dosage","stage1_followup_min","stage1_followup_max","within_stage1_followup_support","within_stage1_time_support"] if x in p.columns];d=c.merge(p[mkeys+value_cols],on=mkeys,how="inner",validate="one_to_one").drop(columns=["_single_iv_time_key"],errors="ignore")
    if require_support and "within_stage1_followup_support" not in d.columns:raise RuntimeError(f"--require-stage1-followup-support was requested but within_stage1_followup_support is absent from {pred_file}; rerun Step 12 v4.4 with --stage1-followup-min/--stage1-followup-max (or --rewrite-support-only)")
    before=d.copy();support=pd.Series(True,index=d.index) if "within_stage1_followup_support" not in d.columns else d.within_stage1_followup_support.fillna(False).astype(bool)
    smin=float(pd.to_numeric(before.stage1_followup_min,errors="coerce").dropna().iloc[0]) if "stage1_followup_min" in before.columns and pd.to_numeric(before.stage1_followup_min,errors="coerce").notna().any() else np.nan;smax=float(pd.to_numeric(before.stage1_followup_max,errors="coerce").dropna().iloc[0]) if "stage1_followup_max" in before.columns and pd.to_numeric(before.stage1_followup_max,errors="coerce").notna().any() else np.nan
    audit={"followup_support_filter_requested":bool(require_support),"stage1_followup_min":smin,"stage1_followup_max":smax,"n_rows_before_support":int(len(before)),"n_participants_before_support":int(before.participant_id.nunique()) if len(before) else 0,"time_min_before_support":float(pd.to_numeric(before.time,errors="coerce").min()) if len(before) else np.nan,"time_max_before_support":float(pd.to_numeric(before.time,errors="coerce").max()) if len(before) else np.nan,"n_rows_within_stage1_followup_support":int(support.sum()),"n_rows_outside_stage1_followup_support":int((~support).sum()),"support_fraction_before_filter":float(support.mean()) if len(support) else np.nan}
    if require_support:d=d.loc[support].copy()
    audit.update({"n_rows_after_support":int(len(d)),"n_participants_after_support":int(d.participant_id.nunique()) if len(d) else 0,"time_min_after_support":float(pd.to_numeric(d.time,errors="coerce").min()) if len(d) else np.nan,"time_max_after_support":float(pd.to_numeric(d.time,errors="coerce").max()) if len(d) else np.nan})
    return d,audit
def cohort_baseline_covariates(d,cohort,selected_pcs,use_pcs):
    out=[];members=cohort_members(cohort)
    if use_pcs:
        if len(members)==1:out += [c for c in selected_pcs.get(members[0],[]) if c in d.columns]
        else:
            for m in members:
                for pc in selected_pcs.get(m,[]):
                    if pc not in d.columns:continue
                    cc=f"{m}_{pc}";d[cc]=np.where(d.cohort.astype(str).eq(m),pd.to_numeric(d[pc],errors="coerce"),0.0);out.append(cc)
    if len(members)>1:
        ref=members[0]
        for m in members[1:]:cc=f"cohort_{m}_vs_{ref}";d[cc]=d.cohort.astype(str).eq(m).astype(float);out.append(cc)
    return d,out
def one_task(task):
    gene,snp,aid,pf,cov_cache,cohort,outcome,max_lag,tv_cov,min_part,age_cubic,require_support,use_pcs,selected_pcs,ar_mode,detail_mode,extra_cov=task
    try:
        global _WORKER_COV,_WORKER_SELECTED_PCS
        if _WORKER_COV is None:_init_worker(cov_cache,selected_pcs)
        cov=_worker_cov(cohort);pcs=_WORKER_SELECTED_PCS if _WORKER_SELECTED_PCS else selected_pcs
        d,audit=load_iv(pf,cov,cohort,require_support,cov_is_subset=True);d,base_cov=cohort_baseline_covariates(d,cohort,pcs,use_pcs);base_cov += extra_cov;r=fit_all_lags(d,outcome,"grex_single_iv",max_lag,baseline_covariates=base_cov,time_varying_covariates=tv_cov,auto_clinical_covariates=True,min_participants=min_part,adjust_current_age_cubic=age_cubic,ar_time_powers=(0,1,2,3),compute_ar=(str(ar_mode)=="all"),compute_cluster_details=(str(detail_mode)=="full"));r.insert(0,"gene",gene);r.insert(1,"lead_snp",snp);r.insert(2,"artifact_id",aid);r.insert(3,"cohort",cohort)
        for k,v in audit.items():r[k]=v
        return r,None
    except Exception as e:return pd.DataFrame(),{"gene":gene,"lead_snp":snp,"artifact_id":aid,"cohort":cohort,"error":repr(e),"traceback":traceback.format_exc()}
def run_tasks(tasks,n_jobs,label_,task_chunksize=8):
    rows=[];errs=[]
    if not tasks:return pd.DataFrame(),errs
    cov_cache=tasks[0][4];selected_pcs=tasks[0][13]
    if n_jobs<=1:
        _init_worker(cov_cache,selected_pcs)
        iterator=map(one_task,tasks)
    else:
        ex=ProcessPoolExecutor(max_workers=n_jobs,initializer=_init_worker,initargs=(cov_cache,selected_pcs));iterator=ex.map(one_task,tasks,chunksize=max(1,int(task_chunksize)))
    try:
        for i,(r,e) in enumerate(iterator,1):
            rows.append(r) if not r.empty else None;errs.extend([e] if e else [])
            if i%100==0 or i==len(tasks):print(f"{label_} {i}/{len(tasks)}",flush=True)
    finally:
        if n_jobs>1:ex.shutdown(wait=True)
    return (pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()),errs
def add_fdr(x):
    z=x.copy();z["q_bh_global"]=bh_adjust(pd.to_numeric(z.pvalue,errors="coerce"));z["ar_q_bh_global"]=bh_adjust(pd.to_numeric(z.ar_pvalue,errors="coerce"));return z
def select_discovery(x,alpha):
    z=add_fdr(x);z["discovery_fdr"]=z.ok.fillna(False).astype(bool)&(pd.to_numeric(z.q_bh_global,errors="coerce")<alpha);sig=z[z.discovery_fdr].copy();sig["weak_iv_AR_nominal_support"]=pd.to_numeric(sig.ar_pvalue,errors="coerce")<0.05;sig["weak_iv_AR_FDR_support"]=pd.to_numeric(sig.ar_q_bh_global,errors="coerce")<alpha;key=["gene","lead_snp"];sel=sig.sort_values(key+["bic","pvalue","lag_order"],na_position="last").drop_duplicates(key,keep="first").copy() if not sig.empty else pd.DataFrame(columns=z.columns)
    if len(sel):sel["selected_by"]="minimum_BIC_among_global_BH_significant_lags_within_each_independent_temporal_tQTL_IV_gene_pair"
    return z,sig,sel
def discovery_files(out):return out/"discovery_all_IV_gene_lag_models.csv.gz",out/"discovery_significant_IV_gene_lag_models.csv",out/"discovery_selected_IV_gene_models.csv"
def valid_discovery_checkpoint(allf,sigf,self,cand,expected):
    meta_file=allf.parent/"discovery_checkpoint.json"
    if not (allf.exists() and sigf.exists() and self.exists() and meta_file.exists()):return False,None
    try:meta=json.loads(meta_file.read_text())
    except Exception as e:print(f"[WARN] Existing discovery metadata is unreadable ({e}); discovery will be rerun.",flush=True);return False,None
    for k,v in expected.items():
        if (meta.get(k, []) if k=="extra_covariates" else meta.get(k))!=v:print(f"[WARN] Discovery checkpoint setting mismatch for {k}: existing={meta.get(k)!r}, requested={v!r}; discovery will be rerun.",flush=True);return False,None
    try:
        a=pd.read_csv(allf,low_memory=False);s=pd.read_csv(sigf,low_memory=False);d=pd.read_csv(self,low_memory=False)
    except Exception as e:
        print(f"[WARN] Existing discovery checkpoint is unreadable ({e}); discovery will be rerun.",flush=True);return False,None
    need={"gene","lead_snp","artifact_id","lag_order","pvalue","q_bh_global","discovery_fdr","bic","coef_sum"}
    if not need.issubset(a.columns):print(f"[WARN] Existing discovery checkpoint lacks columns {sorted(need-set(a.columns))}; discovery will be rerun.",flush=True);return False,None
    if len(a) and pd.to_numeric(a.lag_order,errors="coerce").max()>int(expected["max_lag"]):print("[WARN] Existing discovery checkpoint was generated with a larger --max-lag; discovery will be rerun.",flush=True);return False,None
    allowed=set(zip(cand.gene.astype(str),cand.lead_snp.astype(str),cand.artifact_id.astype(str)));got=set(zip(a.gene.astype(str),a.lead_snp.astype(str),a.artifact_id.astype(str)))
    if not got.issubset(allowed):print("[WARN] Existing discovery checkpoint contains IVs absent from the current candidate file; discovery will be rerun.",flush=True);return False,None
    return True,(a,s,d)
def valid_replication_checkpoint(rawf,meta_file,expected):
    if not (rawf.exists() and meta_file.exists()):return False,None
    try:
        meta=json.loads(meta_file.read_text())
        for k,v in expected.items():
            if (meta.get(k, []) if k=="extra_covariates" else meta.get(k))!=v:print(f"[WARN] Replication checkpoint setting mismatch for {k}: existing={meta.get(k)!r}, requested={v!r}; replication will be rerun.",flush=True);return False,None
        x=pd.read_csv(rawf,low_memory=False)
        return True,x
    except Exception as e:print(f"[WARN] Existing replication checkpoint is unreadable/incompatible ({e}); replication will be rerun.",flush=True);return False,None

def make_replication_targets(disc_sig):
    if disc_sig.empty:return pd.DataFrame(columns=["gene","lead_snp","artifact_id","n_discovery_significant_lags","discovery_significant_lags"])
    x=disc_sig[["gene","lead_snp","artifact_id","lag_order"]].drop_duplicates().copy();x["lag_order"]=pd.to_numeric(x.lag_order,errors="coerce").astype("Int64")
    rows=[]
    for (g,s,a),z in x.groupby(["gene","lead_snp","artifact_id"],sort=False):
        lags=sorted(int(v) for v in z.lag_order.dropna().unique());rows.append({"gene":g,"lead_snp":s,"artifact_id":a,"n_discovery_significant_lags":len(lags),"discovery_significant_lags":";".join(map(str,lags))})
    return pd.DataFrame(rows)
def _best_pair(z,mask):
    pairkey=["gene","lead_snp"];q=z.loc[mask].copy()
    if q.empty:return q
    return q.sort_values(pairkey+["bic","pvalue","lag_order"],na_position="last").drop_duplicates(pairkey,keep="first")
def replication_tables(rep,disc_all,candidates,cohort):
    key=["gene","lead_snp","lag_order"];disc_cols=key+[c for c in ["pvalue","q_bh_global","bic","coef_sum","ar_pvalue","ar_q_bh_global","discovery_fdr"] if c in disc_all.columns];z=add_fdr(rep);ren={"pvalue":"discovery_pvalue","q_bh_global":"discovery_q_bh_global","bic":"discovery_bic","coef_sum":"discovery_coef_sum","ar_pvalue":"discovery_ar_pvalue","ar_q_bh_global":"discovery_ar_q_bh_global","discovery_fdr":"discovery_lag_fdr_significant"};z=z.merge(disc_all[disc_cols].rename(columns=ren),on=key,how="left",validate="one_to_one");z=z.merge(candidates,on=["gene","lead_snp","artifact_id"],how="left",validate="many_to_one");z["nominal_p05"]=pd.to_numeric(z.pvalue,errors="coerce")<0.05;z["direction_concordant"]=np.sign(pd.to_numeric(z.discovery_coef_sum,errors="coerce"))==np.sign(pd.to_numeric(z.coef_sum,errors="coerce"));z["discovery_lag_fdr_significant"]=z.discovery_lag_fdr_significant.fillna(False).astype(bool);z["primary_same_discovery_significant_lag_nominal_p05"]=z.discovery_lag_fdr_significant&z.nominal_p05;z["primary_same_discovery_significant_lag_direction_concordant"]=z.primary_same_discovery_significant_lag_nominal_p05&z.direction_concordant.fillna(False);z["replication_AR_nominal_support"]=pd.to_numeric(z.ar_pvalue,errors="coerce")<0.05;z["replication_cohort"]=cohort;pairkey=["gene","lead_snp"];nlag=z.groupby(pairkey)["lag_order"].transform("nunique").clip(lower=1);z["within_pair_bonferroni_threshold"]=0.05/nlag;z["any_lag_within_pair_bonferroni_p05"]=pd.to_numeric(z.pvalue,errors="coerce")<z.within_pair_bonferroni_threshold;primary_nom=_best_pair(z,z.primary_same_discovery_significant_lag_nominal_p05);primary_dir=_best_pair(z,z.primary_same_discovery_significant_lag_direction_concordant);any_nom=_best_pair(z,z.nominal_p05);any_dir=_best_pair(z,z.nominal_p05&z.direction_concordant.fillna(False));any_bonf=_best_pair(z,z.any_lag_within_pair_bonferroni_p05);return z,primary_nom,primary_dir,any_nom,any_dir,any_bonf
def gene_summary(pair_final):
    if pair_final.empty:return pd.DataFrame(columns=["gene","n_independent_IVs_tested","n_discovery_FDR_IVs","any_external_replication_nominal_p05","n_external_replicated_independent_IVs"])
    reps=[c for c in pair_final.columns if c.endswith("_replicated_nominal_p05") and "any_lag" not in c];rows=[]
    for g,z in pair_final.groupby("gene"):
        r={"gene":g,"n_independent_IVs_tested":int(z.lead_snp.nunique()),"n_discovery_FDR_IVs":int(z.lead_snp.nunique())}
        for c in reps:r[c.replace("_replicated_nominal_p05","_n_replicated_IVs")]=int(z.loc[z[c].fillna(False).astype(bool),"lead_snp"].nunique())
        r["any_external_replication_nominal_p05"]=bool(z.any_external_replication_nominal_p05.fillna(False).any());r["n_external_replicated_independent_IVs"]=int(z.loc[z.any_external_replication_nominal_p05.fillna(False).astype(bool),"lead_snp"].nunique());rows.append(r)
    return pd.DataFrame(rows)
def main():
    ap=argparse.ArgumentParser(description="Unified temporal-tQTL single-IV lagged phenotype analysis with resumable discovery and pair-level all-lag external replication.");ap.add_argument("--root",default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt");ap.add_argument("--analysis-name",default="updrs3");ap.add_argument("--phenotype-cov",required=True);ap.add_argument("--outcome-col",default="mds_updrs_part_iii_summary_score");ap.add_argument("--candidate-file",default="");ap.add_argument("--discovery-cohort",default="PP");ap.add_argument("--replication-cohorts",default="PD,SU");ap.add_argument("--replication-pooled-groups",default="PD+SU");ap.add_argument("--max-lag",type=int,default=4);ap.add_argument("--min-participants",type=int,default=50);ap.add_argument("--fdr-alpha",type=float,default=0.05);ap.add_argument("--time-varying-covariates",default="upd23a_medication_for_pd");ap.add_argument("--extra-covariates",default="",help="Comma-separated mandatory numeric fixed covariates, e.g. eduyrs");ap.add_argument("--use-genotype-pcs",action=argparse.BooleanOptionalAction,default=True);ap.add_argument("--pc-file",default="");ap.add_argument("--pc-manifest",default="");ap.add_argument("--adjust-current-age-cubic",action=argparse.BooleanOptionalAction,default=False);ap.add_argument("--require-stage1-followup-support",action=argparse.BooleanOptionalAction,default=True);ap.add_argument("--require-stage1-time-support",dest="legacy_require_stage1_time_support",action=argparse.BooleanOptionalAction,default=None,help=argparse.SUPPRESS);ap.add_argument("--resume-discovery",action=argparse.BooleanOptionalAction,default=True);ap.add_argument("--resume-replication",action=argparse.BooleanOptionalAction,default=True);ap.add_argument("--ar-mode",choices=["all","off"],default="all",help="all=compute AR-type weak-IV robustness for every lag model; off=skip AR during high-throughput primary screening (primary lagged-association results are unchanged)");ap.add_argument("--detail-mode",choices=["full","minimal"],default="full",help="full=also compute cluster SE/P for coefficient summaries; minimal=skip these non-primary details during high-throughput screening");ap.add_argument("--task-chunksize",type=int,default=8,help="ProcessPool task chunk size; 4-16 is usually efficient");ap.add_argument("--n-jobs",type=int,default=16);ap.add_argument("--out-dir",default="");a=ap.parse_args()
    if a.legacy_require_stage1_time_support is not None:a.require_stage1_followup_support=bool(a.legacy_require_stage1_time_support)
    root=Path(a.root);out=Path(a.out_dir) if a.out_dir else root/"single_iv_causal"/a.analysis_name/"temporal_causal";out.mkdir(parents=True,exist_ok=True);cf=Path(a.candidate_file) if a.candidate_file else root/"single_iv_causal"/"iv_validation"/"single_iv_primary_eligible.csv";cand=pd.read_csv(cf,low_memory=False);req={"gene","lead_snp","artifact_id"};miss=req-set(cand.columns)
    if miss:raise ValueError(f"candidate file missing {sorted(miss)}")
    cov=read_cov(a.phenotype_cov,a.outcome_col);selected_pcs={};pc_file=a.pc_file or str(root/"single_iv_causal"/a.analysis_name/"genotype_pcs"/"single_iv_genotype_pcs.csv");pc_manifest=a.pc_manifest or str(root/"single_iv_causal"/a.analysis_name/"genotype_pcs"/"pc_selection.json")
    if a.use_genotype_pcs:cov,selected_pcs=add_pcs(cov,pc_file,pc_manifest)
    extra_cov=csv_list(a.extra_covariates);missing_extra=[c for c in extra_cov if c not in cov.columns];
    if missing_extra:raise ValueError(f"Required extra covariates missing: {missing_extra}")
    for c in extra_cov:
        cov[c]=pd.to_numeric(cov[c],errors="coerce")
        if cov[c].notna().sum()==0:raise ValueError(f"Extra covariate {c} has no numeric values")
    cache=out/"phenotype_with_covariates.parquet";cov.to_parquet(cache,index=False,compression="zstd");tv=[c for c in csv_list(a.time_varying_covariates) if c in cov.columns];pred_dir=root/"single_iv_causal"/a.analysis_name/"single_iv_genetic_component";component_manifest_file=root/"single_iv_causal"/a.analysis_name/"single_iv_genetic_component_manifest.json";component_manifest=json.loads(component_manifest_file.read_text()) if component_manifest_file.exists() else {};stage1_followup_min=component_manifest.get("stage1_followup_min");stage1_followup_max=component_manifest.get("stage1_followup_max");
    if a.require_stage1_followup_support and (stage1_followup_min is None or stage1_followup_max is None):raise RuntimeError("Stage-I follow-up support is not encoded in single_iv_genetic_component_manifest.json; rerun Stage-II step 05 with explicit --stage1-followup-min/--stage1-followup-max")
    ivs=[]
    for r in cand.itertuples(index=False):
        pf=pred_dir/f"{r.artifact_id}.parquet"
        if pf.exists():ivs.append((str(r.gene),str(r.lead_snp),str(r.artifact_id),pf))
    present={x[2] for x in ivs};cand.loc[~cand.artifact_id.astype(str).isin(present),["gene","lead_snp","artifact_id"]].to_csv(out/"missing_component_pairs.csv",index=False);allf,sigf,self=discovery_files(out);resume_payload=None
    discovery_expected={"complete":True,"discovery_cohort":a.discovery_cohort,"max_lag":int(a.max_lag),"min_participants":int(a.min_participants),"fdr_alpha":float(a.fdr_alpha),"require_stage1_followup_support":bool(a.require_stage1_followup_support),"stage1_followup_min":stage1_followup_min,"stage1_followup_max":stage1_followup_max,"adjust_current_age_cubic":bool(a.adjust_current_age_cubic),"use_genotype_pcs":bool(a.use_genotype_pcs),"ar_mode":str(a.ar_mode),"detail_mode":str(a.detail_mode),"extra_covariates":extra_cov}
    (out/"requested_run_config.json").write_text(json.dumps(discovery_expected|{"replication_cohorts":csv_list(a.replication_cohorts),"replication_pooled_groups":semi_list(a.replication_pooled_groups),"candidate_file":str(cf.resolve()),"outcome_col":a.outcome_col,"task_chunksize":int(a.task_chunksize),"detail_mode":str(a.detail_mode),"extra_covariates":extra_cov},indent=2))
    if a.resume_discovery:
        ok,resume_payload=valid_discovery_checkpoint(allf,sigf,self,cand,discovery_expected)
    else:ok=False
    errs=[]
    if ok:
        disc_all,disc_sig,selected=resume_payload;print(f"[RESUME] Reusing completed PPMI discovery: {allf}",flush=True);print(f"[RESUME] discovery rows={len(disc_all):,}, significant lag models={len(disc_sig):,}, significant IV-gene pairs={selected[['gene','lead_snp']].drop_duplicates().shape[0]:,}",flush=True)
    else:
        tasks=[(g,s,a0,str(pf),str(cache),a.discovery_cohort,a.outcome_col,a.max_lag,tv,a.min_participants,a.adjust_current_age_cubic,a.require_stage1_followup_support,a.use_genotype_pcs,selected_pcs,a.ar_mode,a.detail_mode,extra_cov) for g,s,a0,pf in ivs];disc,derr=run_tasks(tasks,a.n_jobs,"PPMI discovery",a.task_chunksize);errs.extend(derr)
        if disc.empty:raise RuntimeError("No discovery models were produced")
        disc=disc.merge(cand,on=["gene","lead_snp","artifact_id"],how="left",validate="many_to_one");disc_all,disc_sig,selected=select_discovery(disc,a.fdr_alpha);disc_all.to_csv(allf,index=False,compression="gzip");disc_sig.to_csv(sigf,index=False);selected.to_csv(self,index=False);selected.to_csv(out/"discovery_selected_models.csv",index=False);meta={"complete":True,"discovery_cohort":a.discovery_cohort,"max_lag":a.max_lag,"min_participants":a.min_participants,"fdr_alpha":a.fdr_alpha,"require_stage1_followup_support":bool(a.require_stage1_followup_support),"stage1_followup_min":stage1_followup_min,"stage1_followup_max":stage1_followup_max,"adjust_current_age_cubic":bool(a.adjust_current_age_cubic),"use_genotype_pcs":bool(a.use_genotype_pcs),"ar_mode":str(a.ar_mode),"detail_mode":str(a.detail_mode),"extra_covariates":extra_cov,"n_discovery_rows":int(len(disc_all)),"n_significant_lag_models":int(len(disc_sig)),"n_significant_IV_gene_pairs":int(selected[['gene','lead_snp']].drop_duplicates().shape[0]) if len(selected) else 0};(out/"discovery_checkpoint.json").write_text(json.dumps(meta,indent=2))
    if not (out/"discovery_selected_models.csv").exists():selected.to_csv(out/"discovery_selected_models.csv",index=False)
    rep_targets=make_replication_targets(disc_sig);rep_targets.to_csv(out/"replication_targets.csv",index=False);rep_specs=csv_list(a.replication_cohorts)+semi_list(a.replication_pooled_groups);rep_results={}
    if not rep_targets.empty:
        tg=set(zip(rep_targets.gene.astype(str),rep_targets.lead_snp.astype(str)));ivmap={(g,s):(g,s,a0,pf) for g,s,a0,pf in ivs}
        for cohort in rep_specs:
            lab=label(cohort);rawf=out/f"replication_{lab}_all_lags_for_discovery_significant_IV_gene_pairs.csv.gz";rt=[(g,s,a0,str(pf),str(cache),cohort,a.outcome_col,a.max_lag,tv,a.min_participants,a.adjust_current_age_cubic,a.require_stage1_followup_support,a.use_genotype_pcs,selected_pcs,a.ar_mode,a.detail_mode,extra_cov) for (g,s),(g,s,a0,pf) in ivmap.items() if (g,s) in tg]
            allrep=None
            rep_meta=out/f"replication_{lab}_checkpoint.json";rep_expected={"complete":True,"cohort":cohort,"max_lag":int(a.max_lag),"min_participants":int(a.min_participants),"require_stage1_followup_support":bool(a.require_stage1_followup_support),"stage1_followup_min":stage1_followup_min,"stage1_followup_max":stage1_followup_max,"adjust_current_age_cubic":bool(a.adjust_current_age_cubic),"use_genotype_pcs":bool(a.use_genotype_pcs),"ar_mode":str(a.ar_mode),"detail_mode":str(a.detail_mode),"extra_covariates":extra_cov,"n_target_IV_gene_pairs":int(len(tg))}
            if a.resume_replication:
                rep_ok,allrep=valid_replication_checkpoint(rawf,rep_meta,rep_expected)
                if rep_ok:print(f"[RESUME] Reusing completed replication cohort {cohort}: {rawf}",flush=True)
            if allrep is None:
                allrep,rerr=run_tasks(rt,a.n_jobs,f"replication {cohort}",a.task_chunksize);errs.extend(rerr)
                if allrep.empty:continue
                allrep.to_csv(rawf,index=False,compression="gzip");rep_meta.write_text(json.dumps(rep_expected|{"n_models":int(len(allrep)),"replication_design":"all lags 1..max_lag are rerun for every IV-gene pair with >=1 discovery BH-FDR-significant lag"},indent=2))
            full,primary_nom,primary_dir,any_nom,any_dir,any_bonf=replication_tables(allrep,disc_all,cand,cohort);full.to_csv(out/f"replication_{lab}_all_lags_annotated.csv.gz",index=False,compression="gzip");primary_nom.to_csv(out/f"replication_{lab}_PRIMARY_same_discovery_significant_lag_nominal_p05_bestBIC.csv",index=False);primary_dir.to_csv(out/f"replication_{lab}_PRIMARY_same_discovery_significant_lag_direction_concordant_bestBIC.csv",index=False);any_nom.to_csv(out/f"replication_{lab}_SENSITIVITY_any_lag_nominal_p05_bestBIC.csv",index=False);any_dir.to_csv(out/f"replication_{lab}_SENSITIVITY_any_lag_nominal_p05_direction_concordant_bestBIC.csv",index=False);any_bonf.to_csv(out/f"replication_{lab}_SENSITIVITY_any_lag_within_pair_bonferroni_p05_bestBIC.csv",index=False);rep_results[cohort]=(full,primary_nom,primary_dir,any_nom,any_dir,any_bonf)
    final=selected.copy();pairkey=["gene","lead_snp"]
    for cohort,(full,primary_nom,primary_dir,any_nom,any_dir,any_bonf) in rep_results.items():
        lab=label(cohort);pset=set(zip(primary_nom.gene.astype(str),primary_nom.lead_snp.astype(str))) if len(primary_nom) else set();pdset=set(zip(primary_dir.gene.astype(str),primary_dir.lead_snp.astype(str))) if len(primary_dir) else set();aset=set(zip(any_nom.gene.astype(str),any_nom.lead_snp.astype(str))) if len(any_nom) else set();abset=set(zip(any_bonf.gene.astype(str),any_bonf.lead_snp.astype(str))) if len(any_bonf) else set();keys=list(zip(final.gene.astype(str),final.lead_snp.astype(str)));final[f"{lab}_replicated_nominal_p05"]=[k in pset for k in keys];final[f"{lab}_replicated_direction_concordant"]=[k in pdset for k in keys];final[f"{lab}_any_lag_nominal_p05_sensitivity"]=[k in aset for k in keys];final[f"{lab}_any_lag_within_pair_bonferroni_p05_sensitivity"]=[k in abset for k in keys]
        if len(primary_nom):
            nm=primary_nom[pairkey+["lag_order","pvalue","coef_sum","ar_pvalue","n_obs","n_participants"]].rename(columns={c:f"{lab}_{c}" for c in ["lag_order","pvalue","coef_sum","ar_pvalue","n_obs","n_participants"]});final=final.merge(nm,on=pairkey,how="left",validate="one_to_one")
    final["any_external_replication_nominal_p05"]=False
    for c in rep_results:final["any_external_replication_nominal_p05"]|=final[f"{label(c)}_replicated_nominal_p05"].fillna(False).astype(bool)
    if "ar_pvalue" in final.columns:final["discovery_AR_nominal_support"]=pd.to_numeric(final.ar_pvalue,errors="coerce")<0.05
    final["first_stage_F_ge_10_descriptive"]=pd.to_numeric(final.get("ppmi_first_stage_F",np.nan),errors="coerce")>=10;final.to_csv(out/"final_IV_pair_summary.csv",index=False);final.to_csv(out/"final_summary.csv",index=False);gene_summary(final).to_csv(out/"final_gene_summary.csv",index=False);pd.DataFrame(errs).to_csv(out/"errors.csv",index=False)
    summary={"analysis_name":a.analysis_name,"architecture":"unified temporal cis-tQTL single-IV 2sGen-GPS; no dynamic/additive causal branches; no multi-SNP expression predictor","instrument_selection":"PPMI temporal FDR<0.05 -> same SNP-gene PDBP temporal P<0.05 -> trajectory_r>0 -> PPMI LD pruning; each independent IV-gene pair remains separate","fdr_primary":"one BH-FDR across the unified temporal-tQTL IV x lag discovery family","discovery_resume":bool(ok),"replication_target":"IV-gene pair enters replication if >=1 PPMI lag is BH-FDR significant","replication_execution":"for every target IV-gene pair, all lags 1..max_lag are rerun independently in each external cohort","replication_primary":"nominal P<0.05 at a lag that was BH-FDR significant in discovery; direction concordance is descriptive","replication_sensitivity":"any replication lag P<0.05 and within-pair Bonferroni across all rerun lags are also reported","weak_iv_sensitivity":("Anderson-Rubin-type reduced-form participant-cluster robust F using IV dosage x cubic follow-up-time basis for every temporal tQTL" if a.ar_mode=="all" else "deferred/not computed in this high-throughput run; primary lagged-association inference is unchanged"),"detail_mode":str(a.detail_mode),"stage1_followup_support_primary":bool(a.require_stage1_followup_support),"stage1_followup_min":stage1_followup_min,"stage1_followup_max":stage1_followup_max,"n_eligible_IV_pairs":int(len(cand)),"n_eligible_genes":int(cand.gene.nunique()),"n_discovery_IV_pairs":int(selected[["gene","lead_snp"]].drop_duplicates().shape[0]) if len(selected) else 0,"n_discovery_genes":int(selected.gene.nunique()) if len(selected) else 0,"n_replication_target_IV_pairs":int(rep_targets[["gene","lead_snp"]].drop_duplicates().shape[0]) if len(rep_targets) else 0,"n_external_replicated_IV_pairs_primary":int(final.loc[final.any_external_replication_nominal_p05,["gene","lead_snp"]].drop_duplicates().shape[0]) if len(final) else 0,"n_external_replicated_genes_primary":int(final.loc[final.any_external_replication_nominal_p05,"gene"].nunique()) if len(final) else 0};(out/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
