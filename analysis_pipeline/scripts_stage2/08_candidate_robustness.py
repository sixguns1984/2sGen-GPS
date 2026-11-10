#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2,norm
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

_WORKER_COV=None
_WORKER_SELECTED_PCS={}
_WORKER_COHORT_CACHE={}

def _num(x):return pd.to_numeric(x,errors="coerce")
def _p2(z):return float(2*norm.sf(abs(float(z)))) if np.isfinite(z) else np.nan
def csv_list(x):return [z.strip() for z in str(x).split(",") if z.strip()]
def cohort_members(x):return [z.strip() for z in str(x).split("+") if z.strip()]
def label(x):return str(x).replace("+","_PLUS_").replace("/","_").replace(" ","")

def read_cov(path,outcome_col):
    x=pd.read_csv(path,low_memory=False);x=x.drop(columns=[c for c in x.columns if str(c).startswith("Unnamed:")],errors="ignore")
    for c in ["cohort","participant_id","time",outcome_col]:
        if c not in x.columns:raise ValueError(f"Phenotype file missing {c}")
    x["cohort"]=x.cohort.astype(str);x["participant_id"]=x.participant_id.astype(str);x["time"]=_num(x.time);x[outcome_col]=_num(x[outcome_col]);x=x.dropna(subset=["participant_id","time",outcome_col]).sort_values(["cohort","participant_id","time"],kind="mergesort").drop_duplicates(["cohort","participant_id","time"],keep="first").reset_index(drop=True);return x

def add_pcs(cov,pc_file,pc_manifest):
    if not Path(pc_file).exists():raise FileNotFoundError(pc_file)
    pc=pd.read_csv(pc_file,low_memory=False);pc["cohort"]=pc.cohort.astype(str);pc["participant_id"]=pc.participant_id.astype(str);manifest=json.loads(Path(pc_manifest).read_text()) if Path(pc_manifest).exists() else {};selected={}
    for c,r in manifest.get("cohorts",{}).items():selected[str(c)]=[str(z) for z in r.get("selected_columns",[])]
    allcols=sorted({z for v in selected.values() for z in v},key=lambda s:int(s.replace("gPC","")) if s.startswith("gPC") and s.replace("gPC","").isdigit() else s);z=pc[[c for c in ["cohort","participant_id"]+allcols if c in pc.columns]].drop_duplicates(["cohort","participant_id"]);return cov.merge(z,on=["cohort","participant_id"],how="left",validate="many_to_one"),selected

def load_iv(pred_file,cov,cohort,require_support=False):
    wanted=["cohort","participant_id","time","gene","lead_snp","artifact_id","grex_single_iv","iv_dosage","stage1_followup_min","stage1_followup_max","within_stage1_followup_support"]
    try:p=pd.read_parquet(pred_file,columns=wanted)
    except Exception:
        p=pd.read_parquet(pred_file);p=p[[c for c in wanted if c in p.columns]].copy()
    members=cohort_members(cohort);c=cov[cov.cohort.astype(str).isin(members)].copy();p=p[p.cohort.astype(str).isin(members)].copy();keys=["cohort","participant_id","time"]
    for z in (c,p):z["cohort"]=z.cohort.astype(str);z["participant_id"]=z.participant_id.astype(str);z["_timekey"]=_num(z.time).round(10)
    mkeys=["cohort","participant_id","_timekey"]
    if c.duplicated(mkeys).any() or p.duplicated(mkeys).any():raise ValueError("duplicate cohort-participant-time visit keys")
    vals=[x for x in ["gene","lead_snp","artifact_id","grex_single_iv","iv_dosage","stage1_followup_min","stage1_followup_max","within_stage1_followup_support"] if x in p.columns];d=c.merge(p[mkeys+vals],on=mkeys,how="inner",validate="one_to_one").drop(columns="_timekey")
    if require_support:
        if "within_stage1_followup_support" not in d.columns:raise RuntimeError("follow-up support requested but component lacks within_stage1_followup_support")
        d=d[d.within_stage1_followup_support.fillna(False).astype(bool)].copy()
    return d

def cohort_baseline_covariates(d,cohort,selected_pcs,use_pcs):
    out=[];members=cohort_members(cohort)
    if use_pcs:
        if len(members)==1:out += [c for c in selected_pcs.get(members[0],[]) if c in d.columns]
        else:
            for m in members:
                for pc in selected_pcs.get(m,[]):
                    if pc not in d.columns:continue
                    cc=f"{m}_{pc}";d[cc]=np.where(d.cohort.astype(str).eq(m),_num(d[pc]),0.0);out.append(cc)
    if len(members)>1:
        ref=members[0]
        for m in members[1:]:cc=f"cohort_{m}_vs_{ref}";d[cc]=d.cohort.astype(str).eq(m).astype(float);out.append(cc)
    return d,out

def _init_worker(cov_cache,selected_pcs):
    global _WORKER_COV,_WORKER_SELECTED_PCS,_WORKER_COHORT_CACHE
    _WORKER_COV=pd.read_parquet(cov_cache);_WORKER_SELECTED_PCS=selected_pcs or {};_WORKER_COHORT_CACHE={}

def _worker_cov(cohort):
    global _WORKER_COV,_WORKER_COHORT_CACHE
    if cohort not in _WORKER_COHORT_CACHE:_WORKER_COHORT_CACHE[cohort]=_WORKER_COV[_WORKER_COV.cohort.astype(str).isin(cohort_members(cohort))].copy()
    return _WORKER_COHORT_CACHE[cohort]

def _fit_task(t):
    from gengps.amp_pd.single_iv_causal import fit_single_iv_arx
    gene,snp,aid,pf,cohort,lag,outcome,tv,min_part,age_cubic,require_support,use_pcs=t
    try:
        cov=_worker_cov(cohort);d=load_iv(pf,cov,cohort,require_support);d,base_cov=cohort_baseline_covariates(d,cohort,_WORKER_SELECTED_PCS,use_pcs);r=fit_single_iv_arx(d,outcome,"grex_single_iv",int(lag),baseline_covariates=base_cov,time_varying_covariates=tv,auto_clinical_covariates=True,min_participants=min_part,adjust_current_age_cubic=age_cubic,ar_time_powers=(0,1,2,3),compute_ar=True,compute_cluster_details=True)
        row={"gene":gene,"lead_snp":snp,"artifact_id":aid,"cohort":cohort,"lag_order":int(lag),"ok":r.ok,"reason":r.reason,"n_obs":r.n_obs,"n_participants":r.n_participants,"pvalue":r.pvalue,"joint_F":r.joint_F,"joint_df1":r.joint_df1,"joint_df2":r.joint_df2,"exposure_rank":r.exposure_rank,"bic":r.bic,"partial_r2":r.partial_r2,"coef_sum":r.coef_sum,"coef_sum_se_cluster":r.coef_sum_se_cluster,"coef_sum_p_cluster":r.coef_sum_p_cluster,"coef_lag1":r.coef_lag1,"coef_lag1_se_cluster":r.coef_lag1_se_cluster,"coef_lag1_p_cluster":r.coef_lag1_p_cluster,"ar_pvalue":r.ar_pvalue,"ar_F":r.ar_F,"ar_df1":r.ar_df1,"ar_df2":r.ar_df2,"ar_rank":r.ar_rank,"exposure_condition":r.exposure_condition,"ar_condition":r.ar_condition}
        for j,(b,se,p) in enumerate(zip(r.exposure_coefs,r.exposure_ses,r.exposure_pvalues),1):row[f"coef_x_lag{j}"]=b;row[f"se_x_lag{j}"]=se;row[f"p_x_lag{j}"]=p
        return row,None
    except Exception as e:return None,{"gene":gene,"lead_snp":snp,"artifact_id":aid,"cohort":cohort,"lag_order":lag,"error":repr(e),"traceback":traceback.format_exc()}

def run_tasks(tasks,cov_cache,selected_pcs,n_jobs,chunksize,label_):
    rows=[];errs=[]
    if not tasks:return pd.DataFrame(),errs
    if n_jobs<=1:
        _init_worker(cov_cache,selected_pcs);it=map(_fit_task,tasks)
        for i,(r,e) in enumerate(it,1):
            if r:rows.append(r)
            if e:errs.append(e)
            if i%100==0 or i==len(tasks):print(f"{label_} {i}/{len(tasks)}",flush=True)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs,initializer=_init_worker,initargs=(cov_cache,selected_pcs)) as ex:
            for i,(r,e) in enumerate(ex.map(_fit_task,tasks,chunksize=max(1,int(chunksize))),1):
                if r:rows.append(r)
                if e:errs.append(e)
                if i%100==0 or i==len(tasks):print(f"{label_} {i}/{len(tasks)}",flush=True)
    return pd.DataFrame(rows),errs

def fixed_effect(beta,se):
    b=np.asarray(beta,float);s=np.asarray(se,float);ok=np.isfinite(b)&np.isfinite(s)&(s>0);b=b[ok];s=s[ok]
    if len(b)<2:return None
    w=1/(s*s);mu=float(np.sum(w*b)/np.sum(w));sem=float(np.sqrt(1/np.sum(w)));z=mu/sem;Q=float(np.sum(w*(b-mu)**2));df=len(b)-1;qp=float(chi2.sf(Q,df));i2=float(max(0,(Q-df)/Q)) if Q>0 else 0.0
    return {"n":int(len(b)),"beta":mu,"se":sem,"p":_p2(z),"Q":Q,"Q_df":df,"Q_p":qp,"I2":i2}

def summarize_multi_iv(details):
    rows=[];loo=[]
    for gene,z in details.groupby("gene",sort=False):
        z=z.drop_duplicates("lead_snp").copy();b=_num(z.coef_sum).to_numpy(float);s=_num(z.coef_sum_se_cluster).to_numpy(float);ok=np.isfinite(b)&np.isfinite(s)&(s>0);fe=fixed_effect(b,s);r={"gene":gene,"n_independent_stage1_IVs":int(z.lead_snp.nunique()),"n_finite_commonlag_estimates":int(ok.sum())}
        if fe:r.update({f"multiIV_{k}":v for k,v in fe.items()})
        else:r.update({"multiIV_n":int(ok.sum()),"multiIV_beta":np.nan,"multiIV_se":np.nan,"multiIV_p":np.nan,"multiIV_Q":np.nan,"multiIV_Q_df":np.nan,"multiIV_Q_p":np.nan,"multiIV_I2":np.nan})
        r["multiIV_direction_concordance_fraction"]=float(max(np.mean(b[ok]>0),np.mean(b[ok]<0))) if ok.any() else np.nan
        if ok.sum()>=3:
            zz=z.loc[ok].reset_index(drop=True);loo_rows=[]
            for i in range(len(zz)):
                keep=np.ones(len(zz),bool);keep[i]=False;f2=fixed_effect(_num(zz.loc[keep,"coef_sum"]),_num(zz.loc[keep,"coef_sum_se_cluster"]));lr={"gene":gene,"left_out_snp":str(zz.loc[i,"lead_snp"]),"n_remaining":int(keep.sum()),"beta":f2["beta"] if f2 else np.nan,"se":f2["se"] if f2 else np.nan,"p":f2["p"] if f2 else np.nan};loo.append(lr);loo_rows.append(lr)
            lb=np.array([q["beta"] for q in loo_rows],float);lp=np.array([q["p"] for q in loo_rows],float);r["LOO_n"]=len(loo_rows);r["LOO_all_same_direction"]=bool(np.all(np.sign(lb)==np.sign(fe["beta"]))) if fe and np.all(np.isfinite(lb)) else np.nan;r["LOO_max_p"]=float(np.nanmax(lp)) if np.isfinite(lp).any() else np.nan;r["LOO_all_nominal_p05"]=bool(np.all(lp<0.05)) if np.all(np.isfinite(lp)) else np.nan
        else:r.update({"LOO_n":0,"LOO_all_same_direction":np.nan,"LOO_max_p":np.nan,"LOO_all_nominal_p05":np.nan})
        rows.append(r)
    return pd.DataFrame(rows),pd.DataFrame(loo)

def main():
    ap=argparse.ArgumentParser(description="Post-selection causal robustness for frozen PPMI + pooled PD+SU (UPDRS) or PD (MoCA) lag1-direction-consistent IV-gene set. Computes AR/full cluster details at the ORIGINAL selected lags, plus outcome-independent multiple-IV consistency using all Stage-I eligible IVs of candidate genes at one frozen common lag. No MR-Egger and no coloc.")
    ap.add_argument("--root",required=True);ap.add_argument("--analysis-name",required=True);ap.add_argument("--phenotype-cov",required=True);ap.add_argument("--outcome-col",required=True);ap.add_argument("--standard-dir",default="");ap.add_argument("--stage1-eligible",default="");ap.add_argument("--time-varying-covariates",default="upd23a_medication_for_pd");ap.add_argument("--use-genotype-pcs",action=argparse.BooleanOptionalAction,default=True);ap.add_argument("--pc-file",default="");ap.add_argument("--pc-manifest",default="");ap.add_argument("--adjust-current-age-cubic",action=argparse.BooleanOptionalAction,default=True);ap.add_argument("--require-stage1-followup-support",action=argparse.BooleanOptionalAction,default=False);ap.add_argument("--min-participants",type=int,default=50);ap.add_argument("--n-jobs",type=int,default=16);ap.add_argument("--task-chunksize",type=int,default=8);ap.add_argument("--out-dir",default="");a=ap.parse_args()
    root=Path(a.root);std=Path(a.standard_dir) if a.standard_dir else root/"single_iv_causal"/a.analysis_name/"standard_final";out=Path(a.out_dir) if a.out_dir else std/"robustness";out.mkdir(parents=True,exist_ok=True)
    master_file=std/f"{a.analysis_name}_downstream_IV_gene_master.csv";stage1_file=Path(a.stage1_eligible) if a.stage1_eligible else root/"single_iv_causal"/"iv_validation"/"single_iv_primary_eligible.csv"
    if not master_file.exists():raise FileNotFoundError(master_file)
    master=pd.read_csv(master_file,low_memory=False);stage1=pd.read_csv(stage1_file,low_memory=False);keys=["gene","lead_snp"]
    for x,name in [(master,"master"),(stage1,"stage1")]:
        if not set(keys+["artifact_id"] if name=="stage1" else keys).issubset(x.columns):raise ValueError(f"{name} missing required identifiers")
        x["gene"]=x.gene.astype(str);x["lead_snp"]=x.lead_snp.astype(str)
    cov=read_cov(a.phenotype_cov,a.outcome_col);selected_pcs={};pc_file=a.pc_file or str(root/"single_iv_causal"/a.analysis_name/"genotype_pcs"/"single_iv_genotype_pcs.csv");pc_manifest=a.pc_manifest or str(root/"single_iv_causal"/a.analysis_name/"genotype_pcs"/"pc_selection.json")
    if a.use_genotype_pcs:cov,selected_pcs=add_pcs(cov,pc_file,pc_manifest)
    cov_cache=out/"phenotype_with_covariates.parquet";cov.to_parquet(cov_cache,index=False,compression="zstd");tv=[c for c in csv_list(a.time_varying_covariates) if c in cov.columns];pred_dir=root/"single_iv_causal"/a.analysis_name/"single_iv_genetic_component";smap=stage1.drop_duplicates(keys).set_index(keys)
    errs=[];exact_tasks=[]
    cohort_specs=[("PP","PPMI_lag_order"),("PD","PD_lag_order")] if a.analysis_name.lower()=="moca" else [("PP","PPMI_lag_order"),("PD+SU","PD_PLUS_SU_lag_order")]
    for r in master.itertuples(index=False):
        key=(str(r.gene),str(r.lead_snp))
        if key not in smap.index:continue
        sr=smap.loc[key];aid=str(sr.artifact_id);pf=pred_dir/f"{aid}.parquet"
        if not pf.exists():errs.append({"gene":key[0],"lead_snp":key[1],"error":f"missing component {pf}"});continue
        for cohort,lc in cohort_specs:
            if lc not in master.columns:continue
            lag=getattr(r,lc,None)
            if pd.isna(lag):continue
            exact_tasks.append((key[0],key[1],aid,str(pf),cohort,int(lag),a.outcome_col,tv,a.min_participants,a.adjust_current_age_cubic,a.require_stage1_followup_support,a.use_genotype_pcs))
    exact,er=run_tasks(exact_tasks,str(cov_cache),selected_pcs,a.n_jobs,a.task_chunksize,"exact-pair robustness");errs.extend(er);exact.to_csv(out/"exact_candidate_selectedlag_robustness.csv",index=False)
    # Audit FAST numerical equivalence at frozen selected lags; cluster details/AR are the only intended additions.
    audits=[]
    for cohort,prefix in [("PP","PPMI"),("PD","PD"),("SU","SU"),("PD+SU","PD_PLUS_SU")]:
        z=exact[exact.cohort.eq(cohort)].copy()
        if z.empty:continue
        m=master[keys+[c for c in [f"{prefix}_pvalue",f"{prefix}_joint_F",f"{prefix}_bic",f"{prefix}_coef_sum"] if c in master.columns]].merge(z,on=keys,how="inner")
        for metric,newc in [("pvalue",f"{prefix}_pvalue"),("joint_F",f"{prefix}_joint_F"),("bic",f"{prefix}_bic"),("coef_sum",f"{prefix}_coef_sum")]:
            if newc in m.columns:
                d=np.abs(_num(m[metric])-_num(m[newc]));audits.append({"cohort":cohort,"metric":metric,"n":int(d.notna().sum()),"max_abs_diff":float(d.max()) if d.notna().any() else np.nan,"allclose_1e-8":bool(np.allclose(_num(m[metric]),_num(m[newc]),rtol=1e-8,atol=1e-8,equal_nan=True))})
    pd.DataFrame(audits).to_csv(out/"fast_vs_full_detail_numeric_audit.csv",index=False)
    # Common-lag multi-IV consistency: one reference lag/gene frozen from the strongest ORIGINAL full-universe candidate pair.
    rank=master.copy();sort=["gene"];asc=[True]
    for c in ["PPMI_q_bh_global","PPMI_pvalue","PPMI_bic"]:
        if c in rank.columns:rank[c]=_num(rank[c]);sort.append(c);asc.append(True)
    sort.append("lead_snp");asc.append(True);ref=rank.sort_values(sort,ascending=asc,na_position="last").drop_duplicates("gene")[["gene","lead_snp","PPMI_lag_order"]].rename(columns={"lead_snp":"reference_candidate_snp","PPMI_lag_order":"reference_lag"});ref.to_csv(out/"gene_reference_lag.csv",index=False)
    genes=set(ref.gene.astype(str));allivs=stage1[stage1.gene.astype(str).isin(genes)].copy().merge(ref[["gene","reference_lag"]],on="gene",how="left");multi_tasks=[]
    for r in allivs.itertuples(index=False):
        if pd.isna(r.reference_lag):continue
        pf=pred_dir/f"{r.artifact_id}.parquet"
        if pf.exists():multi_tasks.append((str(r.gene),str(r.lead_snp),str(r.artifact_id),str(pf),"PP",int(r.reference_lag),a.outcome_col,tv,a.min_participants,a.adjust_current_age_cubic,a.require_stage1_followup_support,a.use_genotype_pcs))
        else:errs.append({"gene":str(r.gene),"lead_snp":str(r.lead_snp),"error":f"missing component {pf}"})
    mdet,er=run_tasks(multi_tasks,str(cov_cache),selected_pcs,a.n_jobs,a.task_chunksize,"gene-wide common-lag robustness");errs.extend(er);mdet=mdet.merge(ref,on="gene",how="left");mdet.to_csv(out/"candidate_genes_all_stage1_IVs_commonlag_details.csv",index=False)
    msum,loo=summarize_multi_iv(mdet);msum=ref.merge(msum,on="gene",how="left");msum.to_csv(out/"multi_iv_gene_consistency_summary.csv",index=False);loo.to_csv(out/"multi_iv_leave_one_out.csv",index=False)
    pd.DataFrame(errs).to_csv(out/"errors.csv",index=False)
    manifest={"analysis_name":a.analysis_name,"selection_frozen_before_robustness":True,"exact_candidate_pairs":int(len(master)),"candidate_genes":int(master.gene.nunique()),"exact_selectedlag_models_fitted":int(len(exact)),"all_stage1_IV_commonlag_models":int(len(mdet)),"multi_iv_genes_ge2":int((msum.n_independent_stage1_IVs>=2).sum()) if "n_independent_stage1_IVs" in msum else 0,"robustness_only_not_selection":["cluster-robust coef_sum SE/P","Anderson-Rubin-type reduced-form P/F","multiple-IV fixed-effect pooled estimate","Cochran Q","I2","direction concordance","leave-one-IV-out"],"explicitly_not_in_standard_pipeline":["MR-Egger","scalarized MR","colocalization"],"reference_lag_rule":"Within each candidate gene, use the PPMI lag of the strongest newly selected pooled/PD replicated candidate pair ranked by q_bh_global, pvalue, BIC, SNP ID. All Stage-I eligible independent IVs for that gene are evaluated at this same lag.","errors":int(len(errs))};(out/"robustness_manifest.json").write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
