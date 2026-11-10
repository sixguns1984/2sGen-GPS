#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist, spearmanr, t as t_dist
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests

EPS=1e-12


def bh_adjust(p):
    p=np.asarray(p,float)
    q=np.full(len(p),np.nan,float)
    ok=np.isfinite(p)
    if ok.any(): q[ok]=multipletests(p[ok],method="fdr_bh")[1]
    return q


def clean_sex(x):
    s=pd.Series(x).astype(str).str.strip().str.lower()
    out=pd.Series(np.nan,index=s.index,dtype=float)
    out[s.isin(["male","m","1","1.0"])]=1.0
    out[s.isin(["female","f","0","0.0"])]=0.0
    return out


def within_participant_spearman(df,pred_col="grex_frozen",obs_col="observed_residual",pid_col="participant_id"):
    z=df[[pid_col,pred_col,obs_col]].copy()
    z[pred_col]=pd.to_numeric(z[pred_col],errors="coerce")
    z[obs_col]=pd.to_numeric(z[obs_col],errors="coerce")
    z=z.dropna()
    if z.empty:return np.nan
    z[pred_col]=z[pred_col]-z.groupby(pid_col)[pred_col].transform("mean")
    z[obs_col]=z[obs_col]-z.groupby(pid_col)[obs_col].transform("mean")
    ok=np.isfinite(z[pred_col])&np.isfinite(z[obs_col])
    if ok.sum()<3 or z.loc[ok,pred_col].nunique()<2 or z.loc[ok,obs_col].nunique()<2:return np.nan
    return float(spearmanr(z.loc[ok,pred_col],z.loc[ok,obs_col]).statistic)




def single_predictor_cluster_strength(df,pred_col="grex_frozen",obs_col="observed_residual",pid_col="participant_id"):
    z=df[[pid_col,pred_col,obs_col]].copy();z[pred_col]=pd.to_numeric(z[pred_col],errors="coerce");z[obs_col]=pd.to_numeric(z[obs_col],errors="coerce");z=z.replace([np.inf,-np.inf],np.nan).dropna()
    if len(z)<4 or z[pid_col].astype(str).nunique()<3 or z[pred_col].nunique()<2:return {"F":np.nan,"p":np.nan,"partial_R2":np.nan,"beta":np.nan,"n":len(z),"n_participants":z[pid_col].astype(str).nunique()}
    y=z[obs_col].to_numpy(float);x=z[pred_col].to_numpy(float);X=np.column_stack([np.ones(len(z)),x]);fit=OLS(y,X).fit();rob=fit.get_robustcov_results(cov_type="cluster",groups=z[pid_col].astype(str).to_numpy(),use_correction=True,df_correction=True)
    t=float(np.asarray(rob.tvalues,float)[1]);F=t*t;p=float(np.asarray(rob.pvalues,float)[1]);sse0=float(np.sum((y-np.mean(y))**2));sse1=float(np.sum(np.asarray(fit.resid,float)**2));r2=max(0.0,(sse0-sse1)/max(sse0,EPS))
    return {"F":F,"p":p,"partial_R2":r2,"beta":float(np.asarray(fit.params,float)[1]),"n":len(z),"n_participants":z[pid_col].astype(str).nunique()}

def sample_spearman(df,pred_col="grex_frozen",obs_col="observed_residual"):
    x=pd.to_numeric(df[pred_col],errors="coerce")
    y=pd.to_numeric(df[obs_col],errors="coerce")
    ok=np.isfinite(x)&np.isfinite(y)
    if ok.sum()<3 or x[ok].nunique()<2 or y[ok].nunique()<2:return np.nan
    return float(spearmanr(x[ok],y[ok]).statistic)


def find_first_existing(paths):
    for p in paths:
        p=Path(p)
        if p.exists():return p
    return None


def detect_gene_col(df):
    for c in ["gene","Ensembl_ID","gene_id","stage1_gene_id","ensembl_gene_id"]:
        if c in df.columns:return c
    cand=[c for c in df.columns if "gene" in c.lower()]
    if cand:return cand[0]
    raise ValueError(f"No gene column; columns={list(df.columns)}")


def detect_snp_col(df):
    for c in ["Variant_ID","variant_id","snp","SNP","rsid","id"]:
        if c in df.columns:return c
    cand=[c for c in df.columns if "variant" in c.lower() or "snp" in c.lower()]
    if cand:return cand[0]
    raise ValueError(f"No SNP column; columns={list(df.columns)}")


def detect_p_col(df):
    for c in ["Pvalue","pvalue","P_value","p_value","p","P"]:
        if c in df.columns:return c
    cand=[c for c in df.columns if c.lower() in {"pval","p_value"} or "pvalue" in c.lower()]
    if cand:return cand[0]
    raise ValueError(f"No P-value column; columns={list(df.columns)}")


def normalize_qtl_table(df):
    g=detect_gene_col(df);s=detect_snp_col(df);p=detect_p_col(df)
    out=df.copy()
    out["gene"]=out[g].astype(str)
    out["lead_snp"]=out[s].astype(str)
    out["qtl_p"]=pd.to_numeric(out[p],errors="coerce")
    return out


def choose_ppmi_lead_tqtl(tqtl):
    """Return the strongest PPMI temporal tQTL per gene (utility only).

    The production v4 pipeline does not collapse to one IV per gene; it retains all
    replicated LD-independent temporal cis-eQTLs.  This helper is kept only for small
    diagnostics/tests that need one representative temporal tQTL per gene.
    """
    x=normalize_qtl_table(tqtl)
    x=x[np.isfinite(x.qtl_p)].copy()
    x=x.sort_values(["gene","qtl_p","lead_snp"],kind="mergesort").drop_duplicates("gene",keep="first")
    x=x.rename(columns={"qtl_p":"ppmi_tqtl_p"})
    keep=["gene","lead_snp","ppmi_tqtl_p"]
    for c in ["basis_json"]+[c for c in x.columns if str(c).startswith("beta_g_")]:
        if c in x.columns and c not in keep:keep.append(c)
    return x[keep].copy()

def read_legacy_metric(metrics_file,cohort,level):
    p=Path(metrics_file)
    if not p.exists():return pd.DataFrame()
    x=pd.read_csv(p,low_memory=False)
    if "model" in x.columns:x=x[x.model.astype(str).eq("legacy_single_iv")]
    if "cohort" in x.columns:x=x[x.cohort.astype(str).str.upper().eq(str(cohort).upper())]
    if "level" in x.columns:x=x[x.level.astype(str).eq(level)]
    if x.empty:return x
    g=detect_gene_col(x)
    x=x.copy();x["gene"]=x[g].astype(str)
    if "legacy_lead_snp" in x.columns:x["lead_snp"]=x.legacy_lead_snp.astype(str)
    return x


def metric_value(row,names):
    for c in names:
        if c in row.index:
            v=pd.to_numeric(pd.Series([row[c]]),errors="coerce").iloc[0]
            if np.isfinite(v):return float(v)
    return np.nan


def merge_legacy_strength(candidates,legacy_dir,compute_missing_rho=True):
    """Merge historical legacy-single-IV metrics when they refer to the exact same SNP-gene pair.

    v3.1 may retain >1 independent IV per gene, while the historical ctar_legacy_only
    cache was built for one lead IV per gene.  Therefore a legacy metric is attached
    only when legacy_lead_snp matches the candidate SNP.  Non-lead independent IVs are
    kept with missing legacy metrics rather than inheriting the lead SNP's strength.
    """
    legacy=Path(legacy_dir)
    out=candidates.copy()
    ppfile=find_first_existing([legacy/"model_metrics_ppmi_legacy_training_apparent.csv",legacy/"model_metrics_ppmi_oof.csv"])
    pdfile=find_first_existing([legacy/"model_metrics_pdbp_external.csv",legacy/"model_metrics_external_all.csv"])
    pp=read_legacy_metric(ppfile,"PPMI","sample_training_apparent") if ppfile else pd.DataFrame()
    if pp.empty and ppfile:
        allpp=pd.read_csv(ppfile,low_memory=False)
        if "model" in allpp.columns:allpp=allpp[allpp.model.astype(str).eq("legacy_single_iv")]
        if "cohort" in allpp.columns:allpp=allpp[allpp.cohort.astype(str).str.upper().eq("PPMI")]
        if "level" in allpp.columns:allpp=allpp[allpp.level.astype(str).str.startswith("sample")]
        if not allpp.empty:
            g=detect_gene_col(allpp);allpp["gene"]=allpp[g].astype(str)
            if "legacy_lead_snp" in allpp.columns:allpp["lead_snp"]=allpp.legacy_lead_snp.astype(str)
            pp=allpp
    pds=read_legacy_metric(pdfile,"PDBP","sample") if pdfile else pd.DataFrame()
    pdw=read_legacy_metric(pdfile,"PDBP","within_participant") if pdfile else pd.DataFrame()
    def map_rows(df):
        if df.empty:return {}
        if "lead_snp" in df.columns:
            return {(str(r.gene),str(r.lead_snp)):r for _,r in df.drop_duplicates(["gene","lead_snp"]).iterrows()}
        return {(str(r.gene),""):r for _,r in df.drop_duplicates("gene").iterrows()}
    ppm=map_rows(pp);pdm=map_rows(pds);pdwm=map_rows(pdw)
    rows=[]
    for r in out.itertuples(index=False):
        g=str(r.gene);snp=str(r.lead_snp);base=dict(r._asdict())
        rp=ppm.get((g,snp));rd=pdm.get((g,snp));rw=pdwm.get((g,snp))
        base["ppmi_first_stage_partial_R2"]=metric_value(rp,["R2_frozen","partial_R2","oof_partial_R2"]) if rp is not None else np.nan
        base["ppmi_first_stage_F"]=metric_value(rp,["grex_score_F","first_stage_F"]) if rp is not None else np.nan
        base["ppmi_sample_rho"]=metric_value(rp,["spearman","oof_spearman_residual","pearson"]) if rp is not None else np.nan
        base["pdbp_frozen_R2"]=metric_value(rd,["R2_frozen","partial_R2"]) if rd is not None else np.nan
        base["pdbp_sample_rho"]=metric_value(rd,["spearman","pearson"]) if rd is not None else np.nan
        base["pdbp_within_rho"]=metric_value(rw,["spearman","oof_spearman_residual"]) if rw is not None else np.nan
        rows.append(base)
    out=pd.DataFrame(rows)
    # Do not use the historical gene-only prediction files for non-lead IVs.
    if compute_missing_rho:
        for i,r in out.iterrows():
            if np.isfinite(pd.to_numeric(pd.Series([r.get("pdbp_within_rho",np.nan)]),errors="coerce").iloc[0]):continue
            g=str(r.gene);snp=str(r.lead_snp)
            fpd=legacy/"pdbp_predictions"/f"{g}__legacy_single_iv.parquet"
            if not fpd.exists():continue
            try:
                z=pd.read_parquet(fpd)
                z_snp=str(z["legacy_lead_snp"].iloc[0]) if "legacy_lead_snp" in z.columns and len(z) else ""
                if z_snp and z_snp!=snp:continue
                if {"grex_frozen","observed_residual","participant_id"}.issubset(z.columns):out.at[i,"pdbp_within_rho"]=within_participant_spearman(z)
                if {"grex_frozen","observed_residual"}.issubset(z.columns):out.at[i,"pdbp_sample_rho"]=sample_spearman(z)
            except Exception:
                pass
    return out

def _orthogonal_cluster_f(y,X0,E,groups):
    y=np.asarray(y,float);X0=np.asarray(X0,float);E=np.asarray(E,float);groups=np.asarray(groups,dtype=str)
    if E.ndim==1:E=E[:,None]
    B0=np.linalg.lstsq(X0,E,rcond=None)[0]
    R=E-X0@B0
    if R.size==0:return {"ok":False,"reason":"empty_block"}
    U,s,_=np.linalg.svd(R,full_matrices=False)
    if len(s)==0 or not np.isfinite(s[0]) or s[0]<=EPS:return {"ok":False,"reason":"zero_rank"}
    tol=max(R.shape)*np.finfo(float).eps*float(s[0]);rank=int(np.sum(s>tol))
    if rank<1:return {"ok":False,"reason":"zero_rank"}
    Q=U[:,:rank];X=np.column_stack([X0,Q]);p0=X0.shape[1]
    fit=OLS(y,X).fit();rob=fit.get_robustcov_results(cov_type="cluster",groups=groups,use_correction=True,df_correction=True)
    theta=np.asarray(rob.params[p0:p0+rank],float);V=np.asarray(rob.cov_params(),float)[p0:p0+rank,p0:p0+rank]
    if not np.all(np.isfinite(V)):return {"ok":False,"reason":"nonfinite_cov"}
    cr=int(np.linalg.matrix_rank(V,tol=max(np.linalg.norm(V,ord=2),1.0)*1e-12))
    if cr<rank:return {"ok":False,"reason":f"singular_cov:{cr}/{rank}"}
    W=float(theta@np.linalg.pinv(V,rcond=1e-12)@theta);W=max(0.0,W)
    G=int(pd.Series(groups).nunique());df2=G-1
    if df2<1:return {"ok":False,"reason":"too_few_clusters"}
    F=float(W/rank);p=float(f_dist.sf(F,rank,df2));cond=float(s[0]/s[rank-1]) if s[rank-1]>0 else np.inf
    return {"ok":True,"reason":"ok","F":F,"W":W,"df1":rank,"df2":df2,"p":p,"rank":rank,"condition":cond}


def _prepare_visit_lags_all(base,outcome_col,exposure_col,max_lag,time_varying_covariates=()):
    """Prepare all cumulative lag columns once for lag orders 1..max_lag.

    v4.5 performance optimization: the previous implementation repeated the same
    sort/groupby/shift work independently for every lag order.  This routine is
    algebraically identical but creates the complete lag table once per IV-gene
    task and reuses it for all nested lag models.
    """
    d=base.copy();d["participant_id"]=d["participant_id"].astype(str);d["time"]=pd.to_numeric(d["time"],errors="coerce")
    if "duration" not in d.columns and "age" in d.columns and "AAO" in d.columns:d["duration"]=pd.to_numeric(d["age"],errors="coerce")-pd.to_numeric(d["AAO"],errors="coerce")
    d=d.sort_values(["participant_id","time"],kind="mergesort").drop_duplicates(["participant_id","time"],keep="first").copy();g=d.groupby("participant_id",sort=False);L=int(max_lag)
    for k in range(1,L+1):
        d[f"y_lag{k}"]=g[outcome_col].shift(k);d[f"x_lag{k}"]=g[exposure_col].shift(k);d[f"time_lag{k}"]=g["time"].shift(k)
        if "iv_dosage" in d.columns:d[f"iv_lag{k}"]=g["iv_dosage"].shift(k)
        for c in time_varying_covariates:
            if c in d.columns:d[f"{c}_lag{k}"]=g[c].shift(k)
        for c in ["AAO","sex","duration"]:
            if c in d.columns:d[f"{c}_lag{k}"]=g[c].shift(k)
    return d

def _prepare_visit_lags(base,outcome_col,exposure_col,lag_order,time_varying_covariates=()):
    d=_prepare_visit_lags_all(base,outcome_col,exposure_col,lag_order,time_varying_covariates)
    L=int(lag_order)
    for c in ["AAO","sex","duration"]:
        cc=f"{c}_lag{L}"
        if cc in d.columns:d[f"{c}_oldest_lag"]=d[cc]
    return d


@dataclass
class SingleIVFit:
    ok: bool
    reason: str
    lag_order: int
    n_obs: int
    n_participants: int
    pvalue: float
    joint_F: float
    joint_df1: int
    joint_df2: int
    exposure_rank: int
    bic: float
    partial_r2: float
    coef_sum: float
    coef_sum_se_cluster: float
    coef_sum_p_cluster: float
    coef_lag1: float
    coef_lag1_se_cluster: float
    coef_lag1_p_cluster: float
    exposure_coefs: list
    exposure_ses: list
    exposure_pvalues: list
    ar_pvalue: float
    ar_F: float
    ar_df1: int
    ar_df2: int
    ar_rank: int
    exposure_condition: float
    ar_condition: float

def _failed_fit(reason,lag_order,n_obs,n_participants,bic=np.nan):
    return SingleIVFit(ok=False,reason=str(reason),lag_order=int(lag_order),n_obs=int(n_obs),n_participants=int(n_participants),pvalue=np.nan,joint_F=np.nan,joint_df1=0,joint_df2=0,exposure_rank=0,bic=float(bic) if np.isfinite(bic) else np.nan,partial_r2=np.nan,coef_sum=np.nan,coef_sum_se_cluster=np.nan,coef_sum_p_cluster=np.nan,coef_lag1=np.nan,coef_lag1_se_cluster=np.nan,coef_lag1_p_cluster=np.nan,exposure_coefs=[],exposure_ses=[],exposure_pvalues=[],ar_pvalue=np.nan,ar_F=np.nan,ar_df1=0,ar_df2=0,ar_rank=0,exposure_condition=np.nan,ar_condition=np.nan)


def fit_single_iv_arx(base,outcome_col,exposure_col="grex_single_iv",lag_order=1,baseline_covariates=(),time_varying_covariates=(),auto_clinical_covariates=True,min_participants=20,adjust_current_age_cubic=False,ar_time_powers=(0,1,2,3),compute_ar=True,compute_cluster_details=True,_prepared=None):
    L=int(lag_order);d=_prepare_visit_lags(base,outcome_col,exposure_col,L,time_varying_covariates) if _prepared is None else _prepared
    nuisance=[]
    if auto_clinical_covariates:
        sexlag=f"sex_lag{L}";sexc=sexlag if sexlag in d.columns else ("sex_oldest_lag" if "sex_oldest_lag" in d.columns else ("sex" if "sex" in d.columns else None))
        if sexc:
            d["sex_binary_oldcompat"]=clean_sex(d[sexc]);nuisance.append("sex_binary_oldcompat")
        for c in ["AAO","duration"]:
            lagc=f"{c}_lag{L}";cc=lagc if lagc in d.columns else (f"{c}_oldest_lag" if f"{c}_oldest_lag" in d.columns else (c if c in d.columns else None))
            if cc:nuisance.append(cc)
        if adjust_current_age_cubic and "age" in d.columns:
            age=pd.to_numeric(d["age"],errors="coerce")+pd.to_numeric(d["time"],errors="coerce");mu=np.nanmean(age);sd=np.nanstd(age);z=(age-mu)/sd if np.isfinite(sd) and sd>0 else np.zeros(len(d));d["age_z"]=z;d["age_z2"]=z**2;d["age_z3"]=z**3;nuisance += ["age_z","age_z2","age_z3"]
    for c in baseline_covariates:
        if c in d.columns and c not in nuisance:nuisance.append(c)
    for k in range(1,L+1):
        nuisance.append(f"y_lag{k}")
        for c in time_varying_covariates:
            cc=f"{c}_lag{k}"
            if cc in d.columns:nuisance.append(cc)
    xcols=[f"x_lag{k}" for k in range(1,L+1)]
    need=[outcome_col,"participant_id"]+nuisance+xcols+[f"time_lag{k}" for k in range(1,L+1)]+[f"iv_lag{k}" for k in range(1,L+1)]
    z=d[[c for c in need if c in d.columns]].copy()
    for c in z.columns:
        if c!="participant_id":z[c]=pd.to_numeric(z[c],errors="coerce")
    z=z.replace([np.inf,-np.inf],np.nan).dropna()
    if z.empty:return _failed_fit("no_complete_rows",L,0,0)
    groups=z.participant_id.astype(str).to_numpy();G=int(pd.Series(groups).nunique())
    if G<int(min_participants):return _failed_fit(f"too_few_participants:{G}",L,len(z),G)
    y=z[outcome_col].to_numpy(float);Xn=z[nuisance].to_numpy(float) if nuisance else np.empty((len(z),0),float)
    if Xn.shape[1]:
        keep=np.array([np.nanstd(Xn[:,j])>1e-12 for j in range(Xn.shape[1])],bool);Xn=Xn[:,keep]
    X0=np.column_stack([np.ones(len(z)),Xn]);Xe=z[xcols].to_numpy(float);X1=np.column_stack([X0,Xe]);p0=X0.shape[1]
    fit0=OLS(y,X0).fit();fitraw=OLS(y,X1).fit();raw=np.asarray(fitraw.params,float);bse=np.asarray(fitraw.bse,float);pind=np.asarray(fitraw.pvalues,float)
    if compute_cluster_details:
        try:
            robraw=fitraw.get_robustcov_results(cov_type="cluster",groups=groups,use_correction=True,df_correction=True);Vrob=np.asarray(robraw.cov_params(),float);serob=np.asarray(robraw.bse,float);prob=np.asarray(robraw.pvalues,float)
        except Exception:
            Vrob=np.full((X1.shape[1],X1.shape[1]),np.nan,float);serob=np.full(X1.shape[1],np.nan,float);prob=np.full(X1.shape[1],np.nan,float)
    else:
        Vrob=np.full((X1.shape[1],X1.shape[1]),np.nan,float);serob=np.full(X1.shape[1],np.nan,float);prob=np.full(X1.shape[1],np.nan,float)
    formal=_orthogonal_cluster_f(y,X0,Xe,groups)
    if not formal["ok"]:return _failed_fit("exposure_"+formal["reason"],L,len(z),G,float(fitraw.bic))
    sse0=float(np.sum(np.asarray(fit0.resid,float)**2));sse1=float(np.sum(np.asarray(fitraw.resid,float)**2));pr2=max(0.0,(sse0-sse1)/max(sse0,EPS))
    beta=raw[p0:p0+L];se=bse[p0:p0+L];pv=pind[p0:p0+L]
    Vexp=Vrob[p0:p0+L,p0:p0+L] if Vrob.shape[0]>=p0+L else np.full((L,L),np.nan,float);ones=np.ones(L,float)
    if np.all(np.isfinite(Vexp)):
        sum_var=float(ones@Vexp@ones);coef_sum_se_cluster=float(np.sqrt(max(sum_var,0.0))) if np.isfinite(sum_var) else np.nan
    else:coef_sum_se_cluster=np.nan
    coef_sum_val=float(np.sum(beta));coef_sum_t=coef_sum_val/coef_sum_se_cluster if np.isfinite(coef_sum_se_cluster) and coef_sum_se_cluster>0 else np.nan;coef_sum_p_cluster=float(2*t_dist.sf(abs(coef_sum_t),max(G-1,1))) if np.isfinite(coef_sum_t) else np.nan
    coef_lag1_se_cluster=float(serob[p0]) if len(serob)>p0 and np.isfinite(serob[p0]) else np.nan;coef_lag1_p_cluster=float(prob[p0]) if len(prob)>p0 and np.isfinite(prob[p0]) else np.nan
    # Anderson-Rubin-type reduced-form weak-IV robustness.  It does not enter the
    # primary lagged-association test, so v4.5 permits deferring this expensive block
    # during high-throughput screening (--ar-mode off) and running it later only for
    # prioritized candidates.
    if compute_ar:
        tmat=np.column_stack([z[f"time_lag{k}"].to_numpy(float) for k in range(1,L+1)])
        ivmat=np.column_stack([z[f"iv_lag{k}"].to_numpy(float) for k in range(1,L+1)])
        tvec=tmat.reshape(-1);mu=np.nanmean(tvec);sd=np.nanstd(tvec);sd=sd if np.isfinite(sd) and sd>0 else 1.0
        blocks=[];powers=tuple(int(p) for p in ar_time_powers)
        for k in range(L):
            tz=(tmat[:,k]-mu)/sd;g=ivmat[:,k]
            for p in powers:blocks.append(g*(tz**p))
        Zar=np.column_stack(blocks) if blocks else ivmat[:,:1];art=_orthogonal_cluster_f(y,X0,Zar,groups)
        if not art["ok"]:arp=np.nan;arf=np.nan;ard1=0;ard2=G-1;arr=0;arc=np.nan
        else:arp=art["p"];arf=art["F"];ard1=art["df1"];ard2=art["df2"];arr=art["rank"];arc=art["condition"]
    else:
        arp=np.nan;arf=np.nan;ard1=0;ard2=G-1;arr=0;arc=np.nan
    return SingleIVFit(True,"ok",L,len(z),G,formal["p"],formal["F"],formal["df1"],formal["df2"],formal["rank"],float(fitraw.bic),pr2,coef_sum_val,coef_sum_se_cluster,coef_sum_p_cluster,float(beta[0]),coef_lag1_se_cluster,coef_lag1_p_cluster,beta.tolist(),se.tolist(),pv.tolist(),arp,arf,ard1,ard2,arr,formal["condition"],arc)


def fit_all_lags(base,outcome_col,exposure_col,max_lag,baseline_covariates=(),time_varying_covariates=(),auto_clinical_covariates=True,min_participants=20,adjust_current_age_cubic=False,ar_time_powers=(0,1,2,3),compute_ar=True,compute_cluster_details=True):
    rows=[];prepared=_prepare_visit_lags_all(base,outcome_col,exposure_col,max_lag,time_varying_covariates)
    for L in range(1,int(max_lag)+1):
        r=fit_single_iv_arx(base,outcome_col,exposure_col,L,baseline_covariates,time_varying_covariates,auto_clinical_covariates,min_participants,adjust_current_age_cubic,ar_time_powers,compute_ar=compute_ar,compute_cluster_details=compute_cluster_details,_prepared=prepared)
        row={"lag_order":L,"ok":r.ok,"reason":r.reason,"n_obs":r.n_obs,"n_participants":r.n_participants,"pvalue":r.pvalue,"joint_F":r.joint_F,"joint_df1":r.joint_df1,"joint_df2":r.joint_df2,"exposure_rank":r.exposure_rank,"bic":r.bic,"partial_r2":r.partial_r2,"coef_sum":r.coef_sum,"coef_sum_se_cluster":r.coef_sum_se_cluster,"coef_sum_p_cluster":r.coef_sum_p_cluster,"coef_lag1":r.coef_lag1,"coef_lag1_se_cluster":r.coef_lag1_se_cluster,"coef_lag1_p_cluster":r.coef_lag1_p_cluster,"ar_pvalue":r.ar_pvalue,"ar_F":r.ar_F,"ar_df1":r.ar_df1,"ar_df2":r.ar_df2,"ar_rank":r.ar_rank,"exposure_condition":r.exposure_condition,"ar_condition":r.ar_condition}
        for j in range(int(max_lag)):
            row[f"coef_x_lag{j+1}"]=r.exposure_coefs[j] if j<len(r.exposure_coefs) else np.nan
            row[f"se_x_lag{j+1}"]=r.exposure_ses[j] if j<len(r.exposure_ses) else np.nan
            row[f"p_x_lag{j+1}"]=r.exposure_pvalues[j] if j<len(r.exposure_pvalues) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
