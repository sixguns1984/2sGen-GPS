from __future__ import annotations

"""Frozen single-IV temporal genetic-component utilities used by current Stage II.

The retired multi-tQTL Elastic-Net GReX training framework has been removed.  The
current pipeline treats every LD-independent temporal cis-eQTL IV-gene pair as a
separate instrument and reconstructs its frozen Stage-I MPTGA genetic component.
"""

from dataclasses import asdict,dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr

from ..basis import TimeBasis


def _safe_sd(x):
    x=np.asarray(x,float);s=float(np.nanstd(x));return s if np.isfinite(s) and s>1e-8 else 1.0


def add_current_age(meta: pd.DataFrame) -> pd.DataFrame:
    """Create current age for nuisance adjustment: baseline age + follow-up time."""
    z=meta.copy()
    if "age_at_baseline" not in z.columns:raise KeyError("metadata lacks age_at_baseline")
    if "followup_time" not in z.columns:raise KeyError("metadata lacks followup_time")
    z["current_age"]=pd.to_numeric(z["age_at_baseline"],errors="coerce")+pd.to_numeric(z["followup_time"],errors="coerce")
    return z


@dataclass
class CubicAxisBasis:
    center: float
    scale: float

    @classmethod
    def fit(cls,x):
        x=np.asarray(x,float);return cls(float(np.nanmean(x)),_safe_sd(x))

    def transform(self,x):
        z=(np.asarray(x,float)-self.center)/self.scale;return np.column_stack([z,z**2,z**3])


@dataclass
class LegacySingleIVModel:
    """Frozen Stage-I MPTGA genetic component for one temporal cis-eQTL IV-gene pair."""
    gene: str
    snp: str
    basis: dict
    beta_g: list[float]
    genotype_mode: str="dummies"
    source_pvalue: float|None=None
    qtl_method: str="submitted_followup_ctar_mptga"
    model_type: str="legacy_single_iv"

    def to_json(self,path):Path(path).write_text(json.dumps(asdict(self),indent=2))

    @classmethod
    def from_json(cls,path):return cls(**json.loads(Path(path).read_text()))

    @property
    def time_support(self):
        b=TimeBasis.from_dict(self.basis);return float(b.lower),float(b.upper)

    def predict_grex(self,genotype,meta,id_col="participant_id",followup_col="followup_time"):
        b=TimeBasis.from_dict(self.basis);t=pd.to_numeric(meta[followup_col],errors="coerce").to_numpy(float);B=b.transform(t)
        g=genotype.copy();g.index=g.index.astype(str);ids=meta[id_col].astype(str);gv=g.reindex(ids)[self.snp].to_numpy(float);beta=np.asarray(self.beta_g,float);k=B.shape[1]
        if len(beta)==k:channels=gv[:,None]
        elif len(beta)==2*k:
            src=pd.to_numeric(g[self.snp],errors="coerce").to_numpy(float);src_ok=np.isfinite(src)&np.isclose(src,np.rint(src),atol=1e-8);p1=float(np.mean(np.rint(src[src_ok])==1)) if src_ok.any() else 0.0;p2=float(np.mean(np.rint(src[src_ok])==2)) if src_ok.any() else 0.0;obs=np.isfinite(gv)&np.isclose(gv,np.rint(gv),atol=1e-8);channels=np.column_stack([(np.rint(gv)==1).astype(float),(np.rint(gv)==2).astype(float)]);channels[~obs,0]=p1;channels[~obs,1]=p2
        else:raise ValueError(f"single-IV coefficient length {len(beta)} incompatible with {k} time-basis columns")
        coef=beta.reshape(channels.shape[1],k);return np.einsum("ic,ik,ck->i",channels,B,coef)

    def support_mask(self,meta,followup_col="followup_time"):
        b=TimeBasis.from_dict(self.basis);return b.support_mask(pd.to_numeric(meta[followup_col],errors="coerce").to_numpy(float))


def legacy_single_iv_from_qtl_row(row,qtl_method="submitted_followup_ctar_mptga"):
    """Construct one frozen temporal genetic component from a finalized Stage-I row."""
    if isinstance(row,pd.Series):row=row.to_dict()
    if not row.get("basis_json"):raise ValueError("Stage-I row lacks basis_json required for single-IV prediction")
    beta_cols=[]
    for c in row:
        if str(c).startswith("beta_g_"):
            try:beta_cols.append((int(str(c).split("beta_g_")[-1]),c))
            except Exception:pass
    beta_cols=[c for _,c in sorted(beta_cols)];beta=[float(row[c]) for c in beta_cols if pd.notna(row[c])]
    if not beta:raise ValueError("Stage-I row lacks beta_g_* coefficients required for single-IV prediction")
    basis=row["basis_json"]
    if isinstance(basis,str):basis=json.loads(basis)
    return LegacySingleIVModel(gene=str(row["Ensembl_ID"]),snp=str(row["Variant_ID"]),basis=basis,beta_g=beta,genotype_mode="dummies" if len(beta)==2*TimeBasis.from_dict(basis).n_basis else "dosage",source_pvalue=float(row.get("Pvalue",np.nan)),qtl_method=str(qtl_method))


def donor_balanced_weights(groups):
    """Give every participant equal total weight irrespective of visit count."""
    s=pd.Series(np.asarray(groups,str));n=s.map(s.value_counts()).to_numpy(float);w=1.0/n;return w/np.mean(w)


def weighted_lstsq(Z,Y,w):
    Z=np.asarray(Z,float);Y=np.asarray(Y,float);sw=np.sqrt(np.asarray(w,float));Zw=Z*sw[:,None];Yw=Y*sw[:,None] if Y.ndim==2 else Y*sw;return np.linalg.lstsq(Zw,Yw,rcond=None)[0]


def nuisance_design(meta,covariate_cols,age_basis:CubicAxisBasis):
    if "current_age" not in meta.columns:meta=add_current_age(meta)
    B=age_basis.transform(pd.to_numeric(meta["current_age"],errors="coerce").to_numpy(float));C=[]
    for c in covariate_cols:
        if c in meta.columns:C.append(pd.to_numeric(meta[c],errors="coerce").to_numpy(float))
    C=np.column_stack(C) if C else np.empty((len(meta),0),float);return np.column_stack([np.ones(len(meta)),B,C])


def residualize_vector_for_metrics(meta,v,covariate_cols):
    if "current_age" not in meta.columns:meta=add_current_age(meta)
    groups=meta.participant_id.astype(str).to_numpy();w=donor_balanced_weights(groups);ab=CubicAxisBasis.fit(meta.current_age);Z=nuisance_design(meta,covariate_cols,ab);b=weighted_lstsq(Z,np.asarray(v,float),w);return np.asarray(v,float)-Z@b,w


def prediction_metrics(y_resid,pred_resid,weights=None):
    y=np.asarray(y_resid,float);p=np.asarray(pred_resid,float);ok=np.isfinite(y)&np.isfinite(p);y,p=y[ok],p[ok];w=np.ones(len(y),float) if weights is None else np.asarray(weights,float)[ok]
    if len(y)<3:return {}
    base=float(np.sum(w*y**2));r2=1.0-float(np.sum(w*(y-p)**2))/max(base,1e-30);rmse=np.sqrt(float(np.average((y-p)**2,weights=w)))
    return {"R2_frozen":float(r2),"pearson":float(pearsonr(y,p).statistic),"spearman":float(spearmanr(y,p).statistic),"MAE":float(np.average(np.abs(y-p),weights=w)),"RMSE":float(rmse),"n":int(len(y))}


def _score_strength(yres,gres,groups,weights):
    """One-degree frozen-score strength diagnostic; cluster-robust F=t^2."""
    yres=np.asarray(yres,float);gres=np.asarray(gres,float);groups=np.asarray(groups,str);weights=np.asarray(weights,float);ok=np.isfinite(yres)&np.isfinite(gres)&np.isfinite(weights)
    if ok.sum()<5 or np.nanstd(gres[ok])<1e-12:return {"grex_score_beta":np.nan,"grex_score_p":np.nan,"grex_score_F":np.nan}
    try:
        import statsmodels.api as sm
        X=sm.add_constant(gres[ok],has_constant="add");fit=sm.WLS(yres[ok],X,weights=weights[ok]).fit(cov_type="cluster",cov_kwds={"groups":groups[ok]});beta=float(fit.params[1]);p=float(fit.pvalues[1]);t=float(fit.tvalues[1]);return {"grex_score_beta":beta,"grex_score_p":p,"grex_score_F":t*t}
    except Exception:return {"grex_score_beta":np.nan,"grex_score_p":np.nan,"grex_score_F":np.nan}


def evaluate_frozen_grex(meta,y,grex,covariate_cols):
    """Residualized expression support and strength diagnostics for a frozen IV score."""
    if "current_age" not in meta.columns:meta=add_current_age(meta)
    yres,w=residualize_vector_for_metrics(meta,y,covariate_cols);gres,_=residualize_vector_for_metrics(meta,grex,covariate_cols);met=prediction_metrics(yres,gres,w);met.update(_score_strength(yres,gres,meta.participant_id.astype(str),w));return met,yres,gres,w
