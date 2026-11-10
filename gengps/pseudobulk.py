from __future__ import annotations
import numpy as np,pandas as pd,statsmodels.api as sm
from .utils import bh_fdr

def make_pseudobulk(counts,meta,donor_col="donor_id",region_col="region",celltype_col="celltype",min_cells=20):
    if not counts.index.equals(meta.index):raise ValueError("counts/meta cell index mismatch")
    rows=[];info=[]
    for key,idx in meta.groupby([donor_col,region_col,celltype_col]).groups.items():
        if len(idx)<min_cells:continue
        sid="|".join(map(str,key));rows.append(counts.loc[idx].sum().rename(sid))
        d={donor_col:key[0],region_col:key[1],celltype_col:key[2],"n_cells":len(idx),"sample_id":sid};info.append(d)
    return pd.DataFrame(rows),pd.DataFrame(info).set_index("sample_id")

def logcpm(counts,prior=.5):
    lib=counts.sum(axis=1).replace(0,np.nan)
    return np.log2(counts.add(prior).div(lib,axis=0)*1e6)

def genotype_de(logcpm_matrix,sample_meta,donor_cov,genotype,snp,region,celltype,
                covariate_cols=None,donor_col="donor_id",region_col="region",celltype_col="celltype"):
    covariate_cols=covariate_cols or []
    m=sample_meta[(sample_meta[region_col]==region)&(sample_meta[celltype_col]==celltype)]
    Y=logcpm_matrix.loc[m.index];donors=m[donor_col].astype(str).values
    dc=donor_cov.copy();dc.index=dc.index.astype(str);G=genotype.copy();G.index=G.index.astype(str)
    C=dc.reindex(donors)[covariate_cols].to_numpy(float) if covariate_cols else np.empty((len(m),0))
    g=G.reindex(donors)[snp].to_numpy(float);X=np.column_stack([np.ones(len(m)),g,C]);rows=[]
    for gene in Y:
        y=Y[gene].to_numpy(float);ok=np.isfinite(y)&np.all(np.isfinite(X),axis=1)
        if ok.sum()<=X.shape[1]+2:continue
        fit=sm.OLS(y[ok],X[ok]).fit()
        rows.append({"gene":gene,"beta_genotype":float(fit.params[1]),"pvalue":float(fit.pvalues[1]),"n_donors":int(ok.sum())})
    z=pd.DataFrame(rows)
    if len(z):z["FDR"]=bh_fdr(z.pvalue)
    return z
