from __future__ import annotations
import numpy as np,pandas as pd
from .utils import bh_fdr

def global_multiple_testing(results,pcol="pvalue",group_cols=None):
    z=results.copy();m=int(np.isfinite(z[pcol]).sum())
    z["FDR_global"]=bh_fdr(z[pcol]);z["P_bonf_global"]=np.minimum(1,z[pcol].astype(float)*m);z["n_tests_global"]=m
    if group_cols:
        z["FDR_within_group"]=z.groupby(group_cols)[pcol].transform(lambda x:bh_fdr(x.values))
    return z

def participant_bootstrap(data,fit_function,id_col="participant_id",n_boot=1000,seed=1):
    rng=np.random.default_rng(seed);ids=np.array(pd.unique(data[id_col].astype(str)));rows=[]
    for b in range(n_boot):
        sam=rng.choice(ids,len(ids),replace=True);parts=[]
        for j,p in enumerate(sam):
            x=data[data[id_col].astype(str)==p].copy();x[id_col]=f"{p}__{j}";parts.append(x)
        try:
            r=fit_function(pd.concat(parts,ignore_index=True));r["bootstrap"]=b;rows.append(r)
        except Exception:pass
    z=pd.DataFrame(rows)
    if not len(z):return z,{}
    med=np.sign(z.beta.median())
    return z,{"same_direction_fraction":float(np.mean(np.sign(z.beta)==med)),
              "beta_ci_low":float(z.beta.quantile(.025)),"beta_ci_high":float(z.beta.quantile(.975))}
