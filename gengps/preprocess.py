from __future__ import annotations
import numpy as np
import pandas as pd
from .utils import require_aligned

def complete_visit_participants(cov, visit_col="visit", id_col="participant_id",
                                required_visits=None):
    """Participants with one and only one row at every required visit."""
    if required_visits is None:
        required_visits = sorted(pd.unique(cov[visit_col].dropna()))
    required_visits = list(required_visits)
    good=[]
    for pid,g in cov.groupby(id_col, sort=False):
        counts=g[visit_col].value_counts()
        if all(counts.get(v,0)==1 for v in required_visits) and len(g[g[visit_col].isin(required_visits)])==len(required_visits):
            good.append(str(pid))
    return good, required_visits

def filter_complete_grid(cov, expr, genotype, visit_col="visit", id_col="participant_id",
                         required_visits=None):
    """Method-specific filter for the exact submitted fixed-grid implementation."""
    require_aligned(cov,expr)
    good, visits = complete_visit_participants(cov,visit_col,id_col,required_visits)
    keep=cov[id_col].astype(str).isin(good) & cov[visit_col].isin(visits)
    c=cov.loc[keep].copy(); e=expr.loc[keep].copy()
    order=pd.Categorical(c[visit_col],categories=visits,ordered=True)
    c=c.assign(_visit_order=order).sort_values([id_col,"_visit_order"]).drop(columns="_visit_order")
    e=e.loc[c.index]
    participant_order = list(pd.unique(c[id_col].astype(str)))
    g=genotype.copy();g.index=g.index.astype(str);g=g.reindex(participant_order)
    return c,e,g,visits

def visit_grid(cov, visit_col="visit", id_col="participant_id"):
    """Return participant x visit observation-row lookup; missing is -1."""
    visits=list(pd.unique(cov[visit_col].dropna()))
    try: visits=sorted(visits)
    except Exception: pass
    pids=list(pd.unique(cov[id_col].astype(str)))
    pmap={p:i for i,p in enumerate(pids)};vmap={v:i for i,v in enumerate(visits)}
    idx=np.full((len(pids),len(visits)),-1,int)
    for row,(pid,v) in enumerate(zip(cov[id_col].astype(str),cov[visit_col])):
        if v in vmap: idx[pmap[pid],vmap[v]]=row
    return pids,visits,idx

def missing_patterns(cov, visit_col="visit", id_col="participant_id"):
    pids,visits,idx=visit_grid(cov,visit_col,id_col)
    patterns={}
    for i,p in enumerate(pids):
        mask=tuple((idx[i]>=0).tolist())
        patterns.setdefault(mask,[]).append(i)
    return pids,visits,idx,patterns
