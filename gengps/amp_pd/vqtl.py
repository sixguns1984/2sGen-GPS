from __future__ import annotations
import numpy as np
from scipy.stats import chi2


def matrix_rank(a, tol=1e-10):
    return int(np.linalg.matrix_rank(np.asarray(a, float), tol=tol))


def residualize(a, w, w_pinv=None):
    a=np.asarray(a,float);w=np.asarray(w,float)
    w_pinv=np.linalg.pinv(w,rcond=1e-10) if w_pinv is None else np.asarray(w_pinv,float)
    return a-w@(w_pinv@a)


def cluster_wald(y, x, clusters, test_idx, nuisance_rank=0):
    """Backward-compatible cluster-robust Wald helper.

    Retained for the Stage-I validation tests and for compatibility with the first
    vQTL sensitivity patch. The revised cross-fitted fluctuation-QTL analysis does
    not use this helper for its primary test; participant-level HC3 inference is used
    there instead.
    """
    y=np.asarray(y,float).reshape(-1);x=np.asarray(x,float);clusters=np.asarray(clusters).astype(str);test_idx=np.asarray(test_idx,int)
    ok=np.isfinite(y)&np.isfinite(x).all(axis=1);y,x,clusters=y[ok],x[ok],clusters[ok]
    if len(y)<=x.shape[1]+3 or len(np.unique(clusters))<10 or x.shape[1]==0:return np.nan,np.nan,np.full(x.shape[1],np.nan),np.nan
    xtx_inv=np.linalg.pinv(x.T@x,rcond=1e-10);beta=xtx_inv@(x.T@y);resid=y-x@beta
    ug,invg=np.unique(clusters,return_inverse=True);score=np.zeros((len(ug),x.shape[1]),float);np.add.at(score,invg,x*resid[:,None]);meat=score.T@score
    gnum=len(ug);n=len(y);k=max(1,int(nuisance_rank)+matrix_rank(x));scale=(gnum/max(gnum-1,1))*((n-1)/max(n-k,1));cov=scale*(xtx_inv@meat@xtx_inv)
    b=beta[test_idx];v=cov[np.ix_(test_idx,test_idx)];df=matrix_rank(v)
    if df<1 or not np.isfinite(v).all():return np.nan,np.nan,beta,np.nan
    stat=float(b.T@np.linalg.pinv(v,rcond=1e-10)@b);p=float(chi2.sf(max(stat,0.0),df))
    return stat,p,beta,float(df)


def hc3_wald(y, x, test_idx, nuisance_rank=0):
    y=np.asarray(y,float).reshape(-1);x=np.asarray(x,float);test_idx=np.asarray(test_idx,int)
    ok=np.isfinite(y)&np.isfinite(x).all(axis=1);y,x=y[ok],x[ok]
    if len(y)<=x.shape[1]+5 or x.shape[1]==0:return np.nan,np.nan,np.full(x.shape[1],np.nan),np.nan
    xtx_inv=np.linalg.pinv(x.T@x,rcond=1e-10);beta=xtx_inv@(x.T@y);resid=y-x@beta
    h=np.einsum('ij,jk,ik->i',x,xtx_inv,x);adj=resid/np.maximum(1.0-h,1e-6)
    meat=x.T@((adj*adj)[:,None]*x);cov=xtx_inv@meat@xtx_inv
    b=beta[test_idx];v=cov[np.ix_(test_idx,test_idx)];df=matrix_rank(v)
    if df<1 or not np.isfinite(v).all():return np.nan,np.nan,beta,np.nan
    stat=float(b.T@np.linalg.pinv(v,rcond=1e-10)@b);p=float(chi2.sf(max(stat,0.0),df))
    return stat,p,beta,float(df)


def make_balanced_fold_map(participant_ids,n_folds=5,seed=1):
    ids=np.asarray(list(map(str,participant_ids)),dtype=object)
    rng=np.random.default_rng(int(seed));perm=ids[rng.permutation(len(ids))]
    return {str(pid):int(i%int(n_folds)) for i,pid in enumerate(perm)}


def participant_fluctuation(raw_resid,obs_to_part,n_part,min_visits=3):
    r=np.asarray(raw_resid,float);grp=np.asarray(obs_to_part,int);finite=np.isfinite(r)
    nvis=np.bincount(grp[finite],minlength=int(n_part)).astype(int)
    s1=np.bincount(grp[finite],weights=r[finite],minlength=int(n_part));s2=np.bincount(grp[finite],weights=r[finite]**2,minlength=int(n_part))
    var=np.full(int(n_part),np.nan,float);ok=nvis>=int(min_visits)
    var[ok]=np.maximum((s2[ok]-s1[ok]*s1[ok]/nvis[ok])/np.maximum(nvis[ok]-1,1),0.0)
    mad=np.full(int(n_part),np.nan,float)
    for j in np.flatnonzero(ok):
        rr=r[(grp==j)&finite]
        if len(rr)>=int(min_visits):
            med=np.median(rr);mad[j]=np.median(np.abs(rr-med))
    return var,mad,nvis


def adjusted_scale_test(scale_y,genotype,wp,primary='additive',min_group_n=10):
    y=np.asarray(scale_y,float);g=np.asarray(genotype,float);w=np.asarray(wp,float)
    ok=np.isfinite(y)&np.isfinite(g)&np.isfinite(w).all(axis=1);y,g,w=y[ok],g[ok],w[ok]
    if len(y)<30:return {'stat':np.nan,'p':np.nan,'df':np.nan,'beta':np.nan,'n':int(len(y)),'min_group_n':np.nan}
    hard_ok=np.isclose(g,np.rint(g),atol=1e-8,rtol=0)&(g>=0)&(g<=2)
    if not hard_ok.all():raise ValueError(f'variance test received non-hard-call genotype values: {np.unique(g[~hard_ok])[:10].tolist()}')
    if primary=='additive':
        h=g[:,None];x=np.column_stack([w,h]);idx=np.array([x.shape[1]-1]);stat,p,beta,df=hc3_wald(y,x,idx);b=float(beta[idx[0]]) if len(beta)>idx[0] else np.nan
        return {'stat':stat,'p':p,'df':df,'beta':b,'n':int(len(y)),'min_group_n':np.nan}
    hard=np.rint(g).astype(int);counts=np.array([(hard==k).sum() for k in [0,1,2]],int);ming=int(counts.min())
    if ming<int(min_group_n):return {'stat':np.nan,'p':np.nan,'df':np.nan,'beta':np.nan,'n':int(len(y)),'min_group_n':ming}
    h=np.column_stack([(hard==1).astype(float),(hard==2).astype(float)]);x=np.column_stack([w,h]);idx=np.arange(x.shape[1]-2,x.shape[1]);stat,p,beta,df=hc3_wald(y,x,idx)
    return {'stat':stat,'p':p,'df':df,'beta':np.nan,'n':int(len(y)),'min_group_n':ming}
