#!/usr/bin/env python3
"""Exact fixed-rho donor-bootstrap CTAR LRT, accelerated linear algebra.

Statistical target equals ctar_refit.donor_bootstrap: donor multiplicities weight
original whitened GLS cross-products; nuisance and 8-df genetic coefficients
are re-estimated in EACH resample. No chi-square noncentral approximation.
The null/full rho-profiled original-fit audit remains in ctar_refit.py.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import chi2
from ctar_refit import fit_from_sums,fit_profile,donor_bootstrap

DF=8

def _solve_rss(a,b,c):
    if not np.isfinite(c):return np.nan
    try:
        # Cholesky on original normal equations = exact GLS least squares,
        # not regularized. Falling back to Stage-I-compatible pseudoinverse
        # for rank-deficient/ill-conditioned participant draws.
        cf=cho_factor(a,lower=True,check_finite=False,overwrite_a=False)
        diag=np.abs(np.diag(cf[0]))
        if diag.min()/diag.max()<1e-5:
            raise np.linalg.LinAlgError('Near rank deficiency; fallback to Stage-I pseudoinverse')
        x=cho_solve(cf,b,check_finite=False)
        rss=float(c-b@x)
        if np.isfinite(rss) and rss>1.e-10: return rss
    except (np.linalg.LinAlgError,ValueError): pass
    try:
        x=np.linalg.pinv(a,hermitian=True)@b
        rss=float(c-b@x)
        return rss if np.isfinite(rss) and rss>1.e-10 else np.nan
    except (np.linalg.LinAlgError,ValueError):return np.nan

def fit_weighted_arrays(sums,weights,chunk_size=32):
    """Compute LRT for B donor-weighted GLS fits, using BLAS3 donor contractions.

    weights: (B,n_donors), integer counts. The same stats as baseline code;
    use fp64 and do NOT change the overall chi-square 8 df test.
    """
    s=sums;weights=np.asarray(weights,dtype=np.float64)
    if weights.ndim!=2 or weights.shape[1]!=len(s['donor']):
        raise ValueError('weights must have [replicates, donors] shape')
    nrep=len(weights);p0=int(s['p0']);k=s['xy'].shape[1]
    out=np.full(nrep,np.nan,float)
    xx=np.ascontiguousarray(s['xx'].reshape(len(s['donor']),-1),dtype=np.float64)
    xy=np.ascontiguousarray(s['xy'],dtype=np.float64)
    yy=np.asarray(s['yy'],dtype=np.float64)
    obs=np.asarray(s['obs'],dtype=np.float64)
    for st in range(0,nrep,chunk_size):
        en=min(nrep,st+chunk_size);w=np.ascontiguousarray(weights[st:en])
        # Coefficients re-estimated separately for every donor resample.
        gram=(w @ xx).reshape(en-st,k,k)
        cross=w @ xy
        norm=w @ yy
        nobs=w @ obs
        for j in range(en-st):
            if nobs[j]<=k+5:continue
            a=gram[j];b=cross[j];c=float(norm[j])
            rss0=_solve_rss(a[:p0,:p0],b[:p0],c)
            rss1=_solve_rss(a,b,c)
            if np.isfinite(rss0) and np.isfinite(rss1):
                out[st+j]=max(0.,float(nobs[j]*np.log(rss0/rss1)))
    return out

def donor_bootstrap_fast(sums,reps,seed,fraction=1.,replace=True,alpha=.05,
                         save_lrt=False,batch_size=32):
    """Draw same donors as original one-by-one routine; contract per batch."""
    if len(sums)!=1:raise ValueError('Fast backend requires fixed rho; profile-rho uses legacy exact implementation')
    s=sums[0];rng=np.random.default_rng(seed)
    n=len(s['donor']);m=max(16,int(round(n*fraction)))
    if not replace and m>n:raise ValueError('Subsample larger than cohort')
    if replace and fraction>1:raise ValueError('fraction must be <=1')
    lrts=np.empty(reps,dtype=float)
    for st in range(0,reps,batch_size):
        end=min(reps,st+batch_size)
        counts=np.empty((end-st,n),dtype=float)
        for j in range(end-st):
            draw=rng.choice(n,size=m,replace=replace)
            counts[j]=np.bincount(draw,minlength=n)
        lrts[st:end]=fit_weighted_arrays(s,counts,chunk_size=batch_size)
    # A non-identifiable cubic basis can make BLAS summation orders disagree
    # with the Stage-I pseudoinverse even when Cholesky nominally succeeds.
    # Verify reproducibility on the SAME first draws; fail closed to legacy.
    rng_check=np.random.default_rng(seed)
    for j in range(min(6,reps)):
        dr=rng_check.choice(n,size=m,replace=replace)
        check=fit_from_sums(s,np.bincount(dr,minlength=n))['lrt']
        now=lrts[j]
        if not (np.isfinite(now) and np.isfinite(check) and
                abs(now-check)<=max(1.e-5,1.e-7*abs(check))):
            import warnings
            warnings.warn('Ill-conditioned donor GLS; falling back to pseudoinverse bootstrap for this locus',RuntimeWarning)
            return donor_bootstrap(sums,reps,seed,fraction=fraction,replace=replace,alpha=alpha,save_lrt=save_lrt)
    ps=chi2.sf(lrts,DF)
    good=np.isfinite(ps);succ=int(np.sum(ps[good]<alpha));v=int(good.sum())
    return {'bootstrap_detection':succ/v if v else np.nan,'valid':v,'total':int(reps),
            'success':succ,'lrt':lrts if save_lrt else None,'p':ps if save_lrt else None}
