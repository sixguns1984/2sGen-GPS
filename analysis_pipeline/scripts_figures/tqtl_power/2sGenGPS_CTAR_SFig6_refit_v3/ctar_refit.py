#!/usr/bin/env python3
"""Eight-df CTAR fixed-rho OU generalized least squares and donor-level resampling.

The overall genetic mean is two hard-call genotype channels (G=1, G=2)
multiplied by cubic [1,t,t^2,t^3].  Fixed-rho gives a *conditional*
profile-sigma LR; optional rho-grid profiles rho in both null and full models.
The bootstrap is descriptive resampling stability, NOT bias-corrected true power.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import chi2

DF=8

def cubic(t):
    t=np.asarray(t,float)
    return np.column_stack((np.ones(len(t)),t,t*t,t*t*t))

def design(t, genotype, cov=None):
    t=np.asarray(t,float); genotype=np.asarray(genotype,float)
    if cov is None: cov=np.zeros((len(t),0))
    cov=np.asarray(cov,float)
    z=cubic(t)
    x0=np.column_stack((z,cov))
    # MUST match Stage-I genotype_channels(..., mode="binary"):
    # 0->00, 1->01, 2->10, missing(-1)->11.
    c0=np.isin(genotype,[2.,-1.]).astype(float)
    c1=np.isin(genotype,[1.,-1.]).astype(float)
    xg=np.column_stack((c0[:,None]*z, c1[:,None]*z))
    return x0, np.column_stack((x0,xg))

def whiten_subject(t, x, y, rho):
    """Exact OU innovations, rho is correlation at 1-year separation."""
    if not 0<=rho<1:raise ValueError('rho must be in [0,1)')
    t=np.asarray(t,float)
    if len(t)>1 and np.any(np.diff(t)<=0):
        raise ValueError('Duplicate/non-increasing visit time in a participant')
    a=np.array(x,float,copy=True); b=np.array(y,float,copy=True)
    if len(t)<2 or rho==0: return a,b,0.
    r=rho**np.diff(t)
    s=np.sqrt(1-r*r)
    a[1:] = (x[1:] - r[:,None]*x[:-1])/s[:,None]
    b[1:] = (y[1:] - r*y[:-1])/s
    ld=np.log1p(-r*r).sum()
    return a,b,float(ld)

def summaries_per_donor(time, donor, genotype, y, cov, rho):
    time=np.asarray(time,float); donor=np.asarray(donor).astype(str)
    genotype=np.asarray(genotype,float); y=np.asarray(y,float)
    cov=np.asarray(cov,float)
    order=np.lexsort((time,donor))
    time=time[order]; donor=donor[order]; genotype=genotype[order]
    y=y[order]; cov=cov[order]
    if len(time)==0:raise ValueError('No observations')
    ids, first, counts=np.unique(donor,return_index=True,return_counts=True)
    x0,x=design(time,genotype,cov)
    k=x.shape[1]; n=len(ids)
    xx=np.empty((n,k,k)); xy=np.empty((n,k)); yy=np.empty(n); ld=np.empty(n)
    for i,(st,ct) in enumerate(zip(first,counts)):
        sl=slice(st,st+ct)
        a,b,ld[i]=whiten_subject(time[sl],x[sl],y[sl],rho)
        xx[i]=a.T@a; xy[i]=a.T@b; yy[i]=b@b
    return {'xx':xx,'xy':xy,'yy':yy,'logdet':ld,'obs':counts,'donor':ids,'p0':x0.shape[1]}

def _rss(a,b,c):
    # least squares with explicit numerical rank guard; no silent ridge approximation
    # matrices are cross-products, symmetric by construction
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):return float('nan')
    # Production Stage-I GPU uses pseudoinverses for nuisance GLS.
    # Do not reject based on cross-product matrix rank alone; numerical
    # conditioning of raw time polynomials can spuriously lower that rank.
    try:
        beta=np.linalg.pinv(a,hermitian=True)@b
    except np.linalg.LinAlgError:
        return float('nan')
    residual=float(c-np.dot(b,beta))
    if not np.isfinite(residual) or residual<=1e-10:return float('nan')
    return residual

def fit_from_sums(x, count=None):
    if count is None:count=np.ones(len(x['donor']),dtype=float)
    count=np.asarray(count,float)
    nobs=float(count@x['obs'])
    xtx=np.einsum('n,nij->ij',count,x['xx'],optimize=True)
    xty=count@x['xy']; yty=float(count@x['yy'])
    p0=x['p0']; k=len(xty)
    if nobs<=k+5: return {'lrt':np.nan,'p':np.nan,'nobs':nobs}
    rss0=_rss(xtx[:p0,:p0],xty[:p0],yty)
    rss1=_rss(xtx,xty,yty)
    if not np.isfinite(rss0) or not np.isfinite(rss1):
        return {'lrt':np.nan,'p':np.nan,'nobs':nobs}
    lrt=max(0.,nobs*np.log(rss0/rss1))
    ld=float(count@x['logdet'])
    ll0=-.5*(nobs*np.log(rss0/nobs)+ld)
    ll1=-.5*(nobs*np.log(rss1/nobs)+ld)
    return {'lrt':lrt,'p':float(chi2.sf(lrt,DF)), 'll0':ll0,'ll1':ll1,
            'rss0':rss0,'rss1':rss1,'nobs':nobs}

def fit_profile(grid_sums,count=None):
    # Each rho has separate ML fit for H0 and H1. Correlation grid is a
    # user-selectable approximation to the production 100-point search.
    results=[fit_from_sums(s,count) for s in grid_sums]
    h0=max((z['ll0'] for z in results if np.isfinite(z.get('ll0',np.nan))),default=np.nan)
    h1=max((z['ll1'] for z in results if np.isfinite(z.get('ll1',np.nan))),default=np.nan)
    lr=max(0.,2*(h1-h0)) if np.isfinite(h0) and np.isfinite(h1) else np.nan
    return {'lrt':lr,'p':float(chi2.sf(lr,DF)) if np.isfinite(lr) else np.nan}

def donor_bootstrap(grid_sums, reps, seed, fraction=1., replace=True, alpha=.05, save_lrt=False):
    rng=np.random.default_rng(seed)
    n=len(grid_sums[0]['donor']); m=max(16,int(round(n*fraction)))
    lrts=np.empty(reps); ps=np.empty(reps)
    if replace and fraction>1:raise ValueError('fraction cannot exceed 1')
    for i in range(reps):
        # Actual participant indices, not independent RNA observations
        draw=rng.choice(n,size=m,replace=replace)
        count=np.bincount(draw,minlength=n)
        res=fit_from_sums(grid_sums[0],count) if len(grid_sums)==1 else fit_profile(grid_sums,count)
        lrts[i]=res['lrt'];ps[i]=res['p']
    good=np.isfinite(ps)
    k=int(np.sum(ps[good]<alpha)); N=int(good.sum())
    return {'bootstrap_detection':k/N if N else np.nan,'valid':N,'total':int(reps),
            'success':k,'lrt':lrts if save_lrt else None,'p':ps if save_lrt else None}

def exact_rho_grid(center=None, points=15):
    if center is None or not np.isfinite(center): return np.linspace(0.,.99,points)
    grid=np.linspace(0.,.99,points)
    return np.unique(np.r_[grid,np.clip(center,0,.99)])
