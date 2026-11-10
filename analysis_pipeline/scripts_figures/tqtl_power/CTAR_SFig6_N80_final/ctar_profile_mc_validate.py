#!/usr/bin/env python3
"""Optional high-fidelity CTAR profile-likelihood Monte Carlo calibration.

Faithfully implements cubic common trajectory, two hard-call genotype channels,
8-df overall genetic LRT, 6-df genotype-by-time LRT, and OU rho grid profile.
Random nuisance covariates are included but are not identical to production covariates.
This is an additional validation layer, not a replacement for production GPU scanner.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2
from ctar_sfig6_n80 import (meta_templates,ordered_templates,mode_values,find_min,ou_grams,
                             compute_power)

def build_cohort(templates,n,pve,maf,rho,pattern,horizon,rng,nuisance=18):
    # Participant, not visit, is the resampling unit.  Repeats retain visit spacing.
    draws=rng.integers(0,len(templates),size=n)
    tlist=[templates[i] for i in draws]
    genotype=rng.binomial(2,maf,size=n)
    if (np.sum(genotype==2)<5 or np.sum(genotype==1)<5 or np.sum(genotype==0)<5):
        return None
    nvis=np.array([len(v) for v in tlist]);ends=np.cumsum(nvis);starts=np.r_[0,ends[:-1]]
    t=np.concatenate(tlist).astype(float);group=np.repeat(np.arange(n),nvis)
    x=t/horizon
    B=np.column_stack((np.ones(len(x)),x,x**2,x**3))
    # Participant-level + visit-specific nuisance proxies, independent of genotype.
    nbase=nuisance//2;nsample=nuisance-nbase
    cv=np.column_stack((rng.normal(size=(n,nbase))[group],rng.normal(size=(len(x),nsample)))) if nuisance else np.empty((len(x),0))
    Z=np.column_stack((B,cv))
    gen=genotype[group]
    # g=0 is the reference: two genotype indicator channels with four time coefficients each.
    G1=(gen==1).astype(float)[:,None]*B
    G2=(gen==2).astype(float)[:,None]*B
    Xfull=np.column_stack((Z,G1,G2))
    Xstatic=np.column_stack((Z,G1[:,0],G2[:,0]))
    prev=np.arange(len(t))-1
    first=np.zeros(len(t),dtype=bool);first[starts]=True
    interior=np.flatnonzero(~first)
    gaps=t[interior]-t[prev[interior]]
    if np.any(gaps<=0):raise ValueError('Nonpositive follow-up visit gaps')
    aa=np.power(rho,gaps) if rho>0 else np.zeros(len(gaps))
    eps=np.empty(len(t))
    for st,en in zip(starts,ends):
        eps[st]=rng.normal()
        for i in range(st+1,en):
            a=rho**(t[i]-t[i-1]) if rho>0 else 0.
            eps[i]=a*eps[i-1]+np.sqrt(1-a*a)*rng.normal()
    f=mode_values(t,pattern,horizon)
    f2_ref=np.mean(mode_values(np.linspace(0,horizon,5),pattern,horizon)**2)
    # signal/(signal+noise) PVE at reference MAF, unit marginal residual variance
    amp=math.sqrt((pve/(1-pve))/(2*.2*(1-.2)*f2_ref))
    y=amp*(gen-2*maf)*f + eps
    return Xfull,Xstatic,Z,y,prev,interior,gaps

def fixed_rho_logliks(cohort,rhos):
    Xfull,Xstatic,Z,y,prev,interior,gaps=cohort
    n=len(y)
    best=np.array([-np.inf]*3)
    for rho in rhos:
        a=np.power(rho,gaps) if rho>0 else np.zeros(len(gaps))
        std=np.sqrt(np.maximum(1-a*a,1e-12))
        # Whiten exactly using innovations: e_i - rho^(dt) e_prev.
        wX=Xfull.copy();wy=y.copy()
        wX[interior]=(Xfull[interior]-a[:,None]*Xfull[prev[interior]])/std[:,None]
        wy[interior]=(y[interior]-a*y[prev[interior]])/std
        yy=float(wy@wy)
        logdet=float(np.log1p(-a*a).sum())
        k_null=Z.shape[1];k_full=Xfull.shape[1]
        XtX=wX.T@wX;Xty=wX.T@wy
        # Static genetic contrast has the TWO channel intercepts only.
        # Crucial: it is NOT the first k_static columns of Xfull.
        ind0=np.arange(k_null)
        inds=np.r_[ind0,k_null,k_null+4]
        indf=np.arange(k_full)
        for j,cols in enumerate((ind0,inds,indf)):
            M=XtX[np.ix_(cols,cols)];rhs=Xty[cols]
            try:sol=np.linalg.solve(M,rhs)
            except np.linalg.LinAlgError:continue
            rss=max(yy-float(rhs@sol),1e-12)
            ll=-.5*(n*np.log(rss/n)+logdet)
            if ll>best[j]:best[j]=ll
    if not np.isfinite(best).all():return np.nan,np.nan
    p8=chi2.sf(max(0.,2*(best[2]-best[0])),8)
    p6=chi2.sf(max(0.,2*(best[2]-best[1])),6)
    return p8,p6

def main():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description=__doc__)
    p.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    p.add_argument('--metadata',default=None)
    p.add_argument('--loci-file',required=False,default=None,help='loci_1500_real_or_demo.csv from main planning run')
    p.add_argument('--output',default=None)
    p.add_argument('--n',type=int,default=1000,help='N per cohort, typically N80/80 from planning')
    p.add_argument('--n-loci',type=int,default=120)
    p.add_argument('--replications',type=int,default=50)
    p.add_argument('--rho',type=float,default=.5)
    p.add_argument('--rho-points',type=int,default=21,help='Use 100 to match production rho grid')
    p.add_argument('--pve',type=float,default=.01)
    p.add_argument('--pattern',choices=['linear','early','late','static'],default='linear')
    p.add_argument('--horizon',type=float,default=4.)
    p.add_argument('--nuisance',type=int,default=18)
    p.add_argument('--alpha-disc',type=float,required=True)
    p.add_argument('--alpha-inter',type=float,required=True)
    p.add_argument('--seed',type=int,default=20261030)
    p.add_argument('--demo',action='store_true')
    args=p.parse_args()
    out=Path(args.output or (Path(args.root)/'manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_N80_final')).resolve()
    out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    if args.demo:
        groups=ordered_templates(n=100,seed=args.seed)
        loci=pd.DataFrame({'maf':rng.uniform(.12,.43,size=max(args.n_loci*2,200))})
    else:
        groups=meta_templates(args.metadata or Path(args.root)/'data/metadata/amp_pd_samples.parquet')
        if not args.loci_file:
            args.loci_file=out/'loci_1500_real_or_demo.csv'
        loci=pd.read_csv(args.loci_file)
    if len(loci)<args.n_loci:raise ValueError(f'Require >= {args.n_loci} available loci')
    loci=loci.sample(n=args.n_loci,random_state=args.seed).reset_index(drop=True)
    rhos=np.linspace(0,.99,args.rho_points)
    rows=[]
    for idx,row in loci.iterrows():
        maf=float(row.maf)
        accepted=0;temporal=dynamic=0
        for j in range(args.replications):
            c1=build_cohort(groups['PPMI'],args.n,args.pve,maf,args.rho,args.pattern,args.horizon,rng,args.nuisance)
            c2=build_cohort(groups['PDBP'],args.n,args.pve,maf,args.rho,args.pattern,args.horizon,rng,args.nuisance)
            if c1 is None or c2 is None:continue
            d8,d6=fixed_rho_logliks(c1,rhos)
            r8,r6=fixed_rho_logliks(c2,rhos)
            if not np.isfinite([d8,d6,r8,r6]).all():continue
            accepted+=1
            temporal+=int(d8<args.alpha_disc and r8<.05)
            dynamic+=int(d8<args.alpha_disc and d6<args.alpha_inter and r8<.05 and r6<.05)
        rows.append({'locus':idx,'maf':maf,'accepted':accepted,'attempted':args.replications,
                     'power_temporal':temporal/accepted if accepted else np.nan,
                     'power_dynamic':dynamic/accepted if accepted else np.nan})
        if (idx+1)%10==0:
            pd.DataFrame(rows).to_csv(out/'profile_mc_validation_checkpoint.csv',index=False)
            print(f'[profile MC] {idx+1}/{args.n_loci} loci',flush=True)
    tab=pd.DataFrame(rows)
    tab.to_csv(out/'profile_mc_validation_locuswise.csv',index=False)
    valid=tab[tab.accepted>=max(1,int(args.replications*.8))]
    summary={'n_loci_requested':args.n_loci,'n_loci_valid':len(valid),'n_per_cohort':args.n,'pve':args.pve,
             'rho_points':args.rho_points,'replications_per_locus':args.replications,'demo':args.demo,
             'mean_temporal_power':float(valid.power_temporal.mean()) if len(valid) else None,
             'fraction_loci_temporal_power_ge80':float((valid.power_temporal>=.8).mean()) if len(valid) else None,
             'fraction_loci_dynamic_power_ge80':float((valid.power_dynamic>=.8).mean()) if len(valid) else None}
    import json
    (out/'profile_mc_validation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('PROFILE-LRT VALIDATION:',summary)

if __name__=='__main__':main()
