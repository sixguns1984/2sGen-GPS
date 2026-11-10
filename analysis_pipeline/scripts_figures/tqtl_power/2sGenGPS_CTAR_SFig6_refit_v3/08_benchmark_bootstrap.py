#!/usr/bin/env python3
"""Self-contained realistic-dimension benchmark, no PPMI data required.

This is a speed and numerical check, not a claim of genome-wide pipeline throughput.
"""
import argparse, json, time
import numpy as np
from ctar_refit import summaries_per_donor, donor_bootstrap
from ctar_refit_fast import donor_bootstrap_fast

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--donors',type=int,default=1405)
    p.add_argument('--covariates',type=int,default=51)
    p.add_argument('--reps',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=32)
    x=p.parse_args();r=np.random.default_rng(91)
    n=x.donors;v=3
    d=np.repeat(np.arange(n).astype(str),v)
    t=np.tile([0,1.3,2.8],n)+r.normal(0,.13,n*v)
    t[::v]=0
    g=np.repeat(r.choice([0,1,2],n,p=[.5,.4,.1]),v)
    c=r.normal(size=(n*v,x.covariates))
    y=r.normal(size=n*v)+.25*g+.1*t
    s=summaries_per_donor(t,d,g,y,c,.5)
    times={};results={}
    for name,func in [('legacy',donor_bootstrap),('fast',donor_bootstrap_fast)]:
        t0=time.perf_counter()
        if name=='legacy':z=func([s],x.reps,11,save_lrt=True)
        else:z=func([s],x.reps,11,save_lrt=True,batch_size=x.batch_size)
        times[name]=time.perf_counter()-t0;results[name]=z
    err=float(np.nanmax(np.abs(results['legacy']['lrt']-results['fast']['lrt'])))
    report={'donors':n,'covariates':x.covariates,'reps':x.reps,
            'legacy_seconds':round(times['legacy'],3),'fast_seconds':round(times['fast'],3),
            'speedup':round(times['legacy']/max(times['fast'],1e-9),2),'max_lrt_difference':err,
            'same_successes':results['legacy']['success']==results['fast']['success'],
            'note':'Only per-locus bootstrap kernel; I/O, rho profiling, and parallel load are not included'}
    print(json.dumps(report,indent=2))
    if err>1e-5 or not report['same_successes']:raise SystemExit('Numerical mismatch')
if __name__=='__main__':main()
