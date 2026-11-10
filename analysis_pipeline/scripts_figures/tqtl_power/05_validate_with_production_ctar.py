#!/usr/bin/env python3
"""Optional parametric calibration of fast CTAR planning approximation by exact production CTAR.scan.

Runs only requested few full CTAR fits; not included by default in quick workflow.
Simulated participant profiles use real time points/covariates and new independent
simulation IDs, so two duplicates of a source donor remain independent in the
synthetic experiment. Null residuals are generated with the exact CTAR AR(1)
innovation recursion at irregular gaps. Use >=100 replicates for publication-grade
Monte Carlo calibration; 20-30 is a smoke check only.
"""
from __future__ import annotations
import argparse,sys,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import chi2

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    ap.add_argument('--code',default='')
    ap.add_argument('--figroot',default='')
    ap.add_argument('--n',type=int,default=300)
    ap.add_argument('--maf',type=float,default=.05)
    ap.add_argument('--effect',type=float,default=.75)
    ap.add_argument('--rho',type=float,default=.6)
    ap.add_argument('--shape',choices=['constant','linear','quadratic'],default='constant')
    ap.add_argument('--reps',type=int,default=30)
    ap.add_argument('--rho-points',type=int,default=100)
    ap.add_argument('--seed',type=int,default=205011)
    ap.add_argument('--cohort',choices=['PP','PD'],default='PD')
    args=ap.parse_args()
    if not 0<=args.rho<1 or not 0<args.maf<.5 or args.reps<3:raise ValueError('invalid settings')
    root=Path(args.root);code=Path(args.code) if args.code else root/'2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized'
    figroot=Path(args.figroot) if args.figroot else root/'manuscript_figures'/'tqtl_benchmark_v4'
    out=figroot/'power_analysis_planning_v3';out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(code))
    from gengps.amp_pd.config import AmpPdPaths
    from gengps.amp_pd.real_qtl import load_cohort_metadata,_augment_fixed_genotype_pcs,_prepare_irregular_followup_rows,method_covariates,_drop_missing_covariates
    from gengps.methods.ctar_mptga import ContinuousTimeMPTGA
    from importlib.util import spec_from_file_location,module_from_spec
    # reuse power information formula and same empirical sample universe
    spec=spec_from_file_location('powercalc',Path(__file__).with_name('03_ctar_prospective_power.py'))
    powercalc=module_from_spec(spec);spec.loader.exec_module(powercalc)
    paths=AmpPdPaths(root=str(root));meta=load_cohort_metadata(paths,args.cohort)
    expr_ids=set(pd.read_parquet(root/'data'/'expression'/'amp_pd_expression_z.parquet',columns=['sample_id'])['sample_id'].astype(str))
    meta=meta[meta.sample_id.astype(str).isin(expr_ids)].copy()
    meta=_augment_fixed_genotype_pcs(paths,args.cohort,meta,n_pcs=8)
    meta,_,_= _prepare_irregular_followup_rows(meta,pd.DataFrame(index=meta.index))
    covars=method_covariates('submitted_followup_ctar_mptga',args.cohort,meta,cell_adjust=True)
    meta,_=_drop_missing_covariates(meta,pd.DataFrame(index=meta.index),covars)
    # Genotype participant availability filter same as prospective.
    from gengps.amp_pd.real_qtl import bfile_for_cohort
    fam=pd.read_csv(str(bfile_for_cohort(paths,args.cohort))+'.fam',sep=r'\s+',header=None,usecols=[0,1],dtype=str)
    known=set(meta.participant_id.astype(str))
    col=0 if fam.iloc[:,0].isin(known).sum()>=fam.iloc[:,1].isin(known).sum() else 1
    meta=meta[meta.participant_id.astype(str).isin(set(fam.iloc[:,col]))].copy()
    source=[g.sort_values('followup_time') for _,g in meta.groupby('participant_id',sort=True)]
    if not source:raise RuntimeError('No participants')
    times=[g.followup_time.to_numpy(float) for g in source]
    actual_info,_=powercalc.whitened_signal_information(times,args.rho,shape=args.shape)
    approx=powercalc.prospective_power(args.n,args.maf,args.effect,actual_info,.05,12000,np.random.default_rng(args.seed))
    allt=np.concatenate(times);tmu=float(np.mean(allt));tsd=float(np.std(allt))
    def h(t):
        x=(np.asarray(t)-tmu)/tsd
        if args.shape=='constant':return np.ones(len(x))
        if args.shape=='linear':return 1+.35*x
        return 1+.35*x-.12*x*x
    rms=float(np.sqrt(np.mean(h(allt)**2)))
    scanner=ContinuousTimeMPTGA(basis_kind='cubic',genotype_mode='binary',rho_points=args.rho_points,
                                batch_size=128,device='cpu',dtype='float64',reference_interval=1.,rho_max=.99)
    cis=pd.DataFrame({'geneid':['SIM_GENE'],'snpid':['SIM_SNP']})
    rng=np.random.default_rng(args.seed)
    file=out/f'CTAR_exact_validation_N{args.n}_MAF{args.maf:.3f}_effect{args.effect:.2f}_rho{args.rho:.2f}.csv'
    records=[]
    if file.is_file():
        existing=pd.read_csv(file)
        if not existing.empty:records=existing.to_dict('records')
    for r in range(len(records),args.reps):
        inds=rng.integers(0,len(source),size=args.n)
        g=rng.binomial(2,args.maf,size=args.n)
        frames=[];genos=[];yt=[]
        for i,(idx,gt) in enumerate(zip(inds,g)):
            orig=source[int(idx)]
            sub=orig.copy()
            pid=f'SIM{i:07d}'
            sub['participant_id']=pid
            sub['sample_id']=[f'{pid}_V{j:02d}' for j in range(len(sub))]
            sub.index=sub.sample_id.astype(str)
            tt=sub.followup_time.to_numpy(float)
            e=np.zeros(len(tt));e[0]=rng.normal()
            for j in range(1,len(tt)):
                aa=float(args.rho**((tt[j]-tt[j-1])/1.)) if args.rho>0 else 0.
                e[j]=aa*e[j-1]+np.sqrt(1-aa*aa)*rng.normal()
            yt.extend((args.effect*float(gt)*h(tt)/rms+e).tolist())
            frames.append(sub);genos.append((pid,int(gt)))
        m=pd.concat(frames,axis=0)
        m=m.sort_values(['participant_id','followup_time'],kind='mergesort')
        # Original frames already sorted by participant index; sort y safely by sample index.
        ymap={sid:yval for sid,yval in zip([s for f in frames for s in f.index],yt)}
        expression=pd.DataFrame({'SIM_GENE':[ymap[s] for s in m.index]},index=m.index)
        genotype=pd.DataFrame({'SIM_SNP':[x[1] for x in genos]},index=[x[0] for x in genos])
        fit=scanner.scan(m,expression,genotype,cis,covariate_cols=covars,id_col='participant_id',time_col='followup_time')
        p=float(fit.Pvalue.iloc[0])
        rec={'rep':r,'n':args.n,'maf':args.maf,'effect':args.effect,'rho':args.rho,'p':p,
             'detected':int(p<.05),'n_het':int((g==1).sum()),'n_minor_homo':int((g==2).sum()),
             'analytic_power':approx['power'],'rho_points':args.rho_points}
        records.append(rec)
        pd.DataFrame(records).to_csv(file,index=False)
        print(f'[EXACT CTAR] {r+1}/{args.reps}: P={p:.3g}, detected={rec["detected"]}',flush=True)
    result=pd.DataFrame(records)
    print(f'[SUMMARY] exact CTAR power={result.detected.mean():.3f} (n_reps={len(result)}), '
          f'analytic approximation={approx["power"]:.3f}; file={file}',flush=True)

if __name__=='__main__':main()
