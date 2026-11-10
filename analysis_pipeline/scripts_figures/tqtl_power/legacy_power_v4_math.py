#!/usr/bin/env python3
"""Fast planning power from REAL frozen 8-parameter CTAR curves and nuisance design.

Noncentral chi-square fixed-rho approximation (not exact profiled CTAR likelihood).
The expected GLS noncentrality subtracts the projection on the actual Stage-I
nuisance design [common cubic followup-time curve, demographic/genomic PCs,
expression PCs, LM22 covariates].  Two GENOTYPE DUMMY trajectories (8 df) are
preserved; no additive-per-allele assumption.

IMPORTANT: The 8-df chi-square calibration may be unreliable at low MAF and
absent rare homozygotes; output explicit category probabilities and require
(optional) full CTAR null/alt simulation before publishing N80 claims.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import chi2,ncx2,binom

def load_profile(root, code, cohort):
    sys.path.insert(0,str(code))
    from gengps.amp_pd.config import AmpPdPaths
    from gengps.amp_pd.real_qtl import (load_cohort_metadata,_augment_fixed_genotype_pcs,
       _prepare_irregular_followup_rows,method_covariates,_drop_missing_covariates,bfile_for_cohort)
    paths=AmpPdPaths(root=str(root))
    meta=load_cohort_metadata(paths,cohort)
    expr_ids=set(pd.read_parquet(root/'data/expression/amp_pd_expression_z.parquet',columns=['sample_id']).sample_id.astype(str))
    meta=meta[meta.sample_id.astype(str).isin(expr_ids)].copy()
    meta=_augment_fixed_genotype_pcs(paths,cohort,meta,n_pcs=8)
    meta,_,_= _prepare_irregular_followup_rows(meta,pd.DataFrame(index=meta.index))
    cc=method_covariates('submitted_followup_ctar_mptga',cohort,meta,cell_adjust=True)
    meta,_= _drop_missing_covariates(meta,pd.DataFrame(index=meta.index),cc)
    fam=pd.read_csv(str(bfile_for_cohort(paths,cohort))+'.fam',sep=r'\s+',header=None,usecols=[0,1],dtype=str)
    pids=set(meta.participant_id.astype(str));col=0 if fam.iloc[:,0].isin(pids).sum()>=fam.iloc[:,1].isin(pids).sum() else 1
    meta=meta[meta.participant_id.astype(str).isin(set(fam.iloc[:,col].astype(str)))].copy()
    if meta.participant_id.nunique()<100:raise RuntimeError(f'{cohort}: unexpectedly small CTAR cohort')
    return meta,cc

def info_per_donor(meta,covariates,rho,reference_interval=1.):
    """Compute whitened GLS cross-product expectations, incorporating real covars."""
    # All coefficient curves from production use raw t, not standardized time.
    macc=dacc=kacc=None;count=0;obs=0
    for _,z in meta.groupby('participant_id',sort=False):
        t=z.followup_time.to_numpy(float)
        if len(t)>1 and not np.all(np.diff(t)>0):raise ValueError('Follow-up times must be unique/increasing')
        B=np.column_stack([np.ones(len(t)),t,t*t,t*t*t])
        C=z[covariates].to_numpy(float)
        X=np.column_stack([B,C]);Xw=X.copy()
        for j in range(1,len(t)):
            r=rho**((t[j]-t[j-1])/reference_interval) if rho>0 else 0.0
            Xw[j]=(X[j]-r*X[j-1])/np.sqrt(1-r*r)
        Bw=Xw[:,:4]
        M=Xw.T@Xw;D=Xw.T@Bw;K=Bw.T@Bw
        if macc is None:macc=np.zeros_like(M);dacc=np.zeros_like(D);kacc=np.zeros_like(K)
        macc+=M;dacc+=D;kacc+=K;count+=1;obs+=len(t)
    if count<100:raise RuntimeError('No valid donor-level information matrices')
    M=macc/count;D=dacc/count;K=kacc/count
    # SVD pinv is used to avoid instability from correlated PCs and time basis;
    # this matches the pseudoinverse convention of the CTAR scanner.
    L=D.T@np.linalg.pinv(M,rcond=1e-9)@D
    # Even with perfect covariates, S(b) must be >=0 by Schur complement.
    eig_min=float(np.linalg.eigvalsh((K-L+K.T-L.T)*.5).min())
    if eig_min<-1e-5: raise RuntimeError(f'Nuisance GLS matrices not PSD: min eigenvalue={eig_min}')
    return K,(L+L.T)/2,{'rho':rho,'n_participants':count,'n_observations':obs,
        'n_covariates':len(covariates),'nuisance_rank':int(np.linalg.matrix_rank(M)),
        'eigen_min_residualized_basis':eig_min}

def ncp_from_fractions(n, q1,q2,theta1,theta2,K,L):
    mix=q1*theta1+q2*theta2
    per=float(q1*(theta1@K@theta1)+q2*(theta2@K@theta2)-mix@L@mix)
    return max(0.0,float(n)*per)

def power_over_genotypes(n,maf,theta1,theta2,K,L,alpha,draws,rng):
    counts=rng.multinomial(n,[(1-maf)**2,2*maf*(1-maf),maf*maf],size=draws)
    q1=counts[:,1]/n;q2=counts[:,2]/n
    theta1=np.asarray(theta1,float);theta2=np.asarray(theta2,float)
    r1=float(theta1@K@theta1);r2=float(theta2@K@theta2)
    mix=q1[:,None]*theta1[None,:]+q2[:,None]*theta2[None,:]
    val=n*np.maximum(q1*r1+q2*r2-np.einsum('ni,ij,nj->n',mix,L,mix),0.)
    nclass=(counts>0).sum(axis=1)
    df=np.where(nclass>=3,8,np.where(nclass==2,4,0))
    threshold=chi2.isf(alpha,8)
    power=np.where(df>0,ncx2.sf(threshold,np.maximum(df,1),val),0.)
    return {'power':float(np.mean(power)), 'genotype_MC_se':float(np.std(power,ddof=1)/np.sqrt(draws)),
        'full_8df_genotype_design_probability':float(np.mean(df==8)),
        'prob_minor_homozygote_ge1':float(np.mean(counts[:,2]>=1)),
        'prob_minor_homozygote_ge5':float(np.mean(counts[:,2]>=5)),
        'mean_ncp':float(np.mean(val))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True);ap.add_argument('--code',required=True);ap.add_argument('--figroot',required=True)
    ap.add_argument('--mafs',default='.05,.10,.20,.30')
    ap.add_argument('--n-grid',default='100,150,200,250,300,400,500,700,900,1200,1400,1600,2000,2600,3500,5000,7000')
    ap.add_argument('--rho',type=float,default=.6);ap.add_argument('--reference-interval',type=float,default=1.)
    ap.add_argument('--draws',type=int,default=1500);ap.add_argument('--seed',type=int,default=20261029)
    ap.add_argument('--power-target',type=float,default=.8)
    ap.add_argument('--alpha-discovery',type=float,default=None)
    ap.add_argument('--profile',choices=['PP','PD'],default='PD')
    a=ap.parse_args()
    if not 0<=a.rho<.99 or a.draws<300:raise ValueError('Invalid rho/genotype MC settings')
    root=Path(a.root);out=Path(a.figroot)/'power_analysis_empirical_v4';out.mkdir(parents=True,exist_ok=True)
    source=out/'three_real_CTAR_curve_templates.csv'
    if not source.is_file():raise FileNotFoundError(f'Run 06_calibrate_real_ctar_effects.py first: {source}')
    z=pd.read_csv(source,low_memory=False)
    if len(z)!=3:raise RuntimeError('Expect exactly three empirical template effects')
    mafs=sorted(set(float(x) for x in a.mafs.split(',')));ns=sorted(set(int(x) for x in a.n_grid.split(',')))
    if min(ns)<40 or any(not 0<x<=.5 for x in mafs):raise ValueError('Invalid MAF or N')
    pp,cp=load_profile(root,a.code,'PP');pdmeta,cd=load_profile(root,a.code,'PD')
    prof,cols=(pp,cp) if a.profile=='PP' else (pdmeta,cd)
    K,L,info=info_per_donor(prof,cols,a.rho,a.reference_interval)
    if a.alpha_discovery is not None:
        alpha=float(a.alpha_discovery)
    else:
        sig=root/'qtl/PP/submitted_followup_ctar_mptga/final/significant_tqtl.parquet'
        if not sig.exists():raise FileNotFoundError(sig)
        x=pd.read_parquet(sig,columns=['Pvalue','FDR']);pval=pd.to_numeric(x.Pvalue,errors='coerce');q=pd.to_numeric(x.FDR,errors='coerce')
        alpha=float(pval[(q<.05)&np.isfinite(pval)&(pval>0)].max())
    if not 0<alpha<.05:raise ValueError(f'Invalid discovery cutoff {alpha}')
    print(f'[POWER] fixed observed PPMI discovery raw-P cutoff={alpha:.6g}; 8 df; rho={a.rho}',flush=True)
    rng=np.random.default_rng(a.seed)
    rows=[]
    for label,al in [('discovery',alpha),('replication',.05)]:
        for rr in z.itertuples(index=False):
            d=rr._asdict();sigma=float(d['residual_sd']);theta=np.array([float(d[f'beta_g_{j}']) for j in range(8)])/sigma
            # Stage-I BINARY encoding: channel0 is g==2, channel1 is g==1.
            theta1=theta[4:];theta2=theta[:4]
            for p in mafs:
                for n in ns:
                    res=power_over_genotypes(n,p,theta1,theta2,K,L,al,a.draws,rng)
                    rows.append({'threshold':label,'alpha_fixed':al,'n_participants':n,'maf':p,
                      'template_label':str(d['template_label']),'gene':str(d['gene']),
                      'lead_snp':str(d['lead_snp']),'PPMI_empirical_signal_fraction':float(d['signal_fraction']),
                      'PPMI_empirical_MAF':float(d['MAF_observed']),'PPMI_minor_homozygote_n':int(d['n_hom_alt']),
                      'rho_assumed':a.rho,**res})
        print(f'[POWER] {label} complete',flush=True)
    result=pd.DataFrame(rows);result.to_csv(out/'power_grid_real_8parameter_effect_templates.csv',index=False)
    n80=[]
    for key,g in result.groupby(['threshold','maf','template_label'],sort=False):
        g=g.sort_values('n_participants');above=g[g.power>=a.power_target]
        n80.append({'threshold':key[0],'maf':key[1],'template_label':key[2],
          'N80_first_tested':int(above.n_participants.iloc[0]) if len(above) else np.nan,
          'N80_not_reached':len(above)==0,'maximum_tested_N':max(ns),
          'interpretation':'N80 grid approximation using real fitted 8-parameter CTAR trajectory; NOT validated exact minimum'})
    pd.DataFrame(n80).to_csv(out/'N80_grid_empirical_trajectory_maf.csv',index=False)
    # Probability check theoretical minor-homozygote representation at current N.
    gen=[]
    for p in mafs:
        for name,N in [('PPMI',int(pp.participant_id.nunique())),('PDBP',int(pdmeta.participant_id.nunique()))]:
            gen.append({'cohort':name,'N':N,'MAF':p,'mean_minor_homozygotes':N*p*p,
                'prob_ge1':float(binom.sf(0,N,p*p)),'prob_ge5':float(binom.sf(4,N,p*p))})
    pd.DataFrame(gen).to_csv(out/'genotype_class_representation.csv',index=False)
    (out/'power_manifest.json').write_text(json.dumps({'source_templates':str(source),
        'n_PP_participants':int(pp.participant_id.nunique()),'n_PD_participants':int(pdmeta.participant_id.nunique()),
        'profile_design':a.profile,'nuisance_covariates':cols,'information':info,
        'alpha_discovery_fixed':alpha,'alpha_replication_nominal':.05,'rho_assumed':a.rho,
        'effect_unit':'real fitted eight-parameter genotype-category-specific CTAR curves divided by real PPMI nuisance-adjusted residual SD',
        'NO_additive_assumption':True,'N80_interpretation':'first N in user grid with APPROXIMATE >=80% power (not exact minimum)',
        'df_production':8,'method':'fixed-rho expected GLS noncentral chi-square, nuisance projection with real baseline covariates and time basis',
        'calibration_limitations':'selection-biased empirical signal templates; averages donor design matrices; profile rho re-estimation, small genotype-class rank, residual variance estimation, cross-study differences and BH threshold adaptation not captured. Mandatory full CTAR simulation recommended before definite minimum-N claims.',
        'N_grid':ns,'MAFs':mafs,'random_seed':a.seed},indent=2))
    print('[DONE] real-eight-parameter planning power; cohort N PP=',pp.participant_id.nunique(), ' PD=',pdmeta.participant_id.nunique(),flush=True)

if __name__=='__main__':main()
