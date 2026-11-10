#!/usr/bin/env python3
"""CTAR-aware prospective tQTL power and N planning (no SNP re-QC or refits).

Model: genuine binary genotype-channel (two 4-coefficient curves), eight-df joint
LRT.  Given a genotype-additive true *mean* effect b*g*h(t), the noncentrality
is approximated by b**2 * N * Var_sample(g) * mean_subject(h^T R^-1 h) / sigma**2.
R uses the original CTAR exponential time covariance at empirical irregular
visit times. Genotypes are Monte Carlo draws from HWE under user-specified MAF.
For missing homozygote/heterozygote class, actual identifiable rank (4 or 0)
is used with the *software's 8-df chi-square cutoff*; never silently pretend
all 8 genetic coefficients are estimable. This is a planning approximation,
NOT a new full CTAR simulation, and it neglects profile-rho estimation error.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2, ncx2, binom

METHOD='submitted_followup_ctar_mptga'

def load_profile(root: Path, code: Path, cohort: str):
    sys.path.insert(0,str(code))
    from gengps.amp_pd.config import AmpPdPaths
    from gengps.amp_pd.real_qtl import (
        load_cohort_metadata, _augment_fixed_genotype_pcs, _prepare_irregular_followup_rows,
        method_covariates, _drop_missing_covariates)
    paths=AmpPdPaths(root=str(root))
    meta=load_cohort_metadata(paths,cohort)
    expr_path=root/'data'/'expression'/'amp_pd_expression_z.parquet'
    if not expr_path.is_file():raise FileNotFoundError(expr_path)
    expr_ids=set(pd.read_parquet(expr_path,columns=['sample_id'])['sample_id'].astype(str))
    meta=meta.loc[meta.sample_id.astype(str).isin(expr_ids)].copy()
    meta=_augment_fixed_genotype_pcs(paths,cohort,meta,n_pcs=8)
    empty=pd.DataFrame(index=meta.index)
    meta,empty,n_dup=_prepare_irregular_followup_rows(meta,empty)
    covars=method_covariates(METHOD,cohort,meta,cell_adjust=True)
    meta,empty=_drop_missing_covariates(meta,empty,covars)
    # Missing genotype participants cannot contribute to CTAR.
    from gengps.amp_pd.real_qtl import bfile_for_cohort
    fam=Path(str(bfile_for_cohort(paths,cohort))+'.fam')
    if fam.is_file():
        fam_cols=pd.read_csv(fam,sep=r'\s+',header=None,usecols=[0,1],dtype=str)
        pids=set(meta.participant_id.astype(str))
        match_fid=int(fam_cols.iloc[:,0].isin(pids).sum())
        match_iid=int(fam_cols.iloc[:,1].isin(pids).sum())
        id_col=0 if match_fid>=match_iid else 1
        valid=set(fam_cols.iloc[:,id_col].astype(str))
        if len(valid & pids)==0:raise RuntimeError(f'{cohort}: no genotype participant FAM ID matches')
        meta=meta.loc[meta.participant_id.astype(str).isin(valid)].copy()
    if not len(meta):raise RuntimeError(f'{cohort}: no observations with complete CTAR inputs')
    by=meta.groupby('participant_id',sort=True)
    profiles=[]
    for pid,z in by:
        t=z.followup_time.to_numpy(float)
        if np.unique(t).size!=len(t):raise RuntimeError(f'{cohort} duplicate participant-time: {pid}')
        profiles.append(t)
    if len(profiles)<30:raise RuntimeError(f'{cohort}: unusually few CTAR participants ({len(profiles)})')
    print(f'[STAGE1 {cohort}] participants={len(profiles)} observations={sum(len(x) for x in profiles)} '
          f'covariates={len(covars)} duplicate_resolved={n_dup}',flush=True)
    return profiles,{'cohort':cohort,'n_participants':len(profiles),'n_observations':sum(len(x) for x in profiles),
                     'median_visits':float(np.median([len(x) for x in profiles])),
                     'n_covariates':len(covars),'duplicate_resolved':int(n_dup),
                     'source':'data/metadata/amp_pd_samples.parquet + expression sample IDs + PCs/LM22 + genotype FAM'}


def whitened_signal_information(profiles, rho_ref, reference_interval=1., shape='constant'):
    """Mean h^T R^-1 h per donor with exact CTAR innovation whitening.

    h has RMS 1 over the empirical *observations*; beta is expressed in residual
    standard-deviation units per allele. 'constant' h(t)=1 is deliberately a
    clearly defined conservative trajectory benchmark, since the full CTAR
    tests all cubic genotype coefficients. 'linear' varies smoothly with time.
    """
    t_all=np.concatenate(profiles)
    tmu=float(np.mean(t_all)); tsd=float(np.std(t_all))
    if tsd<=0: raise ValueError('No variation in follow-up time')
    def hfun(t):
        x=(np.asarray(t)-tmu)/tsd
        if shape=='constant':return np.ones(len(t))
        if shape=='linear':return 1. +0.35*x
        if shape=='quadratic':return 1. +0.35*x-0.12*x*x
        raise ValueError(shape)
    h0=hfun(t_all)
    scale=float(np.sqrt(np.mean(h0*h0)))
    if scale<=0:raise RuntimeError('Signal shape scale zero')
    info=[]
    for t in profiles:
        h=hfun(t)/scale
        if len(t)>1:
            dt=np.diff(t)
            if np.any(dt<=0):raise RuntimeError('Non-increasing visit times')
            a=np.power(rho_ref,dt/reference_interval) if rho_ref>0 else np.zeros(len(dt))
            den=np.sqrt(1-a*a)
            z=np.empty_like(h);z[0]=h[0]
            z[1:]=(h[1:]-a*h[:-1])/den
        else:z=h
        info.append(float(z@z))
    return float(np.mean(info)),{'rho':float(rho_ref),'shape':shape,'mean_info_per_donor':float(np.mean(info)),
                                 'median_info_per_donor':float(np.median(info)),'n_donors':len(profiles),
                                 'mean_visits':float(np.mean([len(t) for t in profiles]))}


def prospective_power(n,maf,effect,information,alpha,nsim,rng, nominal_df=8):
    """Integrate conditional noncentral-chi² power over HWE genotype realizations.

    Finite-sample genotype prevalence and rank are modeled exactly. Actual-rank
    alt distributions use df=4 when only 2 genotype categories remain; the
    threshold is always nominal df=8 as in the CTAR production implementation.
    Covariate-genotype accidental association and rho-estimation uncertainty
    are not modeled; report as approximate rather than definitive power.
    """
    prob=np.array([(1-maf)**2, 2*maf*(1-maf), maf*maf],float)
    counts=rng.multinomial(n,prob,size=nsim)
    # genotype classes 0/1/2, with channel1=(g==2), channel2=(g==1)
    sum_g=counts[:,1]+2*counts[:,2]
    sum_g2=counts[:,1]+4*counts[:,2]
    gvar=np.maximum(sum_g2/n-(sum_g/n)**2,0.)
    nclass=(counts>0).sum(axis=1)
    df=np.where(nclass>=3,8,np.where(nclass==2,4,0))
    crit=float(chi2.isf(alpha,nominal_df))
    delta=effect*effect*n*gvar*information
    powv=np.where(df>0,ncx2.sf(crit,np.maximum(df,1),delta),0.0)
    return {'power':float(np.mean(powv)), 'mc_se':float(np.std(powv,ddof=1)/np.sqrt(nsim)),
            'p_full_rank':float(np.mean(df==8)), 'p_any_mh':float(np.mean(counts[:,2]>=1)),
            'p_atleast5_mh':float(np.mean(counts[:,2]>=5)),
            'n_mh_expected':float(n*maf*maf)}


def observed_alpha(root):
    p=root/'qtl'/'PP'/METHOD/'final'/'significant_tqtl.parquet'
    if not p.is_file():
        raise FileNotFoundError(f'Cannot infer empirical Stage I discovery threshold: {p}')
    d=pd.read_parquet(p,columns=['Pvalue','FDR'])
    x=pd.to_numeric(d.loc[pd.to_numeric(d.FDR,errors='coerce')<.05,'Pvalue'],errors='coerce')
    x=x.loc[x.between(0,1,inclusive='both')]
    if x.empty:raise RuntimeError('No eligible PPMI BH significant CTAR P-values')
    alpha=float(x.max())
    if not 0<alpha<.05:raise RuntimeError(f'Unexpected empirical discovery threshold {alpha}')
    return alpha,int(len(x)),str(p)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    p.add_argument('--code',default='')
    p.add_argument('--figroot',default='')
    p.add_argument('--mafs',default='0.05,0.10,0.20,0.30')
    p.add_argument('--effects',default='0.25,0.50,0.75,1.00')
    p.add_argument('--n-grid',default='100,150,200,250,300,400,500,700,900,1200,1600,2000,3000,4000,6000')
    p.add_argument('--rho',type=float,default=.6,help='Planning assumption for CTAR residual AR at 1-year gap')
    p.add_argument('--shape',choices=['constant','linear','quadratic'],default='constant')
    p.add_argument('--cohort-profile',choices=['PP','PD'],default='PD')
    p.add_argument('--nsim-genotype',type=int,default=2500)
    p.add_argument('--seed',type=int,default=20261029)
    p.add_argument('--power-target',type=float,default=.80)
    p.add_argument('--alpha-discovery',type=float,default=None,help='Override observed PPMI BH raw-P threshold')
    a=p.parse_args()
    root=Path(a.root)
    code=Path(a.code) if a.code else root/'2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized'
    out=Path(a.figroot) if a.figroot else root/'manuscript_figures'/'tqtl_benchmark_v4'
    work=out/'power_analysis_planning_v3';work.mkdir(parents=True,exist_ok=True)
    if not 0<=a.rho<.99:raise ValueError('--rho must in [0,0.99)')
    if a.nsim_genotype<200:raise ValueError('nsim-genotype must be >=200')
    mafs=sorted({float(x) for x in a.mafs.split(',')});effects=sorted({float(x) for x in a.effects.split(',')})
    ngrid=sorted({int(x) for x in a.n_grid.split(',')})
    if any(not 0<x<=.5 for x in mafs) or any(x<=0 for x in effects) or any(x<30 for x in ngrid):
        raise ValueError('Invalid maf/effect/n grid')
    pp,pmeta=load_profile(root,code,'PP')
    pdprof,pdmeta=load_profile(root,code,'PD')
    profile={'PP':pp,'PD':pdprof}[a.cohort_profile]
    info,design=whitened_signal_information(profile,a.rho,shape=a.shape)
    alpha,num_sig,pp_path=(float(a.alpha_discovery),None,None) if a.alpha_discovery is not None else observed_alpha(root)
    if not 0<alpha<.05:raise ValueError('Discovery alpha must be in (0,0.05)')
    rng=np.random.default_rng(a.seed)
    rows=[]
    for level,cutoff in [('discovery',alpha),('replication',.05)]:
        for maf in mafs:
            # Use identical genotype draws across effect sizes to reduce numerical noise.
            for n in ngrid:
                prob=np.array([(1-maf)**2,2*maf*(1-maf),maf*maf]);c=rng.multinomial(n,prob,size=a.nsim_genotype)
                mg=c[:,1]+2*c[:,2];var=np.maximum((c[:,1]+4*c[:,2])/n-(mg/n)**2,0.)
                actualdf=np.where((c>0).sum(axis=1)>=3,8,np.where((c>0).sum(axis=1)==2,4,0))
                crit=float(chi2.isf(cutoff,8))
                for effect in effects:
                    ncp=(effect**2)*n*var*info
                    z=np.where(actualdf>0,ncx2.sf(crit,np.maximum(actualdf,1),ncp),0.)
                    rows.append({'threshold':level,'alpha':cutoff,'n_participants':n,'maf':maf,
                                 'effect_sd_per_allele':effect,'estimated_power':float(np.mean(z)),
                                 'mc_error':float(np.std(z,ddof=1)/np.sqrt(a.nsim_genotype)),
                                 'p_three_genotype_classes':float((actualdf==8).mean()),
                                 'p_atleast_one_minor_homozygote':float((c[:,2]>=1).mean()),
                                 'p_atleast_five_minor_homozygotes':float((c[:,2]>=5).mean()),
                                 'expected_minor_homozygotes':n*maf*maf,
                                 'mean_subject_info':info,'rho':a.rho,'shape':a.shape})
    z=pd.DataFrame(rows)
    z.to_csv(work/'prospective_power_grid.csv',index=False)
    estimates=[]
    for (level,maf,effect),x in z.groupby(['threshold','maf','effect_sd_per_allele'],sort=True):
        x=x.sort_values('n_participants')
        ok=x[x.estimated_power>=a.power_target]
        n80=int(ok.n_participants.iloc[0]) if len(ok) else None
        # Additional genotype-representation diagnostic, not a formal power threshold.
        eligible=[n for n in ngrid if binom.sf(4,n,maf*maf)>=.95]
        n95_5=int(eligible[0]) if eligible else None
        estimates.append({'threshold':level,'maf':maf,'effect_sd_per_allele':effect,
                          'N80_grid_first':n80,'N95_atleast5_minor_homozygotes_grid_first':n95_5,
                          'N80_not_reached':n80 is None,'grid_max_N':max(ngrid),
                          'note':'N80 approximated under fixed rho, normalized per-allele effect; 5-homo threshold is separate stability diagnostic'})
    ns=pd.DataFrame(estimates)
    ns.to_csv(work/'minimum_n_80pct_by_maf_effect.csv',index=False)
    # Sensitivity to the residual autoregressive assumption is cheap because no CTAR refits are needed.
    sens=[]
    for rr in [0.2,0.6,0.9]:
        ri,_=whitened_signal_information(profile,rr,shape=a.shape)
        for N in sorted({pmeta['n_participants'],pdmeta['n_participants'],2*pdmeta['n_participants']}):
            for eff in [0.5,0.75]:
                pr=prospective_power(int(N),.05,eff,ri,alpha,a.nsim_genotype,np.random.default_rng(
                    np.random.SeedSequence([a.seed,int(rr*100),N,int(eff*100)]).generate_state(1)[0]))
                sens.append({'rho':rr,'N':N,'maf':.05,'effect_sd_per_allele':eff,
                             'estimated_discovery_power':pr['power'],'mc_error':pr['mc_se']})
    pd.DataFrame(sens).to_csv(work/'residual_rho_sensitivity_MAF005.csv',index=False)
    config={'analysis':'Planning power for 8-df two-channel CTAR LRT, not single-df additive test',
            'PPMI_profile':pmeta,'PDBP_profile':pdmeta,'design_profile':design,
            'empirical_PPMI_discovery_p_cutoff':alpha,'n_PPMI_significant_pairs':num_sig,
            'source_PPMI_significant':pp_path,'rho_ref_assumed':a.rho,
            'effect_definition':'true genotype effect b*dosage*h(t), b in residual SD units per allele; h(t) has RMS=1',
            'genotype_model':'Hardy-Weinberg g~Binomial(2,MAF), CTAR binary genotype channels, nominal df=8',
            'rank_model':'rank 4 if only two genotype categories, rank 0 if monomorphic; always use df=8 critical value as production does',
            'covariance':'CTAR exact innovation covariance at real irregular visit times; no estimated-rho profiling in analytic power',
            'limitations':'Planning approximation; finite-sample nuisance confounding and variance/rho estimation uncertainty not included; actual calibration requires CTAR simulation',
            'note':'N80 depends on effect, MAF, covariate adjustment, visit design, residual rho, and testing threshold. A universal minimum N cannot be supplied.',
            'seed':a.seed,'nsim_genotype':a.nsim_genotype,'mafs':mafs,'effects':effects,'n_grid':ngrid,'power_target':a.power_target}
    (work/'analysis_manifest.json').write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    print('[DONE] analytic prospective power',flush=True)
    print('[COHORTS] PP',pmeta,'PD',pdmeta,flush=True)
    print(f'[DESIGN] rho={a.rho} info per participant={info:.4f}; observed PPMI BH raw-P boundary={alpha:.6g}',flush=True)
    print('[MAF=0.05 N80] \n'+ns[(ns.maf==.05)&(ns.threshold=='discovery')].to_string(index=False),flush=True)

if __name__=='__main__': main()
