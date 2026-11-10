#!/usr/bin/env python3
"""Observed-trajectory and cohort-design CTAR power sensitivity (fast analytic layer).

Original-paper effect: PVE = Var[f_G(t)] / Var[observed RNA]; f_G(t) contains
ALL two genotype dummy channels × four raw-time polynomial coefficients.
Planning power: prospective fixed-rho 8-df GLS noncentral chi-square approximation,
using each real variant's observed coded-genotype frequencies, observed residual
SD and full vector of eight fitted coefficients; no additive dosage assumption,
no universal MAF=0.05 or 0.21 assumption.

NOTE: calibrated profiles come from significant frozen Stage I candidates,
therefore effect sizes and conditional power are affected by selection/winner's curse.
The conservative no-full-three-genotypes convention can underestimate production
scanner's reduced-rank detection. Not a validated minimum sample size.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2,ncx2
from legacy_power_v4_math import load_profile,info_per_donor

BETAS=[f'beta_g_{i}' for i in range(8)]


def per_donor_q(counts, theta, K, L):
    """Real coded 0/1/2 genotype; g==2 uses theta[0:4], g==1 theta[4:8]."""
    n=np.sum(counts,axis=1).astype(float)
    q1=counts[:,1]/n
    q2=counts[:,2]/n
    b2=theta[:4]; b1=theta[4:]
    m=q1[:,None]*b1[None,:]+q2[:,None]*b2[None,:]
    return np.maximum(0, n*(q1*(b1@K@b1)+q2*(b2@K@b2)-np.einsum('ni,ij,nj->n',m,L,m)))


def mc_power(counts,theta,K,L,alpha):
    """Eight-df detection event with full three-class estimability requirement."""
    ncp=per_donor_q(counts,theta,K,L)
    has_all=(counts>0).all(axis=1)
    ps=np.zeros(len(counts))
    ps[has_all]=ncx2.sf(chi2.isf(alpha,8),8,ncp[has_all])
    return float(ps.mean()),float(has_all.mean()),float(np.quantile(ncp,.5))


def default_alpha(root):
    f=root/'qtl/PP/submitted_followup_ctar_mptga/final/significant_tqtl.parquet'
    if not f.is_file(): raise FileNotFoundError(f)
    x=pd.read_parquet(f,columns=['Pvalue','FDR'])
    p=pd.to_numeric(x.Pvalue,errors='coerce');q=pd.to_numeric(x.FDR,errors='coerce')
    p=p[(p>0)&(q<.05)&np.isfinite(p)]
    if p.empty:raise RuntimeError('Cannot infer PP discovery raw-P BH cutoff')
    return float(p.max())


def make_probs(r, use_hwe=False):
    n=np.array([r['n_hom_ref'],r['n_het'],r['n_hom_alt']],dtype=float)
    if not np.isfinite(n).all() or (n<0).any() or n.sum()<40:
        raise ValueError('Genotype counts missing or invalid')
    if use_hwe:
        q=(n[1]+2*n[2])/(2*n.sum())  # coded dosage-2 allele frequency; NOT minor-allele substitution
        return np.array([(1-q)**2,2*q*(1-q),q*q]),float(min(q,1-q)),float(q)
    return n/n.sum(), float(min((n[1]+2*n[2])/(2*n.sum()),1-(n[1]+2*n[2])/(2*n.sum()))), float((n[1]+2*n[2])/(2*n.sum()))


def power_table(calib, ns, alphaPP, KPP,LPP,KPD,LPD, n_draws=600,seed=20261029,
                multipliers=(1.0,.75,.5), hwe=False):
    rng=np.random.default_rng(seed)
    records=[]; excluded=[]
    # Different PP/PD cohort visits and covariate structure via K/L.
    for ix,row in calib.reset_index(drop=True).iterrows():
        try:
            probs,maf,encoded_p2=make_probs(row,hwe)
            th=row[BETAS].to_numpy(dtype=float)/float(row['residual_sd'])
            if not np.isfinite(th).all() or not np.isfinite(maf) or float(row['residual_sd'])<=0:
                raise ValueError('Invalid beta/sigma/MAF')
            for n in ns:
                cnt=rng.multinomial(n,probs,size=n_draws)
                for fac in multipliers:
                    pp,rank,lambdaPP=mc_power(cnt,th*fac,KPP,LPP,alphaPP)
                    pd_,_,lambdaPD=mc_power(cnt,th*fac,KPD,LPD,.05)
                    records.append({'gene':row['gene'],'lead_snp':row['lead_snp'],
                        'MAF':maf,'coded_allele_frequency':encoded_p2,
                        'observed_PVE_original':row.get('observed_PVE_original_VarGenetic_over_VarRNA',np.nan),
                        'observed_partial_proxy':row.get('signal_fraction',np.nan),
                        'n_participants':int(n),'effect_multiplier':fac,
                        'power_discovery':pp,'power_replication':pd_,
                        'power_joint_assuming_independent_cohorts':pp*pd_,
                        'full_three_class_probability':rank,
                        'median_ncp_PP':lambdaPP,'median_ncp_PD':lambdaPD})
        except (ValueError,KeyError,TypeError) as exc:
            excluded.append({'gene':row.get('gene'),'lead_snp':row.get('lead_snp'),'reason':str(exc)})
        if (ix+1)%50==0: print(f'[POWER] {ix+1}/{len(calib)} profiles',flush=True)
    return pd.DataFrame(records),pd.DataFrame(excluded,columns=['gene','lead_snp','reason'])


def summarize_raw(raw, ns, threshold=.8):
    summary=(raw.groupby(['n_participants','effect_multiplier'],as_index=False)
              .agg(n_profiles=('gene','size'),
                   pct_profiles_80_discovery=('power_discovery',lambda x:float(np.mean(x>=threshold))),
                   pct_profiles_80_replication=('power_replication',lambda x:float(np.mean(x>=threshold))),
                   pct_profiles_80_joint=('power_joint_assuming_independent_cohorts',lambda x:float(np.mean(x>=threshold))),
                   median_power_PP=('power_discovery','median'),
                   median_power_PD=('power_replication','median'),
                   median_joint=('power_joint_assuming_independent_cohorts','median'),
                   median_full_rank=('full_three_class_probability','median')))
    n80=[]
    for (gene,snp,fac),one in raw.groupby(['gene','lead_snp','effect_multiplier'],sort=False):
        one=one.sort_values('n_participants')
        out={'gene':gene,'lead_snp':snp,'effect_multiplier':fac,'MAF':float(one.MAF.iloc[0]),
             'observed_PVE_original':one.observed_PVE_original.iloc[0],
             'observed_partial_proxy':one.observed_partial_proxy.iloc[0]}
        for col,name in [('power_discovery','discovery'),('power_replication','replication'),('power_joint_assuming_independent_cohorts','joint')]:
            reached=one[one[col]>=threshold]
            out[f'N80_{name}_first_grid']=int(reached.n_participants.iloc[0]) if len(reached) else np.nan
        n80.append(out)
    return summary,pd.DataFrame(n80)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True);p.add_argument('--code',required=True);p.add_argument('--figroot',required=True)
    p.add_argument('--n-grid',default='70,100,150,200,300,400,500,700,900,1100,1200,1373,1405,1600,2000,2600,3500,5000,7000')
    p.add_argument('--draws',type=int,default=600)
    p.add_argument('--rho',default='auto')
    p.add_argument('--effect-multipliers',default='1.0,0.75,0.5')
    p.add_argument('--hwe',action='store_true',help='Use HWE expected counts, not observed genotype proportions')
    p.add_argument('--alpha-discovery',type=float,default=None)
    p.add_argument('--seed',type=int,default=20261029)
    p.add_argument('--power-target',type=float,default=.8)
    p.add_argument('--max-profiles',type=int,default=0,help='0=use every calibrated real SNP–gene profile')
    a=p.parse_args();root=Path(a.root);out=Path(a.figroot)/'power_analysis_CTAR_v6';out.mkdir(parents=True,exist_ok=True)
    f=out/'calibrated_real_full8_CTAR_PVE.csv'
    if not f.is_file():raise FileNotFoundError(f)
    calib=pd.read_csv(f,low_memory=False)
    required=['gene','lead_snp','n_hom_ref','n_het','n_hom_alt','residual_sd']+BETAS
    missing=set(required)-set(calib.columns)
    if missing:raise KeyError(f'Calibration required columns missing: {sorted(missing)}')
    calib=calib.drop_duplicates(['gene','lead_snp']).dropna(subset=required).reset_index(drop=True)
    if a.max_profiles>0:
        calib=calib.sample(min(a.max_profiles,len(calib)),random_state=a.seed).reset_index(drop=True)
    if len(calib)<5:raise RuntimeError('Too few actual CTAR profiles; see calibration exclusions')
    ns=sorted(set(int(x) for x in a.n_grid.split(',')))
    facs=sorted(set(float(x) for x in a.effect_multipliers.split(',')),reverse=True)
    if min(ns)<30 or a.draws<100 or any(f<=0 or f>1 for f in facs):raise ValueError('Invalid grid/draws/multipliers')
    rhos=pd.to_numeric(calib.get('rho_PP',pd.Series(dtype=float)),errors='coerce').dropna()
    rhos=rhos[rhos.between(0,.98,inclusive='left')]
    if a.rho=='auto' and rhos.empty:
        # Primary eligible-IV CSV need not contain a column named 'rho'.
        # Fetch the original Stage1 CTAR rho distribution, not an invented default.
        pf=root/'qtl/PP/submitted_followup_ctar_mptga/final/significant_tqtl.parquet'
        try:
            rhos=pd.to_numeric(pd.read_parquet(pf,columns=['rho']).rho,errors='coerce').dropna()
        except Exception as e:
            raise RuntimeError(f'Cannot establish actual Stage1 rho distribution from {pf}; supply --rho') from e
        rhos=rhos[rhos.between(0,.98,inclusive='left')]
    rho=float(rhos.median()) if a.rho=='auto' else float(a.rho)
    if not 0<=rho<.98 or not np.isfinite(rho):raise ValueError('Invalid rho')
    alpha=default_alpha(root) if a.alpha_discovery is None else a.alpha_discovery
    if not 0<alpha<=.05:raise ValueError('Invalid alpha')
    metaPP,covPP=load_profile(root,Path(a.code),'PP')
    metaPD,covPD=load_profile(root,Path(a.code),'PD')
    KPP,LPP,infPP=info_per_donor(metaPP,covPP,rho)
    KPD,LPD,infPD=info_per_donor(metaPD,covPD,rho)
    nPP=int(metaPP.participant_id.nunique())
    nPD=int(metaPD.participant_id.nunique())
    # Actual cohorts generally have different independent-participant counts.
    ns=sorted(set(ns+[nPP,nPD]))
    print(f'[DESIGN] PPMI {metaPP.participant_id.nunique()} people, PDBP {metaPD.participant_id.nunique()} people, rho={rho:.3f}, alphaDiscovery={alpha:.4g}',flush=True)
    raw,errors=power_table(calib,ns,alpha,KPP,LPP,KPD,LPD,a.draws,a.seed,facs,a.hwe)
    if raw.empty:raise RuntimeError('No prospective planning results')
    raw.to_csv(out/'actual_pair_power_grid.csv',index=False)
    errors.to_csv(out/'power_exclusions.csv',index=False)
    summary,n80=summarize_raw(raw,ns,a.power_target)
    summary.to_csv(out/'power_summary_by_N.csv',index=False)
    n80.to_csv(out/'N80_by_real_pair.csv',index=False)
    # Actual PP and PD Ns are not equal. Never interpret the same-N design curve
    # as a direct estimate of the existing 1405/1373 cohort pair.
    key=['gene','lead_snp','effect_multiplier']
    ap=raw.loc[raw.n_participants.eq(nPP),key+['power_discovery','MAF']].rename(columns={'MAF':'MAF_original'})
    ad=raw.loc[raw.n_participants.eq(nPD),key+['power_replication']]
    actual=ap.merge(ad,on=key,how='inner',validate='one_to_one')
    actual['power_joint_at_actual_sample_sizes']=actual.power_discovery*actual.power_replication
    actual.to_csv(out/'actual_cohort_pairwise_joint_power.csv',index=False)
    actual_summary=(actual.groupby('effect_multiplier',as_index=False).agg(
        n_profiles=('gene','size'),
        fraction_joint_power80=('power_joint_at_actual_sample_sizes',lambda v:float(np.mean(v>=a.power_target))),
        median_joint_power=('power_joint_at_actual_sample_sizes','median'),
        fraction_PP_power80=('power_discovery',lambda v:float(np.mean(v>=a.power_target))),
        fraction_PD_power80=('power_replication',lambda v:float(np.mean(v>=a.power_target)))))
    actual_summary['n_PPMI']=nPP;actual_summary['n_PDBP']=nPD
    actual_summary.to_csv(out/'actual_cohort_conditional_power_summary.csv',index=False)
    # annotate support caveat for future readers
    manifest={'source':str(f),'n_profiles':int(raw[['gene','lead_snp']].drop_duplicates().shape[0]),
      'n_rejected_profiles':len(errors),'PPMI_actual_N':int(metaPP.participant_id.nunique()),
      'PDBP_actual_N':int(metaPD.participant_id.nunique()),'cohort_info_PP':infPP,'cohort_info_PD':infPD,
      'rho_fixed':rho,'alpha_discovery_fixed_empirical_BH':alpha,'alpha_replication_nominal':.05,
      'full_model':'two genotype dummy channels times (1,t,t^2,t^3), 8-df joint genetic effect',
      'genotype_coding':'0 baseline, g=1 beta_g_4..7, g=2 beta_g_0..3; use ACTUAL coded-frequency not MAF orientation',
      'sample_genotype_distribution':'Hardy-Weinberg expected' if a.hwe else 'PPMI observed called-category proportions; multinomial',
      'signal_strength':'unshrunk eight fitted CTAR coefficients / empirical residual SD; multipliers are declared selection-bias sensitivity scenarios, NOT corrected effect estimates',
      'original_manuscript_PVE':'Var(f_G(t)) / Var(observed RNA), includes full 8-parameter genetic curve, descriptive and selection-biased',
      'alpha_limitation':'PPMI empirical BH raw P cutoff kept fixed across sample-size grid; BH not rerun; replication P<0.05',
      'joint_power_limitation':'product of discovery and replication marginal powers, assuming independent cohorts and the same underlying genetic effect across cohorts',
      'N80_limitation':'First tested N with approximate power >=0.8 for a single cohort (or both cohorts if same N), conditional on fitted effects; not a generic minimum sample size',
      'power_limitation':'8-df fixed-rho Gaussian GLS noncentral chi-square; 0 when any of three genotype classes absent; production CTAR estimates rho and may use rank changes',
      'sampling_limitation':'Only frozen replicated single-IV profiles, selected on significance; winner’s curse; do not generalize to all unobserved tQTLs',
      'power_target':a.power_target,'cohort_design':'actual PPMI and PDBP longitudinal visit/covariate profiles',
      'grid_N':ns,'effect_multipliers':facs,'n_draws':a.draws,'seed':a.seed,'HWE_assumption':a.hwe}
    (out/'power_manifest_v6.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    print('[COMPLETE] Real-profile 8df conditional planning; see power_analysis_CTAR_v6',flush=True)

if __name__=='__main__':main()
