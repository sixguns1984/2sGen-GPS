#!/usr/bin/env python3
"""CTAR 8-df planning sensitivity with matched standardized signal strengths.

No additive dosage assumption. Trajectory shapes are empirical Stage1 8-coefficient
vectors, but amplitudes are rescaled to a COMMON GLS partial-signal fraction
at a fixed reference MAF / longitudinal design / nuisance covariance.

The partial fraction here is a *design-based GLS signal fraction*, not
observational genetic heritability nor an unbiased partial R2 estimate.
The test is an approximate fixed-rho 8-df noncentral-chi-square test, NOT
production profiled-rho CTAR likelihood. Optional full simulation is required
before strong numerical minimum-N claims.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import chi2,ncx2,binom
from legacy_power_v4_math import load_profile, info_per_donor

def ref_q(p):return 2*p*(1-p),p*p

def information(theta1,theta2,maf,K,L):
    q1,q2=ref_q(maf); mix=q1*theta1+q2*theta2
    return float(q1*(theta1@K@theta1)+q2*(theta2@K@theta2)-mix@L@mix)

def normalize_shape(raw8,sigma,target_r2,pref,K,L,obs_per_person):
    theta=np.asarray(raw8,float)/float(sigma)
    b2=theta[:4]; b1=theta[4:] # production CTAR binary g=2->coeff 0:4, g=1->4:8
    base=information(b1,b2,pref,K,L)
    if base<=1e-15 or not np.isfinite(base):raise ValueError('Zero GLS genetic information')
    target_info=obs_per_person*target_r2/(1-target_r2)
    factor=float(np.sqrt(target_info/base))
    return b1*factor,b2*factor,factor,float(base),float(target_info)

def conditional_power(counts,theta1,theta2,K,L,alpha):
    n=np.asarray(counts.sum(axis=1),float)
    q1=counts[:,1]/n;q2=counts[:,2]/n
    r1=float(theta1@K@theta1);r2=float(theta2@K@theta2)
    mix=q1[:,None]*theta1[None,:]+q2[:,None]*theta2[None,:]
    ncp=n*np.maximum(q1*r1+q2*r2-np.einsum('ni,ij,nj->n',mix,L,mix),0.0)
    full_rank=(counts>0).all(axis=1)
    # Production 8-df test is not estimable if one genotype category is entirely missing.
    # Define this as 0 full-model detection power, and report P(full rank) separately.
    prob=np.zeros(len(counts),float)
    prob[full_rank]=ncx2.sf(chi2.isf(alpha,8),8,ncp[full_rank])
    return float(prob.mean()),float(prob.std(ddof=1)/np.sqrt(len(prob))),float(full_rank.mean())

def pick_profiles(z,n,seed):
    z=z.copy()
    z['signal_fraction']=pd.to_numeric(z.signal_fraction,errors='coerce')
    z['MAF_observed']=pd.to_numeric(z.MAF_observed,errors='coerce')
    z=z.dropna(subset=['signal_fraction','MAF_observed','residual_sd']+[f'beta_g_{i}' for i in range(8)])
    z=z[(z.signal_fraction>0)&(z.residual_sd>0)&(z.MAF_observed>0)&(z.MAF_observed<.5)]
    z=z.drop_duplicates(['gene','lead_snp'])
    if len(z)<3:raise RuntimeError('Too few valid empirical CTAR 8-coefficient profiles')
    rng=np.random.default_rng(seed)
    # Include diverse observed effect shapes, not just three selected genes.
    z=z.sort_values('signal_fraction').reset_index(drop=True)
    n=min(n,len(z));inds=np.sort(rng.choice(len(z),n,replace=False))
    return z.iloc[inds].copy().reset_index(drop=True)

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--root',required=True);pa.add_argument('--code',required=True);pa.add_argument('--figroot',required=True)
    pa.add_argument('--pve-levels',default='0.005,0.01,0.02,0.05')
    pa.add_argument('--mafs',default='0.05,0.10,0.20,0.30')
    pa.add_argument('--n-grid',default='100,150,200,250,300,400,500,700,900,1200,1373,1405,1600,2000,2600,3500,5000,7000')
    pa.add_argument('--rho',default='auto',help='auto=median PP calibrated CTAR rho, or numeric [0,1)')
    pa.add_argument('--reference-maf',default='auto',help='auto=median MAF among frozen Stage1 IVs')
    pa.add_argument('--profile',choices=['PP','PD'],default='PD')
    pa.add_argument('--n-shapes',type=int,default=24)
    pa.add_argument('--draws',type=int,default=3000)
    pa.add_argument('--seed',type=int,default=20261029)
    pa.add_argument('--alpha-discovery',type=float,default=None)
    pa.add_argument('--power-target',type=float,default=.8)
    a=pa.parse_args();root=Path(a.root);out=Path(a.figroot)/'power_analysis_CTAR_v5';out.mkdir(parents=True,exist_ok=True)
    z=pd.read_csv(out/'empirical_CTAR_eight_parameter_effects.csv',low_memory=False)
    if a.rho=='auto':
        values=pd.to_numeric(z.get('rho_PP',pd.Series(dtype=float)),errors='coerce').dropna()
        if values.empty:
            ppfile=root/'qtl/PP/submitted_followup_ctar_mptga/final/significant_tqtl.parquet'
            values=pd.to_numeric(pd.read_parquet(ppfile,columns=['rho']).rho,errors='coerce').dropna()
        rho=float(values[(values>=0)&(values<.98)].median())
    else:rho=float(a.rho)
    if not np.isfinite(rho) or not 0<=rho<.98:raise ValueError(f'Invalid rho={rho}; use --rho numeric')
    ref_file=out/'MAF_frozen_Stage1_IVs.csv'
    if a.reference_maf=='auto':
        if not ref_file.exists():raise FileNotFoundError(f'Required MAF reference missing: {ref_file}')
        refmaf=float(pd.read_csv(ref_file).MAF.median())
    else:refmaf=float(a.reference_maf)
    if not 0<refmaf<.5:raise ValueError(f'Invalid reference MAF {refmaf}')
    r2s=sorted(set(float(x) for x in a.pve_levels.split(',')))
    mafs=sorted(set([refmaf]+[float(x) for x in a.mafs.split(',')]))
    ns=sorted(set(int(x) for x in a.n_grid.split(',')))
    if min(ns)<50 or max(r2s)>=.5 or min(r2s)<=0 or any(not 0<p<=.5 for p in mafs) or a.draws<500:
        raise ValueError('Invalid PVE/MAF/N/draws')
    shapes=pick_profiles(z,a.n_shapes,a.seed)
    ppro,ppc=load_profile(root,a.code,'PP')
    pdro,pdc=load_profile(root,a.code,'PD')
    refdata,refcov=(ppro,ppc) if a.profile=='PP' else (pdro,pdc)
    K,L,info=info_per_donor(refdata,refcov,rho)
    visits=float(info['n_observations']/info['n_participants'])
    if a.alpha_discovery is None:
        sig=root/'qtl/PP/submitted_followup_ctar_mptga/final/significant_tqtl.parquet'
        q=pd.read_parquet(sig,columns=['Pvalue','FDR'])
        d=pd.to_numeric(q.Pvalue,errors='coerce');F=pd.to_numeric(q.FDR,errors='coerce')
        good=d[(F<.05)&(d>0)&np.isfinite(d)]
        if good.empty:raise RuntimeError('Cannot determine empirical discovery BH boundary')
        alpha=float(good.max())
    else:alpha=a.alpha_discovery
    if not 0<alpha<=.05:raise ValueError(f'Invalid discovery alpha {alpha}')
    print(f'[REFERENCE] MAF={refmaf:.4f} rho={rho:.4f} visits/person={visits:.2f} empirical alpha={alpha:.6g}; cohort N={ppro.participant_id.nunique()}/{pdro.participant_id.nunique()}',flush=True)
    chosen=shapes[['gene','lead_snp','signal_fraction','MAF_observed','residual_sd']+[f'beta_g_{j}' for j in range(8)]].copy()
    chosen.to_csv(out/'selected_real_CTAR_shapes_v5.csv',index=False)
    rng=np.random.default_rng(a.seed)
    # One common genotype count draw per N,MAF. Shared by all PVE and real-shape settings.
    genotype_draws={(n,p):rng.multinomial(n,[(1-p)**2,2*p*(1-p),p*p],size=a.draws)
                    for n in ns for p in mafs}
    records=[];scaling=[]
    for ir,row in shapes.iterrows():
        b=np.array([row[f'beta_g_{j}'] for j in range(8)],float)
        for target in r2s:
            try:
                b1,b2,scale,original,targetinfo=normalize_shape(b,row.residual_sd,target,refmaf,K,L,visits)
            except ValueError:
                continue
            scaling.append({'shape_id':ir,'gene':row.gene,'lead_snp':row.lead_snp,'target_R2':target,
                'raw_inform_per_donor':original,'target_inform_per_donor':targetinfo,'scaling_factor':scale,
                'beta_g_0':b2[0],'beta_g_1':b2[1],'beta_g_2':b2[2],'beta_g_3':b2[3],
                'beta_g_4':b1[0],'beta_g_5':b1[1],'beta_g_6':b1[2],'beta_g_7':b1[3]})
            for n in ns:
                for maf in mafs:
                    counts=genotype_draws[(n,maf)]
                    for thresh,al in [('discovery_fixed_BH',alpha),('replication_nominal',.05)]:
                        power,se,rank=conditional_power(counts,b1,b2,K,L,al)
                        records.append({'shape_id':ir,'gene':row.gene,'lead_snp':row.lead_snp,
                          'target_R2_at_reference_MAF':target,'maf':maf,'n_participants':n,
                          'threshold':thresh,'alpha':al,'power':power,'genotype_MC_se':se,
                          'P_all_three_genotypes_present':rank})
        if (ir+1)%5==0:print(f'[POWER] normalized real shapes {ir+1}/{len(shapes)}',flush=True)
    raw=pd.DataFrame(records)
    if raw.empty:raise RuntimeError('No usable normalized shapes')
    raw.to_csv(out/'power_per_real_shape_v5.csv',index=False)
    pd.DataFrame(scaling).to_csv(out/'normalized_eight_coefficient_templates_v5.csv',index=False)
    keys=['target_R2_at_reference_MAF','maf','n_participants','threshold','alpha']
    agg=(raw.groupby(keys,as_index=False).agg(
        median_power=('power','median'),p10_power=('power',lambda v:float(np.quantile(v,.1))),
        p90_power=('power',lambda v:float(np.quantile(v,.9))),
        min_power=('power','min'),max_power=('power','max'),
        mean_power=('power','mean'),
        full_genotype_rank_probability=('P_all_three_genotypes_present','mean'),
        n_shapes=('shape_id','nunique')))
    agg.to_csv(out/'power_standardized_PVE_grid_v5.csv',index=False)
    result=[]
    for (t,maf,th),group in agg.groupby(['target_R2_at_reference_MAF','maf','threshold']):
        d=group.sort_values('n_participants')
        def crossing(column):
            hits=d[d[column]>=a.power_target]
            return int(hits.n_participants.iloc[0]) if len(hits) else np.nan
        result.append({'target_R2_at_reference_MAF':t,'maf':maf,'threshold':th,
                       'N80_median_first_grid':crossing('median_power'),
                       'N80_p10_shapes_first_grid':crossing('p10_power'),
                       'N80_p90_shapes_first_grid':crossing('p90_power'),
                       'max_tested_N':max(ns),'n_shapes':int(d.n_shapes.iloc[0]),
                       'definition':'first tested N with approximate power >=.80; not a universal minimum'})
    pd.DataFrame(result).to_csv(out/'N80_standardized_PVE_v5.csv',index=False)
    manifest={'method':'CTAR-8df-fixed-rho-GLS information and noncentral chi-square approximation',
              'genetic_encoding':'genotype==2->beta_g_0:3, genotype==1->beta_g_4:7 (not additive)',
              'target_effect':'Covariate-and-time adjusted, GLS-whitened genetic signal / (signal+residual), defined at reference MAF and cohort visit design',
              'R2_is_not':'Unbiased heritability, GWAS PVE, or post-hoc variance explained of each discovery SNP',
              'fixed_genotype_coefficients_across_maf':True,
              'n_accepted_shapes':int(raw.shape_id.nunique()),'source_shapes':str(out/'empirical_CTAR_eight_parameter_effects.csv'),
              'reference_maf':refmaf,'rho':rho,'real_cohort_profile':a.profile,
              'participants_PP':int(ppro.participant_id.nunique()),'participants_PD':int(pdro.participant_id.nunique()),
              'mean_visits_per_donor':visits,'discovery_fixed_BH_rawP_cutoff':alpha,
              'nominal_replication_P_cutoff':.05,'power_target':a.power_target,'grid_N':ns,'MAF_scenarios':mafs,
              'PVE_scenarios':r2s,'draws':a.draws,'seed':a.seed,
              'rank_handling':'8-df detection power set to zero if simulated sample lacks any genotype category; report full-rank probability',
              'limitations':'A design planning approximation; empirical FDR threshold not re-estimated at other N; rho estimated in production, not fixed; candidate curves selected among significant IVs; gene and cohort effects may differ; validate important scenarios using production CTAR simulations.',
              'metadata':info}
    (out/'power_manifest_v5.json').write_text(json.dumps(manifest,indent=2))
    print(f'[COMPLETE] {len(shapes)} real trajectory shapes, MAFref={refmaf:.4f}, rho={rho:.3f}, N PPMI={manifest["participants_PP"]}, PDBP={manifest["participants_PD"]}',flush=True)

if __name__=='__main__':main()
