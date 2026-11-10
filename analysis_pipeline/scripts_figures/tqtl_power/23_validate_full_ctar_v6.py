#!/usr/bin/env python3
"""Optional full production CTAR simulation for one REAL frozen 8-parameter effect.

Checks both null false-positive frequency and alternative detection probability
with the actual scanner (rho-profile likelihood). This is deliberately optional
because each simulation refits CTAR at its rho grid.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True);ap.add_argument('--code',required=True);ap.add_argument('--figroot',required=True)
    ap.add_argument('--n',type=int,default=1373);ap.add_argument('--maf',default='auto',help='auto uses source pair original MAF, preserving coded-dose direction')
    ap.add_argument('--gene',default=None)
    ap.add_argument('--snp',default=None)
    ap.add_argument('--profile-quantile',type=float,default=0.5,help='If gene/snp omitted, auto-select observed-PVE quantile (0..1)')
    ap.add_argument('--effect-multiplier',type=float,default=1.0)
    ap.add_argument('--rho',default='auto');ap.add_argument('--reps',type=int,default=30)
    ap.add_argument('--rho-points',type=int,default=100);ap.add_argument('--seed',type=int,default=20261029)
    ap.add_argument('--alpha',type=float,default=None,help='default: same fixed PPMI BH raw-P boundary as power planning')
    a=ap.parse_args()
    if a.reps<5 or a.n<50 or not 0<a.effect_multiplier<=1:
        raise ValueError('Invalid validation parameters')
    root=Path(a.root);code=Path(a.code);out=Path(a.figroot)/'power_analysis_CTAR_v6'/'exact_ctar_validation';out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(code))
    from gengps.amp_pd.config import AmpPdPaths
    from gengps.amp_pd.real_qtl import (load_cohort_metadata,_augment_fixed_genotype_pcs,
         _prepare_irregular_followup_rows,method_covariates,_drop_missing_covariates,bfile_for_cohort)
    from gengps.methods.ctar_mptga import ContinuousTimeMPTGA
    from gengps.basis import TimeBasis
    manifest=json.loads((Path(a.figroot)/'power_analysis_CTAR_v6/power_manifest_v6.json').read_text())
    alpha=a.alpha if a.alpha is not None else float(manifest['alpha_discovery_fixed_empirical_BH'])
    rho=float(manifest['rho_fixed']) if a.rho=='auto' else float(a.rho)
    if not 0<=rho<.98:raise ValueError('rho out of bounds')
    if not 0<alpha<1:raise ValueError('test alpha out of range')
    templates=pd.read_csv(Path(a.figroot)/'power_analysis_CTAR_v6/calibrated_real_full8_CTAR_PVE.csv')
    if (a.gene is None)!=(a.snp is None):
        raise ValueError('Specify gene and snp together, or neither for automatic selection')
    if a.gene is None:
        if not 0<=a.profile_quantile<=1: raise ValueError('profile quantile outside [0,1]')
        col='observed_PVE_original_VarGenetic_over_VarRNA'
        z=templates[np.isfinite(pd.to_numeric(templates[col],errors='coerce'))].copy()
        t=float(z[col].quantile(a.profile_quantile))
        q=z.loc[(z[col]-t).abs().idxmin()]
        a.gene=str(q['gene']);a.snp=str(q['lead_snp'])
        print(f'[AUTO TEMPLATE] quantile={a.profile_quantile}: {a.gene}, {a.snp}, observed PVE={q[col]:.4g}',flush=True)
    else:
        x=templates[(templates.gene.astype(str)==a.gene)&(templates.lead_snp.astype(str)==a.snp)]
        if len(x)!=1: raise RuntimeError('Selected gene and SNP not unique in calibration table')
        q=x.iloc[0]
    maf=float(q['MAF_observed']) if a.maf=='auto' else float(a.maf)
    if not 0<maf<=.5:raise ValueError('MAF must be >0 and <=.5')
    theta=np.array([float(q[f'beta_g_{j}']) for j in range(8)]) / float(q['residual_sd']) * a.effect_multiplier
    basis=TimeBasis.from_dict({'kind':'cubic','parameterization':'raw_followup_time_polynomial','columns':['1','t','t2','t3']})
    paths=AmpPdPaths(root=str(root));m=load_cohort_metadata(paths,'PD')
    ids=set(pd.read_parquet(root/'data/expression/amp_pd_expression_z.parquet',columns=['sample_id']).sample_id.astype(str))
    m=m[m.sample_id.astype(str).isin(ids)].copy()
    m=_augment_fixed_genotype_pcs(paths,'PD',m,n_pcs=8)
    m,_,_=_prepare_irregular_followup_rows(m,pd.DataFrame(index=m.index))
    covars=method_covariates('submitted_followup_ctar_mptga','PD',m,cell_adjust=True)
    m,_=_drop_missing_covariates(m,pd.DataFrame(index=m.index),covars)
    fam=pd.read_csv(str(bfile_for_cohort(paths,'PD'))+'.fam',sep=r'\s+',header=None,usecols=[0,1],dtype=str)
    pids=set(m.participant_id.astype(str));c=0 if fam.iloc[:,0].isin(pids).sum()>=fam.iloc[:,1].isin(pids).sum() else 1
    m=m[m.participant_id.astype(str).isin(set(fam.iloc[:,c]))].copy()
    donor=[g.sort_values('followup_time') for _,g in m.groupby('participant_id')]
    if len(donor)<50:raise RuntimeError('No usable real PDBP sample/time profiles')
    scanner=ContinuousTimeMPTGA(basis_kind='cubic',genotype_mode='binary',rho_points=a.rho_points,
                                  batch_size=2,device='cpu',dtype='float64',reference_interval=1.,rho_max=.99)
    cis=pd.DataFrame({'geneid':['SIM_ALT','SIM_NULL'],'snpid':['SIM_SNP','SIM_SNP']})
    dst=out/f'validation_{a.gene}_{a.snp}_fac{a.effect_multiplier:.2f}_N{a.n}_maf{maf:.3f}_rho{rho:.2f}.csv'
    rows=[]
    if dst.exists():rows=pd.read_csv(dst).to_dict('records')
    for r in range(len(rows),a.reps):
        rng=np.random.default_rng(np.random.SeedSequence([a.seed,r]).generate_state(1)[0]);inds=rng.integers(0,len(donor),size=a.n)
        p2=maf if float(q['dosage_allele_frequency'])<=.5 else 1-maf
        gs=rng.binomial(2,p2,size=a.n)
        frames=[];response={};genos={}
        for i,(k,g) in enumerate(zip(inds,gs)):
            src=donor[int(k)];pid=f'SIM{i:07d}'
            t=src.followup_time.to_numpy(float)
            temp=src.copy();temp['participant_id']=pid
            temp['sample_id']=[f'{pid}_V{j}' for j in range(len(t))]
            temp.index=temp.sample_id.astype(str);frames.append(temp)
            b=basis.transform(t)
            # exact binary Stage1: g==2 -> beta[0:4]; g==1 -> beta[4:8].
            f=(b@theta[:4] if g==2 else (b@theta[4:] if g==1 else np.zeros(len(t))))
            e=np.zeros(len(t));e[0]=rng.standard_normal()
            for j in range(1,len(t)):
                ar=rho**(t[j]-t[j-1]) if rho>0 else 0.0
                e[j]=ar*e[j-1]+np.sqrt(1-ar*ar)*rng.standard_normal()
            genos[pid]=int(g)
            response.update({sid:(float(x),float(y)) for sid,x,y in zip(temp.index,f+e,e)})
        meta=pd.concat(frames).sort_values(['participant_id','followup_time'],kind='mergesort')
        y=pd.DataFrame([response[sid] for sid in meta.index],index=meta.index,columns=['SIM_ALT','SIM_NULL'])
        geno=pd.DataFrame({'SIM_SNP':list(genos.values())},index=list(genos.keys()))
        fit=scanner.scan(meta,y,geno,cis,covariate_cols=covars,id_col='participant_id',time_col='followup_time')
        pv=dict(zip(fit.Ensembl_ID.astype(str),fit.Pvalue.astype(float)))
        row={'rep':r,'gene':a.gene,'lead_snp':a.snp,'effect_multiplier':a.effect_multiplier,'maf':maf,'n':a.n,'rho':rho,
             'p_alt':pv['SIM_ALT'],'p_null':pv['SIM_NULL'],'test_alpha':alpha,
             'n_g0':int((gs==0).sum()),'n_g1':int((gs==1).sum()),'n_g2':int((gs==2).sum()),'P_all_three_classes':bool(len(np.unique(gs))==3)}
        rows.append(row);pd.DataFrame(rows).to_csv(dst,index=False)
        print(f'[VALIDATE] {r+1}/{a.reps} alt P={row["p_alt"]:.3g} null P={row["p_null"]:.3g}',flush=True)
    z=pd.DataFrame(rows)
    from scipy.stats import binomtest
    k=int((z.p_alt<alpha).sum());N=len(z)
    ci=binomtest(k,N).proportion_ci(confidence_level=.95)
    print('[VALIDATION CI] alt power k/n=',f'{k}/{N}', '95% exact binomial interval',f'{ci.low:.3f}–{ci.high:.3f}')
    print('[SUMMARY] N=',a.n,' MAF=',maf,' empirical-alt P<alpha=',alpha,':', (z.p_alt<alpha).mean(),
          'null type-I at fixed alpha=',alpha,':', (z.p_null<alpha).mean(), 'null nominal P<0.05=', (z.p_null<.05).mean(),' n=',len(z),' file=',dst,flush=True)
if __name__=='__main__':main()
