#!/usr/bin/env python3
"""Optional production CTAR calibration of a V5 normalized 8-parameter effect.

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
    ap.add_argument('--n',type=int,default=1400);ap.add_argument('--maf',type=float,default=.05)
    ap.add_argument('--shape-id',type=int,default=0)
    ap.add_argument('--target-r2',type=float,default=.01)
    ap.add_argument('--rho',type=float,default=.6);ap.add_argument('--reps',type=int,default=30)
    ap.add_argument('--rho-points',type=int,default=100);ap.add_argument('--seed',type=int,default=20261029)
    ap.add_argument('--alpha',type=float,default=None,help='default: same fixed PPMI BH raw-P boundary as power planning')
    a=ap.parse_args()
    if a.reps<5 or a.n<50 or not 0<a.maf<=.5:raise ValueError('Invalid validation parameters')
    root=Path(a.root);code=Path(a.code);out=Path(a.figroot)/'power_analysis_CTAR_v5'/'exact_ctar_validation';out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(code))
    from gengps.amp_pd.config import AmpPdPaths
    from gengps.amp_pd.real_qtl import (load_cohort_metadata,_augment_fixed_genotype_pcs,
         _prepare_irregular_followup_rows,method_covariates,_drop_missing_covariates,bfile_for_cohort)
    from gengps.methods.ctar_mptga import ContinuousTimeMPTGA
    from gengps.basis import TimeBasis
    manifest=json.loads((Path(a.figroot)/'power_analysis_CTAR_v5/power_manifest_v5.json').read_text())
    alpha=a.alpha if a.alpha is not None else float(manifest['discovery_fixed_BH_rawP_cutoff'])
    if not 0<alpha<1:raise ValueError('test alpha out of range')
    templates=pd.read_csv(Path(a.figroot)/'power_analysis_CTAR_v5/normalized_eight_coefficient_templates_v5.csv')
    x=templates[(templates.shape_id==a.shape_id)&np.isclose(templates.target_R2,a.target_r2)]
    if len(x)!=1:raise RuntimeError('No unique selected normalized template; inspect normalized_eight_coefficient_templates_v5.csv')
    q=x.iloc[0];theta=np.array([float(q[f'beta_g_{j}']) for j in range(8)])
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
    dst=out/f'validation_shape{a.shape_id}_R2{a.target_r2:.3f}_N{a.n}_maf{a.maf:.3f}_rho{a.rho:.2f}_alpha{alpha:.5g}.csv'
    rows=[]
    if dst.exists():rows=pd.read_csv(dst).to_dict('records')
    for r in range(len(rows),a.reps):
        rng=np.random.default_rng(np.random.SeedSequence([a.seed,r]).generate_state(1)[0]);inds=rng.integers(0,len(donor),size=a.n)
        gs=rng.binomial(2,a.maf,size=a.n)
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
                ar=a.rho**(t[j]-t[j-1]) if a.rho>0 else 0.0
                e[j]=ar*e[j-1]+np.sqrt(1-ar*ar)*rng.standard_normal()
            genos[pid]=int(g)
            response.update({sid:(float(x),float(y)) for sid,x,y in zip(temp.index,f+e,e)})
        meta=pd.concat(frames).sort_values(['participant_id','followup_time'],kind='mergesort')
        y=pd.DataFrame([response[sid] for sid in meta.index],index=meta.index,columns=['SIM_ALT','SIM_NULL'])
        geno=pd.DataFrame({'SIM_SNP':list(genos.values())},index=list(genos.keys()))
        fit=scanner.scan(meta,y,geno,cis,covariate_cols=covars,id_col='participant_id',time_col='followup_time')
        pv=dict(zip(fit.Ensembl_ID.astype(str),fit.Pvalue.astype(float)))
        row={'rep':r,'shape_id':a.shape_id,'target_R2':a.target_r2,'maf':a.maf,'n':a.n,'rho':a.rho,
             'p_alt':pv['SIM_ALT'],'p_null':pv['SIM_NULL'],'test_alpha':alpha,
             'n_g0':int((gs==0).sum()),'n_g1':int((gs==1).sum()),'n_g2':int((gs==2).sum())}
        rows.append(row);pd.DataFrame(rows).to_csv(dst,index=False)
        print(f'[VALIDATE] {r+1}/{a.reps} alt P={row["p_alt"]:.3g} null P={row["p_null"]:.3g}',flush=True)
    z=pd.DataFrame(rows)
    print('[SUMMARY] N=',a.n,' MAF=',a.maf,' empirical-alt P<alpha=',alpha,':', (z.p_alt<alpha).mean(),
          'null type-I at fixed alpha=',alpha,':', (z.p_null<alpha).mean(), 'null nominal P<0.05=', (z.p_null<.05).mean(),' n=',len(z),' file=',dst,flush=True)
if __name__=='__main__':main()
