"""Public-facing small-data interface to the manuscript's original CTAR and ARX engines.

The publication workflow for genome-wide datasets is under analysis_pipeline/.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from .methods.ctar_mptga import ContinuousTimeMPTGA
from .utils import bh_fdr
from .amp_pd.single_iv_causal import fit_all_lags
from .genotypes import load_genotypes


def table(path):
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(p)
    if p.suffix.lower()=='.parquet': return pd.read_parquet(p)
    return pd.read_csv(p)


def save(frame,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if p.suffix.lower()=='.parquet':frame.to_parquet(p,index=False)
    else:frame.to_csv(p,index=False)
    print(f'[saved] {p} ({len(frame)} rows)')


def cov_expression(cov,expr,id_col,time_col):
    for c in [id_col,time_col]:
        if c not in cov:raise ValueError(f'Covariates missing {c}')
    if 'sample_id' not in cov or 'sample_id' not in expr:raise ValueError('Both covariates and expression must contain sample_id')
    if cov.sample_id.duplicated().any() or expr.sample_id.duplicated().any():raise ValueError('Duplicate sample_id')
    if not expr.sample_id.astype(str).isin(cov.sample_id.astype(str)).all():raise ValueError('Expression has unmatched sample_ids')
    c=cov.set_index('sample_id',drop=False);e=expr.set_index('sample_id').reindex(c.index)
    if e.isna().all(axis=1).any():raise ValueError('Missing expression rows for some covariate sample_ids')
    e=e.apply(pd.to_numeric,errors='raise')
    if e.isna().any().any():raise ValueError('Missing expression values; prepare a complete analysis matrix')
    return c,e


def bootstrap_redetection(cov,expr,geno,cis,kwargs,reps,seed,alpha,id_col,time_col,covar_cols):
    """Donor-level empirical redetection against a fixed nominal threshold.

    Note: a bootstrap-only fraction is not a formal IV strength statistic.
    """
    rng=np.random.default_rng(seed)
    ids=cov[id_col].astype(str).unique()
    hits={tuple(x):0 for x in cis[['geneid','snpid']].astype(str).itertuples(index=False,name=None)}
    valid={k:0 for k in hits}
    for b in range(reps):
        selected=rng.choice(ids,size=len(ids),replace=True)
        rows=[];genorows=[]
        for j,pid in enumerate(selected):
            z=cov.loc[cov[id_col].astype(str).eq(pid)].copy()
            new=f'bootstrap_{b}_{j}'
            z[id_col]=new; z.index=[f'{new}_row_{k}' for k in range(len(z))]
            rows.append(z)
            gr=geno.loc[[str(pid)]].copy();gr.index=[new];genorows.append(gr)
        cb=pd.concat(rows); eb=expr.loc[[s for pid in selected for s in cov.index[cov[id_col].astype(str).eq(pid)]]].copy()
        eb.index=cb.index
        gb=pd.concat(genorows)
        try:
            scan=ContinuousTimeMPTGA(**kwargs).scan(cb,eb,gb,cis,covariate_cols=covar_cols,id_col=id_col,time_col=time_col)
            for r in scan.itertuples(index=False):
                key=(str(r.Ensembl_ID),str(r.Variant_ID))
                if key in hits and np.isfinite(r.Pvalue):
                    valid[key]+=1
                    hits[key]+=int(float(r.Pvalue)<alpha)
        except (ValueError,RuntimeError,np.linalg.LinAlgError) as ex:
            print(f'[bootstrap] replicate {b+1} failed: {ex}',file=sys.stderr)
        if (b+1)%max(1,reps//10)==0:print(f'[bootstrap] {b+1}/{reps}',flush=True)
    return pd.DataFrame([{'Ensembl_ID':k[0],'Variant_ID':k[1],
        'bootstrap_n_valid':valid[k], 'bootstrap_n_redetected':hits[k],
        'bootstrap_redetection_fraction':hits[k]/valid[k] if valid[k] else np.nan}
        for k in hits])


def cmd_ctar(a):
    cov,expr=cov_expression(table(a.covariates),table(a.expression),a.id_col,a.time_col)
    cis=table(a.cis_map)
    need={'geneid','snpid'}
    if not need.issubset(cis.columns):raise ValueError(f'Cis map requires {need}')
    cis=cis[['geneid','snpid']].drop_duplicates().astype(str)
    geno=load_genotypes(a.genotypes,cis,cov[a.id_col].astype(str).unique(),id_col=a.id_col,plink_sample_id=a.plink_sample_id)
    cols=[z for z in a.covariates_cols.split(',') if z]
    kwargs=dict(basis_kind='cubic',genotype_mode=a.genotype_mode,rho_points=a.rho_points,
        reference_interval=a.reference_interval,rho_max=a.rho_max,device=a.device,batch_size=a.batch_size)
    z=ContinuousTimeMPTGA(**kwargs).scan(cov,expr,geno,cis,covariate_cols=cols,
        id_col=a.id_col,time_col=a.time_col)
    if z.empty: raise RuntimeError('No SNP-gene pairs analyzed; check matching gene and variant identifiers')
    # Recompute over FULL evaluated test family rather than scanner's diagnostic chunk q-values.
    z['FDR']=bh_fdr(z['Pvalue'].to_numpy(float));z['q_bh_global']=z['FDR']
    z['significant_overall']=z['FDR']<a.alpha
    z=z.drop(columns=['Interaction_FDR'],errors='ignore')
    save(z,a.output)
    if a.bootstrap_reps:
        b=bootstrap_redetection(cov,expr,geno,cis,kwargs,a.bootstrap_reps,a.seed,a.alpha,a.id_col,a.time_col,cols)
        save(b,a.bootstrap_output or str(Path(a.output).with_name('bootstrap_redetection.csv')))
        print('[notice] Bootstrap significance uses nominal threshold; specify cohort-specific cutoff if matching manuscript screening.')
    print(json.dumps({'tested_pairs':len(z),'significant_overall':int(z.significant_overall.sum()),'alpha':a.alpha},indent=2))


def cmd_stage2(a):
    x=table(a.input)
    for col in [a.id_col,'time',a.outcome,a.exposure]:
        if col not in x:raise ValueError(f'Input lacks {col}')
    if a.id_col!='participant_id':x=x.rename(columns={a.id_col:'participant_id'})
    # Explicit pair IDs avoid accidentally mixing different predictions for the same participant.
    keys=[z for z in ['gene','lead_snp'] if z in x.columns]
    if not keys:keys=['__single_pair'];x['__single_pair']='all'
    rows=[]
    for group,z in x.groupby(keys,sort=False):
        result=fit_all_lags(z,outcome_col=a.outcome,exposure_col=a.exposure,max_lag=a.max_lag,
            baseline_covariates=[p for p in a.covariates_cols.split(',') if p],
            auto_clinical_covariates=not a.no_clinical,min_participants=a.min_participants,
            adjust_current_age_cubic=a.age_cubic,compute_ar=a.reduced_form,
            compute_cluster_details=True)
        if not isinstance(group,tuple):group=(group,)
        for k,v in zip(keys,group):result[k]=v
        rows.append(result)
    z=pd.concat(rows,ignore_index=True)
    z['q_bh_global']=bh_fdr(z['pvalue'].to_numpy(float))
    z['significant']=z['ok'].fillna(False)&(z['q_bh_global']<a.alpha)
    z['SC']=z['bic']
    save(z,a.output)
    if a.evidence:
        # 'compute_ar=True' runs the source code's genotype-derived reduced-form analysis.
        # Full cross-cohort A/B/C grading additionally needs independent replication,
        # multiple independent IVs, and cross-gene LD and is provided by pipeline scripts.
        if 'iv_dosage' not in x:raise ValueError('--evidence needs iv_dosage in the input')
        out=z[[c for c in ['gene','lead_snp','lag_order','ar_pvalue','ar_F','ar_df1','ar_df2','q_bh_global','coef_sum'] if c in z]].copy()
        out['reduced_form_support_nominal']=out['ar_pvalue']<0.05
        save(out,str(Path(a.output).with_name('stage2_reduced_form_evidence.csv')))
        print('[notice] Full A/B/C tiers require cohort-specific robustness and LD inputs; see gps evidence and analysis_pipeline.')
    print(json.dumps({'fitted_lags':len(z),'significant':int(z.significant.sum())},indent=2))


def cmd_evidence(a):
    """Use exactly the manuscript's existing post-selection tier implementation."""
    from .amp_pd.evidence import add_evidence_annotations
    x=table(a.input)
    # Strict required fields prevent silent fabrication of grade A from absent tests.
    needed={'gene','lead_snp','coef_sum_direction_concordant','robust_PPMI_ar_pvalue',
        'robust_PD_PLUS_SU_ar_pvalue','multiIV_p','multiIV_Q_p','multiIV_n',
        'cross_gene_ld_flag','LOO_all_same_direction','LOO_all_nominal_p05'}
    missing=needed-set(x.columns)
    if missing:raise ValueError('Evidence table must contain actual computed robustness inputs: '+', '.join(sorted(missing)))
    z=add_evidence_annotations(x)
    save(z,a.output)
    print(z.descriptive_evidence_tier.value_counts().to_string())


def main(argv=None):
    p=argparse.ArgumentParser(prog='2sgengps',description='2sGen-GPS: temporal cis-eQTL CTAR mapping and Stage-II cumulative ARX')
    sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('ctar',help='Stage-I CTAR-MPTGA (CSV or PLINK BED) and global BH FDR')
    for name in ['covariates','expression','genotypes','cis-map','output']:
        c.add_argument('--'+name,required=True,help=('CSV/Parquet genotype matrix, PLINK .bed or PLINK prefix' if name=='genotypes' else None))
    c.add_argument('--covariates-cols',default='',help='Comma-separated covariates, e.g. sex,PC1,PC2')
    c.add_argument('--id-col',default='participant_id');c.add_argument('--time-col',default='followup_time')
    c.add_argument('--plink-sample-id',choices=['iid','fid_iid'],default='iid',help='BED/FAM participant matching: IID or FID:IID')
    c.add_argument('--genotype-mode',choices=['binary','dummies','dosage'],default='binary')
    c.add_argument('--rho-points',type=int,default=100);c.add_argument('--rho-max',type=float,default=.99)
    c.add_argument('--reference-interval',type=float,default=1.0);c.add_argument('--batch-size',type=int,default=256)
    c.add_argument('--device',default='cpu');c.add_argument('--alpha',type=float,default=.05)
    c.add_argument('--bootstrap-reps',type=int,default=0);c.add_argument('--bootstrap-output');c.add_argument('--seed',type=int,default=1)
    c.set_defaults(func=cmd_ctar)
    s=sub.add_parser('stage2',help='Stage-II cumulative ARX and global BH FDR')
    s.add_argument('--input',required=True);s.add_argument('--output',required=True)
    s.add_argument('--id-col',default='participant_id');s.add_argument('--outcome',required=True)
    s.add_argument('--exposure',default='grex_single_iv');s.add_argument('--covariates-cols',default='')
    s.add_argument('--max-lag',type=int,default=4);s.add_argument('--min-participants',type=int,default=20)
    s.add_argument('--age-cubic',action='store_true');s.add_argument('--no-clinical',action='store_true')
    s.add_argument('--reduced-form',action='store_true',help='Also evaluate genotype-derived reduced-form predictor block')
    s.add_argument('--evidence',action='store_true',help='Write reduced-form evidence table; requires iv_dosage')
    s.add_argument('--alpha',type=float,default=.05);s.set_defaults(func=cmd_stage2)
    e=sub.add_parser('evidence',help='Assign manuscript-style descriptive tiers from precomputed robustness table')
    e.add_argument('--input',required=True);e.add_argument('--output',required=True);e.set_defaults(func=cmd_evidence)
    a=p.parse_args(argv)
    if getattr(a,'evidence',False):a.reduced_form=True
    try:a.func(a)
    except Exception as ex:
        p.exit(2,f'ERROR: {ex}\n')

if __name__=='__main__':main()
