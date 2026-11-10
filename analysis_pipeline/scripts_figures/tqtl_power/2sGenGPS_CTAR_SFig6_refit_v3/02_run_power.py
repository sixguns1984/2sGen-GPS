#!/usr/bin/env python3
"""Actual CTAR 8-df donor-bootstrap refitting, PPMI and PDBP separately.

Each locus is checked against its Stage-I overall test; raw allele frequency
comes from PLINK, never fabricated. Results are written incrementally and
resume using --resume; failed loci are included in the failure audit.
"""
from __future__ import annotations
import argparse,json,traceback,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2
from ctar_refit import summaries_per_donor,fit_from_sums,fit_profile,donor_bootstrap,exact_rho_grid
from ctar_refit_fast import donor_bootstrap_fast
from production_adapter import ProductionMetadata,ProductionBedGenotypes,ProductionExpression,build_production_locus

def evaluate(data,cohort,rho,alpha,reps,seed,profile=False,grid_size=15,subsample=(0.25,0.5,0.75,0.9,1.),sens_reps=60,backend="fast",batch_size=32):
    if pd.isna(rho):raise ValueError('Stage-I fitted rho is unavailable. Refusing an invented default; supply actual cohort CTAR rho')
    r=float(rho)
    if not 0<=r<1:raise ValueError('Invalid fitted Stage-I rho: '+str(rho))
    grid=exact_rho_grid(r,grid_size) if profile else [r]
    sums=[summaries_per_donor(data['time'],data['donor'],data['genotype'],data['y'],data['cov'],rho=x) for x in grid]
    actual=fit_profile(sums) if profile else fit_from_sums(sums[0])
    if not np.isfinite(actual['lrt']):raise ValueError('GLS singular or invalid residual variance')
    t_start=time.perf_counter()
    bootstrap_fn=donor_bootstrap_fast if backend=='fast' and not profile else donor_bootstrap
    fast_args={'batch_size':batch_size} if bootstrap_fn is donor_bootstrap_fast else {}
    boot=bootstrap_fn(sums,reps,seed,alpha=alpha,save_lrt=True,**fast_args)
    if boot['valid']<max(30,int(.9*reps)):
        raise ValueError(f'Only {boot["valid"]}/{reps} valid bootstrap LRTs; genotype imbalance or rank deficiency')
    result={'cohort':cohort,'n_donors':data['n_donors'],'n_obs':data['n_obs'],
            'maf':data['maf'],'genotype_counts':':'.join(map(str,data['g_counts'])),
            'n_covariates_used':data['n_covariates'],'rho_fixed':r,'profile_rho':profile,
            'refit_lrt':actual['lrt'],'refit_pvalue':actual['p'],'alpha_test':alpha,
            'bootstrap_power':boot['bootstrap_detection'],
            'bootstrap_success':boot['success'],'bootstrap_valid':boot['valid'],
            'bootstrap_total':reps}
    hist=pd.DataFrame({'lrt':boot['lrt'],'pvalue':boot['p']})
    subset=[]
    if sens_reps:
        for f in subsample:
            # donor sampling for different *real* cohort sizes (not multiplying lambda)
            bs=bootstrap_fn(sums,sens_reps,seed+int(f*10000),fraction=f,
                               replace=False,alpha=alpha,save_lrt=False,**fast_args)
            subset.append({'cohort':cohort,'fraction':f,'n_donors':int(round(data['n_donors']*f)),
                           'n_boot':bs['valid'],'frac_pass':bs['bootstrap_detection']})
    result['elapsed_bootstrap_sec']=round(time.perf_counter()-t_start,3)
    result['bootstrap_backend']='batch_blas_cholesky' if bootstrap_fn is donor_bootstrap_fast else 'legacy_pinv'
    return result,hist,subset

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    a.add_argument('--outdir');a.add_argument('--selected-file');a.add_argument('--metadata');a.add_argument('--expression')
    a.add_argument('--cov-file',action='append',default=[])
    a.add_argument('--cov-cols',default='')
    a.add_argument('--cohort-col');a.add_argument('--participant-col');a.add_argument('--sample-col');a.add_argument('--time-col')
    a.add_argument('--pp-raw');a.add_argument('--pd-raw')
    a.add_argument('--alpha-disc',type=float,default=None)
    a.add_argument('--alpha-rep',type=float,default=.05)
    a.add_argument('--bootstrap-reps',type=int,default=250)
    a.add_argument('--sensitivity-reps',type=int,default=0)
    a.add_argument('--sensitivity-pairs',type=int,default=250)
    a.add_argument('--profile-rho',action='store_true')
    a.add_argument('--rho-points',type=int,default=15)
    a.add_argument('--max-pairs',type=int,default=0)
    a.add_argument('--min-donors',type=int,default=80)
    a.add_argument('--audit-pairs',type=int,default=6,help='Number of loci per cohort with exact 100-rho profile-LRT audit')
    a.add_argument('--min-genotype-group',type=int,default=5)
    a.add_argument('--allow-covariate-mismatch',action='store_true',help='Deprecated: production covariates always required')
    a.add_argument('--backend',choices=['fast','legacy'],default='fast')
    a.add_argument('--batch-size',type=int,default=32)
    a.add_argument('--shard-id',type=int,default=0)
    a.add_argument('--num-shards',type=int,default=1)
    a.add_argument('--resume',action='store_true');a.add_argument('--seed',type=int,default=1173)
    x=a.parse_args();root=Path(x.root)
    if x.profile_rho and x.bootstrap_reps>200 and x.max_pairs==0:
        print('[WARNING] Profiling 100 rho values in EVERY bootstrap may take extremely long; fixed-rho bootstrap + exact original-LRT audit is recommended.',flush=True)
    out=Path(x.outdir or root/'manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_refit_v3')
    out.mkdir(parents=True,exist_ok=True)
    selected=Path(x.selected_file or out/'selected_loci.csv')
    df=pd.read_csv(selected)
    if x.max_pairs:df=df.head(x.max_pairs)
    if x.num_shards<1 or not 0<=x.shard_id<x.num_shards:raise ValueError('Invalid --shard-id/--num-shards')
    df=df.reset_index(drop=True)
    df['_global_index']=np.arange(len(df))
    if x.num_shards>1:
        df=df.iloc[x.shard_id::x.num_shards].copy()
        if df.empty:raise ValueError('More workers than selected loci')
    if x.cov_file or x.cov_cols or x.metadata or x.time_col or x.sample_col or x.participant_col:
        raise ValueError('v4 uses exact Stage-I production metadata/covariate helpers; manual overrides are disabled to prevent model drift.')
    md={co:ProductionMetadata(root,co) for co in ('PP','PD')}
    gt={co:ProductionBedGenotypes(root,co,md[co].data.participant_id,df.variant.astype(str).unique())
        for co in ('PP','PD')}
    ex=ProductionExpression(root,df.gene.astype(str).unique(),cache_file=x.expression)
    print('[STAGE1 ALIGNMENT]',{co:md[co].summary for co in md},flush=True)
    print('[GENOTYPE MATCH]',{co:gt[co].coverage for co in gt},flush=True)
    if ex.unknown: print('[WARN] Unresolved Ensembl genes:',list(sorted(ex.unknown))[:20], 'total=',len(ex.unknown),flush=True)
    # The true genomewide BH raw cutoff is derived from observed PPMI
    # discoveries. Requires alpha-disc <= max among all BH-significant hits.
    audit=json.loads((out/'selection_audit.json').read_text()) if (out/'selection_audit.json').exists() else {}
    alpha_disc=x.alpha_disc or float(audit.get('observed_max_raw_p_of_all_hits',np.nan))
    if not 0<alpha_disc<1:raise ValueError('Missing actual BH raw P cutoff; provide --alpha-disc from finalized PPMI Stage-I')
    status=out/'locus_bootstrap_results.csv'
    if x.resume and status.exists():
        saved=pd.read_csv(status)
        finished=set(zip(saved['gene'].astype(str),saved['variant'].astype(str),saved.cohort.astype(str)))
    else:
        if status.exists(): raise FileExistsError(f'{status} already exists. Use --resume or choose new output dir')
        finished=set()
    failpath=out/'locus_failures.csv';failure_list=[]
    diag=out/'input_model_audit.json'
    metadata_audit={co:md[co].summary for co in md}
    diag.write_text(json.dumps({'metadata':metadata_audit,'expression_mode':getattr(ex,'mode','wide_symbol_mapped'),
                                'actual_discovery_raw_cutoff':alpha_disc,'PD_nominal_alpha':x.alpha_rep,
                                'test':'Overall 8df CTAR genetic block ONLY',
                                'rho':'Original Stage-I fitted rho held fixed for primary bootstrap' if not x.profile_rho else 'rho profiled independently under H0/H1 over finite grid',
                                'power_estimate':'Observed-participant cluster-bootstrap redetection (conditional resampling), not unbiased prospective true power'},indent=2))
    (out/'bootstrap_LRT_examples').mkdir(exist_ok=True)
    with status.open('a') as handle:
        for i,record in enumerate(df.itertuples(index=False)):
            gene=str(record.gene);variant=str(record.variant)
            for co in ('PP','PD'):
                if (gene,variant,co) in finished:continue
                if co=='PD' and getattr(record,'rep_match','both')!='both':
                    failure_list.append({'gene':gene,'variant':variant,'cohort':co,'reason':'Unmatched PDBP SNP-gene test'})
                    continue
                try:
                    data=build_production_locus(md[co],gt[co],ex,gene,variant,min_donors=x.min_donors,
                                     min_minor_group=x.min_genotype_group)
                    expected_n=(int(getattr(record,'n_covariates_used',-1)) if pd.notna(getattr(record,'n_covariates_used',np.nan)) else -1) if co=='PP' else -1
                    if expected_n>0 and expected_n!=data['n_covariates'] and not x.allow_covariate_mismatch:
                        raise ValueError(f'Covariate count mismatch: Stage-I n_covariates_used={expected_n}, loader={data["n_covariates"]}. Supply --cov-file/--cov-cols or carefully audit.')
                    rho=getattr(record,'rho',np.nan) if co=='PP' else getattr(record,'pd_rho',np.nan)
                    alpha=alpha_disc if co=='PP' else x.alpha_rep
                    gi=int(record._global_index) if hasattr(record,'_global_index') else int(df.iloc[i]['_global_index'])
                    sens=x.sensitivity_reps if gi<x.sensitivity_pairs else 0
                    result,hist,sensitivity=evaluate(data,co,rho,alpha,x.bootstrap_reps,
                                x.seed+2*gi+int(co=='PD'),x.profile_rho,x.rho_points,sens_reps=sens,
                                backend=x.backend,batch_size=x.batch_size)
                    original=(getattr(record,'lrt',np.nan) if co=='PP' else getattr(record,'pd_lrt',np.nan))
                    # Stage-I stored rho is FULL-model optimum. Its null model
                    # profiles rho separately. Use the exact 100 point production
                    # grid for a small audit subset, independent of fixed-rho bootstrap.
                    result['fixed_rho_lrt']=result['refit_lrt']
                    if gi < x.audit_pairs:
                        exact=[summaries_per_donor(data['time'],data['donor'],data['genotype'],data['y'],data['cov'],rho=float(k)/100.)
                               for k in range(100)]
                        audit_fit=fit_profile(exact)
                        result['refit_lrt']=audit_fit['lrt']
                    else:
                        result['refit_lrt']=np.nan
                    result['stage1_lrt']=np.nan
                    result['lrt_abs_diff']=np.nan
                    if pd.notna(original):
                        result['stage1_lrt']=float(original)
                        if np.isfinite(result['refit_lrt']):
                            result['lrt_abs_diff']=abs(result['refit_lrt']-float(original))
                    result.update({'gene':gene,'variant':variant,'stage1_p':getattr(record,'pvalue',np.nan) if co=='PP' else getattr(record,'pd_pvalue',np.nan)})
                    # Keep small sample of full LRT distributions; all full refit results kept in CSV.
                    if gi<20 or (.76<=result['bootstrap_power']<=.84) or (.87<=result['bootstrap_power']<=.93):
                        hist.assign(gene=gene,variant=variant,cohort=co).to_csv(out/'bootstrap_LRT_examples'/f'{gi:04d}_{co}.csv',index=False)
                    if sensitivity:
                        senspath=out/'donor_sample_size_sensitivity.csv'
                        pd.DataFrame([{**v,'gene':gene,'variant':variant} for v in sensitivity]).to_csv(
                            senspath,index=False,header=not senspath.exists(),mode='a')
                    pd.DataFrame([result]).to_csv(handle,index=False,header=(handle.tell()==0))
                    handle.flush()
                    finished.add((gene,variant,co))
                except Exception as exc:
                    failure_list.append({'gene':gene,'variant':variant,'cohort':co,'reason':str(exc)[:800]})
            if (i+1)%25==0:
                print(f'[PROGRESS] {i+1}/{len(df)} pairs; {len(finished)} cohort fits; {len(failure_list)} failures',flush=True)
                if failure_list:pd.DataFrame(failure_list).to_csv(failpath,index=False)
    if failure_list:pd.DataFrame(failure_list).to_csv(failpath,index=False)
    if not status.exists() or status.stat().st_size==0:
        from collections import Counter
        ctr=Counter(z['reason'] for z in failure_list)
        print('[NO FITS] Top errors:',ctr.most_common(12),flush=True)
        raise SystemExit('Zero valid Stage-I-aligned CTAR fits; see '+str(failpath))
    print(f'[FINISH] {status}; failures={len(failure_list)}. Verify model audit before using results.',flush=True)
if __name__=='__main__':main()
