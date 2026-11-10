#!/usr/bin/env python3
"""Audit two final CSF IVs against Stage1 expression and genotypes, without changing analyses."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

def joint_test(fit, indices):
    if not indices: return {'chi2':np.nan,'df':0,'p':np.nan,'wald_per_df':np.nan}
    R=np.zeros((len(indices),len(fit.params)))
    for k,i in enumerate(indices):R[k,i]=1.
    cov=np.asarray(fit.cov_params(),float)
    v=R@cov@R.T
    if np.linalg.matrix_rank(v)<len(indices):
        return {'chi2':np.nan,'df':len(indices),'p':np.nan,'wald_per_df':np.nan}
    b=R@np.asarray(fit.params,float)
    stat=float(b@np.linalg.solve(v,b))
    return {'chi2':stat,'df':len(indices),'p':float(chi2.sf(stat,len(indices))),'wald_per_df':stat/len(indices)}

def independent_columns(X,names,groups):
    indices=[];base=np.empty((X.shape[0],0)); selected=[]
    for j in range(X.shape[1]):
        if np.linalg.matrix_rank(np.column_stack([base,X[:,j]]),tol=1e-9)>len(selected):
            selected.append(j);base=np.column_stack([base,X[:,j]])
    return X[:,selected],[names[j] for j in selected],selected

def fit_one(meta,y,geno,grex,covars,cohort,gene,snp):
    from gengps.amp_pd.grex import donor_balanced_weights,evaluate_frozen_grex
    pid=meta.participant_id.astype(str).to_numpy();n_pid=len(np.unique(pid))
    age=np.asarray(meta.current_age,float); time=np.asarray(meta.followup_time,float)
    agez=(age-np.mean(age))/max(np.std(age),1e-10)
    tz=(time-np.mean(time))/max(np.std(time),1e-10)
    g=geno.copy();g.index=g.index.astype(str)
    doses=pd.to_numeric(g.reindex(meta.participant_id.astype(str))[snp],errors='coerce').to_numpy(float)
    # Do not invent dosage or impute genotype here. Stage-II genotype source may have imputed raw missing values earlier.
    good=np.isfinite(y)&np.isfinite(doses)&np.isfinite(grex)&np.isfinite(age)&np.isfinite(time)
    for c in covars:good&=np.isfinite(pd.to_numeric(meta[c],errors='coerce').to_numpy(float))
    meta=meta.iloc[np.where(good)[0]].copy();y=np.asarray(y)[good];pid=pid[good];doses=doses[good];grex=np.asarray(grex)[good];agez=agez[good];tz=tz[good]
    if len(np.unique(pid))<15:raise ValueError('Too few participants')
    nuisance=[np.ones(len(y)),agez,agez**2,agez**3];names=['intercept','age','age2','age3']
    for c in covars:nuisance.append(pd.to_numeric(meta[c],errors='coerce').to_numpy(float));names.append(c)
    C=np.column_stack(nuisance);C,cn,_=independent_columns(C,names,None)
    # The complete additive genetic component is g*(1, t, t^2, t^3).
    # NO categorical genotype expansion. Test all four genetic terms jointly.
    if np.nanstd(doses) < 1e-12:
        raise ValueError(f'{cohort}/{gene}/{snp}: genotype dosage is constant')
    gbasis=np.column_stack([doses*(tz**k) for k in range(4)])
    gnames=['dose_t0','dose_t1','dose_t2','dose_t3']
    Xfull=np.column_stack([C,gbasis])
    rank=int(np.linalg.matrix_rank(Xfull))
    condition=float(np.linalg.cond(Xfull))
    weights=donor_balanced_weights(pid)
    base_fit=sm.WLS(y,C,weights=weights).fit()
    full_fit=sm.WLS(y,Xfull,weights=weights).fit()
    rss_reduced=float(np.sum(weights*(y-base_fit.fittedvalues)**2))
    rss_full=float(np.sum(weights*(y-full_fit.fittedvalues)**2))
    dfr=int(len(y)-rank)
    partial_r2=float(max(0.,(rss_reduced-rss_full)/rss_reduced)) if rss_reduced>0 else np.nan
    nested_f=float(((rss_reduced-rss_full)/4)/(rss_full/dfr)) if rss_full>0 and dfr>0 else np.nan
    nested_p=float(__import__('scipy').stats.f.sf(nested_f,4,dfr)) if np.isfinite(nested_f) and nested_f>=0 else np.nan
    error=''
    if rank!=Xfull.shape[1]:
        error=f'Full genetic design rank deficient ({rank}/{Xfull.shape[1]})'
        genetic={'chi2':np.nan,'df':4,'p':np.nan,'wald_per_df':np.nan}
    else:
        fit=sm.WLS(y,Xfull,weights=weights).fit(cov_type='cluster',cov_kwds={'groups':pid})
        genetic=joint_test(fit,list(range(C.shape[1],C.shape[1]+4)))
        if not np.isfinite(genetic['chi2']):error='Genetic robust covariance singular; Wald undefined'
    records=[]
    records.append({'cohort':cohort,'gene':gene,'snp':snp,'test':'full_genetic_t0_t1_t2_t3',
        'n_samples':len(y),'n_participants':len(np.unique(pid)),
        'genotype_classes':str(dict(pd.Series(doses).round(3).value_counts().sort_index())),
        'n_genotype_channels':1,'chi2':genetic['chi2'],'df':genetic['df'],
        'p':genetic['p'],'wald_per_df_NOT_first_stage_F':genetic['wald_per_df'],
        'partial_R2_genetic_4_terms':partial_r2,
        'nested_OLS_F_4terms_NOT_IV_F':nested_f,'nested_OLS_p':nested_p,
        'design_rank':rank,'design_columns':Xfull.shape[1],
        'design_condition_number':condition,'warning':error})
    # Frozen score: exactly evaluate_frozen_grex() using stage2 covariates
    metric,yres,gres,w=evaluate_frozen_grex(meta,y,grex,covars)
    records.append({'cohort':cohort,'gene':gene,'snp':snp,'test':'frozen_grex_score_stage2_definition','n_samples':len(y),'n_participants':len(np.unique(pid)),'genotype_classes':str(dict(pd.Series(doses).round(3).value_counts().sort_index())),'n_genotype_channels':1, 'chi2':metric.get('grex_score_F',np.nan),'df':1,'p':metric.get('grex_score_p',np.nan),'wald_per_df_NOT_first_stage_F':metric.get('grex_score_F',np.nan),'pearson_sq':metric.get('pearson',np.nan)**2,'frozen_R2':metric.get('R2_frozen',np.nan),'warning':'PPMI training-apparent score; not independent validation' if cohort=='PPMI' else ''})
    return records

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--code',required=True);ap.add_argument('--out-dir',required=True)
    a=ap.parse_args();root=Path(a.root);sys.path.insert(0,str(Path(a.code)))
    from gengps.amp_pd.config import AmpPdPaths
    from gengps.amp_pd.real_qtl import resolve_expression_gene_columns
    from gengps.amp_pd.stage2_strength import init_worker,_prepare_batch_context,_gene_inputs
    from gengps.amp_pd.grex import legacy_single_iv_from_qtl_row
    inp=root/'single_iv_causal/iv_validation/single_iv_primary_eligible.csv'
    if not inp.exists():raise FileNotFoundError(inp)
    cand=pd.read_csv(inp,low_memory=False)
    targets=[('AC084117.1','rs16935599'),('HDLBP','rs141463303')]
    tasks=[]
    expr_map=resolve_expression_gene_columns(AmpPdPaths(root=str(root)),[g for g,s in targets],allow_unresolved=False)
    for gene,snp in targets:
        z=cand.loc[(cand.gene.astype(str)==gene)&(cand.lead_snp.astype(str)==snp)]
        if len(z)!=1:raise RuntimeError(f'{gene}/{snp}: expected one Stage1 frozen row, got {len(z)}')
        row=z.iloc[0].to_dict()
        tasks.append({'gene':gene,'expr_gene':expr_map[gene],'snps':[snp],'legacy_row':{**row,'Ensembl_ID':gene,'Variant_ID':snp,'Pvalue':row.get('ppmi_tqtl_p',np.nan)}})
    out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    init_worker({'root':str(root),'out_dir':str(out),'qtl_label':'submitted_followup_ctar_mptga','blas_threads':1})
    ctx=_prepare_batch_context(tasks)
    records=[]
    for task in tasks:
        model=legacy_single_iv_from_qtl_row(task['legacy_row'])
        for cohort,label in [('PP','PPMI'),('PD','PDBP')]:
            meta,y,cov,geno=_gene_inputs(task,cohort,ctx)
            grex=model.predict_grex(geno,meta)
            records.extend(fit_one(meta,y,geno,grex,cov,label,task['gene'],task['snps'][0]))
    pd.DataFrame(records).to_csv(out/'iv_full_genetic_t0_t3_and_frozen_score.csv',index=False)
    (out/'method_notes.txt').write_text('Complete additive genetic component: dosage x (1, centered-scaled-followup, squared, cubed). Test all four coefficients jointly using participant-cluster-robust Wald chi-square (df=4). Additional nested weighted least-squares partial R2 and descriptive classical F compare this model to the same non-genetic covariates. This is NOT the original Stage1 ctAR model, and these statistics do not establish valid causal IV strength. Frozen score reproduces Stage2 metric; PPMI in-training assessment.\n')
    print(pd.DataFrame(records)[['cohort','gene','snp','test','chi2','df','p','pearson_sq' if 'pearson_sq' in pd.DataFrame(records) else 'warning']].to_string(index=False))
    print('OUTPUT:',out/'iv_full_genetic_t0_t3_and_frozen_score.csv')
if __name__=='__main__':main()
