#!/usr/bin/env python3
"""MRI DBM post-selection genotype reduced-form diagnostic and clinical phenotype annotation.
Uses MRI-visit frozen model predictions; never substitutes UPDRS GReX or old MRI SNP columns.
"""
import argparse,json,re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
ROOT='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt'

def norm_id(x):
    return x.astype('string').str.strip().str.replace(r'^PP-','',regex=True).str.replace(r'\.0$','',regex=True)

def fit(X,y,group):
    n,k=X.shape
    if n<=k+3 or np.linalg.matrix_rank(X)<k:return None
    b=np.linalg.lstsq(X,y,rcond=None)[0];e=y-X@b;uniq,idx=np.unique(group,return_inverse=True);g=len(uniq)
    if g<3:return None
    scores=np.zeros((g,k));np.add.at(scores,idx,X*e[:,None]);v=np.linalg.pinv(X.T@X)@(scores.T@scores)@np.linalg.pinv(X.T@X)*(g/(g-1))*((n-1)/(n-k))
    return b,e,v

def partial_r2(y,X0,X1):
    b0=np.linalg.lstsq(X0,y,rcond=None)[0];b1=np.linalg.lstsq(X1,y,rcond=None)[0]
    ss0=float(np.sum((y-X0@b0)**2));ss1=float(np.sum((y-X1@b1)**2))
    return max(0.,1.-ss1/ss0) if ss0>0 else np.nan

def reduced_form(y,conf,geno,t,group,max_degree):
    """Joint cluster Wald on SNP dosage x [1,t,t²,t³], not classical AR inference."""
    tc=t-np.nanmean(t);scale=np.nanstd(tc);tc=tc/scale if scale>1.e-10 else tc*0
    Z=np.column_stack([geno*tc**j for j in range(max_degree+1)])
    X=np.column_stack([conf,Z]);k0=conf.shape[1];n,k=X.shape
    if np.linalg.matrix_rank(X)<k:return None
    result=fit(X,y,group)
    if result is None:return None
    b,e,v=result;sub=v[k0:,k0:];rank=int(np.linalg.matrix_rank(sub))
    if rank==0:return None
    w=float(b[k0:]@np.linalg.pinv(sub)@b[k0:]);p=float(chi2.sf(max(w,0),rank))
    return {'iv_reduced_form_wald_chi2':w,'iv_reduced_form_df':rank,'iv_reduced_form_p':p,'iv_reduced_form_partial_r2':partial_r2(y,conf,X),'iv_reduced_form_n_parameters':Z.shape[1]}

def clinical_annotation(clinical,root):
    keys=['gene','lead_snp'];phenotypes=['updrs1','updrs2','updrs3','moca']
    if clinical:
        path=Path(clinical)
        if not path.exists():raise FileNotFoundError(f'Explicit --clinical-master not found: {path}')
        t=pd.read_csv(path,low_memory=False)
        if not set(keys).issubset(t):raise ValueError(f'Clinical master lacks {keys}: {path}')
        if t.duplicated(keys).any():raise ValueError('Explicit clinical master has duplicate gene/lead_snp keys')
        return t
    from functools import reduce
    parts=[]
    for ph in phenotypes:
        folder=root/'single_iv_causal'/ph/'standard_final'
        candidates=[folder/f'{ph}_FINAL_causal_IV_gene_master_FULL.csv',folder/f'{ph}_FINAL_causal_IV_gene_master.csv',folder/f'{ph}_downstream_IV_gene_master.csv']
        path=next((x for x in candidates if x.is_file()),None)
        if path is None:raise FileNotFoundError(f'No clinical result for {ph}. Checked: '+', '.join(map(str,candidates)))
        t=pd.read_csv(path,low_memory=False)
        if not set(keys).issubset(t.columns):raise ValueError(f'Clinical file missing identifiers: {path}')
        t=t.dropna(subset=keys)
        if t.duplicated(keys).any():raise ValueError(f'Duplicate IV-gene rows in {path}; refusing ambiguous clinical annotations')
        t[f'significant_{ph}']=True
        renamed={c:f'{ph}_{c}' for c in t if c not in keys and c!=f'significant_{ph}'}
        t=t.rename(columns=renamed)
        parts.append(t)
        print('[CLINICAL]',ph,len(t),'IV-gene pairs:',path,flush=True)
    out=reduce(lambda x,y:x.merge(y,on=keys,how='outer',validate='one_to_one'),parts)
    for ph in phenotypes:out[f'significant_{ph}']=out[f'significant_{ph}'].fillna(False).astype(bool)
    out['phenotypes_significant']=out.apply(lambda r:';'.join(ph for ph in phenotypes if r[f'significant_{ph}']),axis=1)
    out['n_significant_phenotypes']=out[[f'significant_{ph}' for ph in phenotypes]].sum(axis=1)
    print('[CLINICAL] union',len(out),'IV-gene pairs',flush=True)
    return out

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--root',default=ROOT);a.add_argument('--mri-file',default='/public2/data/luojunfeng/2sTGCS_binary/outcome/DBM/confound_pass_snp/MRI_DBM.csv')
    a.add_argument('--mri-all',default='');a.add_argument('--mri-fdr',default='');a.add_argument('--clinical-master',default='');a.add_argument('--output-dir',default='')
    a.add_argument('--component',type=int,default=1);a.add_argument('--min-participants',type=int,default=30);a.add_argument('--max-ar-degree',type=int,default=3)
    a.add_argument('--strict-hit-match',action='store_true',default=True)
    args=a.parse_args();root=Path(args.root);base=root/'single_iv_causal/mri_dbm_lag1'/f'pc{args.component}'
    ma=Path(args.mri_all) if args.mri_all else base/'MRI_DBM_lag1_ALL_associations.csv'
    mf=Path(args.mri_fdr) if args.mri_fdr else base/'MRI_DBM_lag1_global_FDR05.csv'
    clinical=args.clinical_master
    out=Path(args.output_dir) if args.output_dir else base/'publication_robustness';out.mkdir(parents=True,exist_ok=True)
    hits=pd.read_csv(mf,low_memory=False);alltab=pd.read_csv(ma,low_memory=False)
    if not {'gene','lead_snp','artifact_id','roi'}.issubset(hits):raise ValueError('MRI FDR data lacks identifiers')
    annotation=clinical_annotation(clinical,root)
    keys=['gene','lead_snp'];miss=hits[keys].merge(annotation[keys],on=keys,how='left',indicator=True)
    if (miss._merge=='left_only').any():print('[WARN] Missing clinical annotation for',int((miss._merge=='left_only').sum()),'MRI rows; provide latest 255-pair master',flush=True)
    pcpath=root/'single_iv_causal/updrs3/genotype_pcs/single_iv_genotype_pcs.csv'
    pcmanifest=root/'single_iv_causal/updrs3/genotype_pcs/pc_selection.json'
    if not pcpath.exists():raise FileNotFoundError(pcpath)
    pcs=pd.read_csv(pcpath,low_memory=False);pcs=pcs[pcs.cohort.astype(str).eq('PP')].copy();cols=[c for c in pcs if re.fullmatch('gPC[0-9]+',c)]
    if pcmanifest.exists():
        selections=json.loads(pcmanifest.read_text()).get('cohorts',{}).get('PP',{}).get('selected_columns',[])
        if selections:cols=[c for c in selections if c in pcs]
    pcs['pid']=norm_id(pcs.participant_id);pcs=pcs[['pid']+cols].drop_duplicates('pid')
    m=pd.read_csv(args.mri_file,low_memory=False);m['pid']=norm_id(m.PATNO)
    for col in ['time','age','IQR']:m[col]=pd.to_numeric(m[col],errors='coerce')
    m['sex_num']=m.sex.astype(str).str.strip().str.lower().map({'male':1,'m':1,'1':1,'female':0,'f':0,'0':0,'2':0})
    m=m.merge(pcs,on='pid',how='left',validate='many_to_one').sort_values(['pid','time'])
    if m.duplicated(['pid','time']).any():raise ValueError('Duplicate MRI visits')
    m['previous_time']=m.groupby('pid').time.shift(1)
    m['age_c']=m.age-65.;m['age_c2']=m.age_c**2;m['age_c3']=m.age_c**3
    seen={};records=[];errors=[]
    for key,part in hits.groupby(['gene','lead_snp','artifact_id'],sort=False):
        gene,snp,aid=key;path=root/'single_iv_causal/mri_dbm_lag1/single_iv_genetic_component'/f'{aid}.parquet'
        try:
            if not path.exists():raise FileNotFoundError(path)
            gp=pd.read_parquet(path)
            if 'iv_dosage' not in gp:raise ValueError('Prediction parquet lacks iv_dosage; cannot conduct IV reduced-form diagnostics')
            gp=gp[gp.cohort.astype(str).eq('PP')].copy();gp['pid']=norm_id(gp.participant_id)
            gp['time']=pd.to_numeric(gp.time,errors='coerce')
            if 'within_stage1_followup_support' not in gp:raise ValueError('Stage1 support flag missing')
            gp=gp[gp.within_stage1_followup_support.fillna(False).astype(bool)]
            gp=gp[['pid','time','iv_dosage','grex_single_iv']].drop_duplicates(['pid','time'])
            gp=gp.rename(columns={'time':'previous_time','iv_dosage':'iv_previous','grex_single_iv':'grex_previous'})
            z=m.merge(gp,on=['pid','previous_time'],how='left',validate='many_to_one')
            z['iv_previous']=pd.to_numeric(z.iv_previous,errors='coerce')
            z['grex_previous']=pd.to_numeric(z.grex_previous,errors='coerce')
            for roi in part.roi:
                if roi not in z:raise ValueError(f'ROI absent: {roi}')
                w=z[['pid','time','previous_time','age_c','age_c2','age_c3','sex_num','IQR','iv_previous','grex_previous']+cols+[roi]].copy()
                w['previous_dbm']=z.groupby('pid')[roi].shift(1).to_numpy()
                cov=['previous_dbm','age_c','age_c2','age_c3','sex_num','IQR']+cols
                w=w.dropna(subset=[roi,'iv_previous','grex_previous','previous_time']+cov)
                n=int(len(w));people=int(w.pid.nunique())
                rec={'gene':gene,'lead_snp':snp,'artifact_id':aid,'roi':roi,'robust_n_obs':n,'robust_n_people':people,'iv_ar_test_type':'joint genotype reduced-form Wald (AR-type; NOT classical Anderson-Rubin)'}
                if n<40 or people<args.min_participants:
                    rec['robust_status']='insufficient_sample';records.append(rec);continue
                y=w[roi].to_numpy(float);C=np.column_stack([np.ones(n),w[cov].to_numpy(float)]);x=w.grex_previous.to_numpy(float);g=w.iv_previous.to_numpy(float);ids=w.pid.to_numpy(str)
                X=np.column_stack([C,x]);res=fit(X,y,ids)
                if res is None:rec['robust_status']='singular_primary';records.append(rec);continue
                b,e,v=res;se=float(np.sqrt(max(v[-1,-1],0)));rec.update(robust_status='ok',robust_beta=float(b[-1]),robust_se_cluster=se,robust_partial_r2=partial_r2(y,C,X))
                rf=reduced_form(y,C,g,w.previous_time.to_numpy(float),ids,args.max_ar_degree)
                if rf:rec.update(rf)
                else:rec['robust_status']='singular_reduced_form'
                records.append(rec)
            print('[DONE]',gene,snp,'MRI FDR ROIs',len(part),'/',len(hits),flush=True)
        except Exception as e:errors.append({'gene':gene,'lead_snp':snp,'artifact_id':aid,'error':repr(e)})
    detail=pd.DataFrame(records)
    if detail.empty:raise RuntimeError('No results; inspect errors.csv')
    detail.to_csv(out/'MRI_DBM_IV_reduced_form_AR_type_diagnostics.csv',index=False)
    tab=hits.merge(detail,on=['gene','lead_snp','artifact_id','roi'],how='left',validate='one_to_one')
    tab=tab.merge(annotation,on=keys,how='left',validate='many_to_one')
    tab['clinical_annotation_found']=tab['phenotypes_significant'].notna() if 'phenotypes_significant' in tab else tab[keys].merge(annotation[keys],on=keys,how='left',indicator=True)['_merge'].eq('both').to_numpy()
    if 'phenotypes_significant' not in tab:
        sig=[f'significant_{x}' for x in ['updrs1','updrs2','updrs3','moca']]
        tab['phenotypes_significant']=tab.apply(lambda r:';'.join(c for c in ['updrs1','updrs2','updrs3','moca'] if r.get('significant_'+c,False)==True),axis=1)
    tab['mri_only_ppmi']=True;tab['independent_mri_replication']=False
    tab['interpretation_note']='Exploratory PPMI-only imaging association; PC1 sign is not atrophy direction; AR-type reduced-form is not proof of causal IV validity.'
    tab.to_csv(out/'Supplement_MRI_DBM_significant_IV_gene_ROI_integrated.csv',index=False)
    a2=alltab.merge(annotation,on=keys,how='left',validate='many_to_one')
    a2.to_csv(out/'Supplement_MRI_DBM_ALL_associations_clinical_annotated.csv',index=False)
    pd.DataFrame(errors,columns=['gene','lead_snp','artifact_id','error']).to_csv(out/'errors.csv',index=False)
    (out/'method_manifest.json').write_text(json.dumps({'selection':'Original MRI global BH FDR<0.05 frozen before robustness','IV_test':'Participant-cluster robust Wald test of SNP dosage x polynomial time (degree 0..3) in reduced-form DBM model, adjusted for previous DBM and identical MRI covariates; referred to as AR-type diagnostic but NOT classical Anderson-Rubin','phenotype_annotation_source':str(clinical) if clinical else 'four phenotype-specific standard_final masters','post_selection_p_values':'descriptive only; not independent confirmation','diagnosis_adjusted':False,'warning':'MRI main model did not include diagnosis; re-fit if mixed PD/control and diagnosis effect may confound','errors':len(errors)},indent=2))
    print('[OUTPUT]',out,'hits',len(tab),'errors',len(errors))
if __name__=='__main__':main()
