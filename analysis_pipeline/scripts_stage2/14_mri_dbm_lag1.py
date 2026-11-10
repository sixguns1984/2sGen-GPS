#!/usr/bin/env python3
"""PPMI longitudinal MRI DBM lag-1 association using frozen 8-parameter single-IV GReX.
Statistical association; not a proven structural IV causal effect.
"""
import argparse,json,re,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

ROOT='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt'
DEFAULT_MRI='/public2/data/luojunfeng/2sTGCS_binary/outcome/DBM/confound_pass_snp/MRI_DBM.csv'

def txtid(s):
    return (s.astype('string').str.strip().str.replace(r'^PP-', '',regex=True)
            .str.replace(r'\.0$', '',regex=True))

def read_candidates(path):
    a=pd.read_csv(path,low_memory=False)
    if not {'gene','lead_snp'}.issubset(a.columns):raise ValueError('Candidate list needs gene and lead_snp')
    if 'PD_artifact_id' in a.columns:a['artifact_id']=a.PD_artifact_id
    if 'artifact_id' not in a.columns:raise ValueError('Candidate list needs artifact_id or PD_artifact_id; supply cross-phenotype validated IV table')
    a=a[['gene','lead_snp','artifact_id']].dropna().drop_duplicates(['gene','lead_snp'])
    if a[['gene','lead_snp']].duplicated().any():raise ValueError('Conflicting IV-gene candidates')
    return a.reset_index(drop=True)

def get_pcs(root,pc_path,manifest_path):
    p=Path(pc_path) if pc_path else root/'single_iv_causal/updrs3/genotype_pcs/single_iv_genotype_pcs.csv'
    manifest=Path(manifest_path) if manifest_path else root/'single_iv_causal/updrs3/genotype_pcs/pc_selection.json'
    if not p.exists():raise FileNotFoundError(f'Genotype PCs required, missing {p}')
    pc=pd.read_csv(p,low_memory=False)
    pc=pc.loc[pc.cohort.astype(str).eq('PP')].copy()
    cols=[x for x in pc.columns if re.fullmatch(r'gPC\d+',str(x))]
    if manifest.exists():
        selected=json.loads(manifest.read_text()).get('cohorts',{}).get('PP',{}).get('selected_columns',[])
        if selected:cols=[c for c in selected if c in pc.columns]
    if not cols:raise ValueError(f'No selected genotype PCs in {p}')
    pc['pid']=txtid(pc.participant_id)
    pc=pc[['pid']+cols].drop_duplicates('pid')
    for col in cols:pc[col]=pd.to_numeric(pc[col],errors='coerce')
    return pc,cols

def prepare_mri(mri,pc,pcs,pc_component,min_delta):
    d=pd.read_csv(mri,low_memory=False)
    cols=[c for c in d.columns if re.search(r'\.pc[123]$',str(c),re.I) and str(c).lower().endswith('.pc'+str(pc_component))]
    if not cols:raise ValueError('No regional .pc{pc_component} columns found')
    required=['PATNO','time','age','sex','IQR']
    if any(c not in d for c in required):raise ValueError(f'MRI required columns missing {set(required)-set(d)}')
    d['pid']=txtid(d.PATNO)
    for c in ['time','age','IQR']+cols:d[c]=pd.to_numeric(d[c],errors='coerce')
    d['sex_num']=d.sex.astype(str).str.lower().str.strip().map({'male':1,'m':1,'1':1,'female':0,'f':0,'2':0,'0':0})
    d=d.dropna(subset=['pid','time']).copy()
    if d.duplicated(['pid','time']).any():
        conflict=d.groupby(['pid','time']).size().loc[lambda s:s>1]
        raise ValueError(f'Duplicate MRI participant/time (first 5): {conflict.head().to_dict()}')
    d=d.merge(pc,on='pid',how='left',validate='many_to_one')
    if d[pcs].isna().all(axis=1).any():print('[WARN] MRI visits without genotype PCs:',int(d[pcs].isna().all(axis=1).sum()),flush=True)
    d=d.sort_values(['pid','time']).reset_index(drop=True)
    # Visit lag is created from MRI chronology, never rows of a different dataset.
    d['previous_time']=d.groupby('pid').time.shift(1)
    d['gap_years']=d.time-d.previous_time
    d['previous_age']=d.groupby('pid').age.shift(1)
    for c in cols:d['prev__'+c]=d.groupby('pid')[c].shift(1)
    d=d[d.gap_years>=min_delta].reset_index(drop=True)
    d['age_c']=d.age-65.
    d['age_c2']=d.age_c**2;d['age_c3']=d.age_c**3
    return d,cols

def get_grex(root,pair,analysis_name):
    aid=str(pair.artifact_id).strip()
    p=root/'single_iv_causal'/analysis_name/'single_iv_genetic_component'/f'{aid}.parquet'
    if not p.exists():raise FileNotFoundError(f'MRI-specific frozen Stage1 prediction missing {p}; run MRI prediction steps 03 and 05')
    a=pd.read_parquet(p)
    needed={'cohort','participant_id','time','grex_single_iv','within_stage1_followup_support'}
    if not needed.issubset(a):raise ValueError(f'Invalid MRI-specific prediction {p}: missing {needed-set(a)}')
    a=a[a.cohort.astype(str).eq('PP')].copy()
    if 'gene' in a:a=a[a.gene.astype(str).eq(str(pair.gene))]
    if 'lead_snp' in a:a=a[a.lead_snp.astype(str).eq(str(pair.lead_snp))]
    a['pid']=txtid(a.participant_id)
    a['time']=pd.to_numeric(a.time,errors='coerce')
    a['grex_single_iv']=pd.to_numeric(a.grex_single_iv,errors='coerce')
    a=a[a.within_stage1_followup_support.fillna(False).astype(bool)].dropna(subset=['time','grex_single_iv'])
    if a.duplicated(['pid','time']).any():raise ValueError(f'Duplicate MRI-specific GReX for participant/time {p}')
    return a[['pid','time','grex_single_iv']]

def grex_at_times(mri,grex):
    z=mri[['pid','previous_time']].merge(grex.rename(columns={'time':'previous_time','grex_single_iv':'GReX_lag1'}),on=['pid','previous_time'],how='left',validate='many_to_one',sort=False)
    if len(z)!=len(mri):raise RuntimeError('MRI GReX join expanded rows')
    return z[['GReX_lag1']].reset_index(drop=True)

def fit_one(y,yp,x,cov,clusters,min_people,min_obs):
    # OLS ARX with participant-cluster robust sandwich covariance for GReX coefficient.
    mat=np.column_stack([np.ones(len(y)),x,yp,cov]);mask=np.isfinite(mat).all(axis=1)&np.isfinite(y)
    Y=y[mask];A=mat[mask];C=clusters[mask]
    n=len(Y);g=len(np.unique(C));k=A.shape[1]
    if n<min_obs or g<min_people or n<=k+3 or np.linalg.matrix_rank(A)<k:return None
    beta=np.linalg.lstsq(A,Y,rcond=None)[0];u=Y-A@beta
    inv=np.linalg.pinv(A.T@A);score=np.zeros((g,k))
    for j,c in enumerate(np.unique(C)):score[j,:]=A[C==c,:].T@u[C==c]
    small=(g/(g-1))*((n-1)/(n-k))
    cov_b=small*inv@(score.T@score)@inv
    se=float(np.sqrt(max(0,cov_b[1,1])));b=float(beta[1]);p=float(2*norm.sf(abs(b/se))) if se>1e-14 else np.nan
    return {'n_obs':n,'n_people':g,'beta_grex_lag1':b,'se_cluster':se,'z_cluster':b/se if se>0 else np.nan,'p_cluster':p,'beta_DBM_lag1':float(beta[2]),'mean_MRI_gap_years':np.nan }

def task(arg):
    row,root,base,cols,pcs,min_people,min_obs,analysis_name=arg
    pair=type('Pair',(),row)
    try:
        g=get_grex(Path(root),pair,analysis_name)
        m=base.copy();z=grex_at_times(m,g);m=pd.concat([m,z],axis=1)
        out=[]
        m['pid_code']=pd.factorize(m.pid)[0]
        covnames=['age_c','age_c2','age_c3','sex_num','IQR']+pcs
        C=m[covnames].to_numpy(float);x=m.GReX_lag1.to_numpy(float);cluster=m.pid_code.to_numpy(int)
        for col in cols:
            y=m[col].to_numpy(float);yp=m['prev__'+col].to_numpy(float)
            result=fit_one(y,yp,x,C,cluster,min_people,min_obs)
            if result is None:continue
            result.update(gene=row['gene'],lead_snp=row['lead_snp'],artifact_id=row['artifact_id'],roi=col,
                          PC_component=col.rsplit('.',1)[-1],model='DBM_current ~ GReX_previous + DBM_previous + age_cubic + sex + genotype_PCs + IQR',
                          grex_alignment="direct_MRI_visit_prediction")
            result['mean_MRI_gap_years']=float(m.loc[np.isfinite(y)&np.isfinite(yp)&np.isfinite(x),'gap_years'].mean())
            out.append(result)
        return out,None
    except Exception as e:return [],{'gene':row['gene'],'lead_snp':row['lead_snp'],'artifact_id':row['artifact_id'],'error':repr(e)}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default=ROOT);ap.add_argument('--mri-file',default=DEFAULT_MRI)
    ap.add_argument('--candidates',default='');ap.add_argument('--out-dir',default='')
    ap.add_argument('--pc-file',default='');ap.add_argument('--pc-manifest',default='')
    ap.add_argument('--component',type=int,choices=[1,2,3],default=1)
    ap.add_argument('--analysis-name',default='mri_dbm_lag1')
    ap.add_argument('--min-participants',type=int,default=30);ap.add_argument('--min-obs',type=int,default=50)
    ap.add_argument('--min-mri-gap-years',type=float,default=1e-6);ap.add_argument('--n-jobs',type=int,default=8)
    a=ap.parse_args();root=Path(a.root)
    cand=Path(a.candidates) if a.candidates else root/'single_iv_causal/cross_phenotype_summary_lag1/PD_validated_IV_gene_stage1_candidates.csv'
    out=Path(a.out_dir) if a.out_dir else root/'single_iv_causal/mri_dbm_lag1_frozen_255'
    out.mkdir(parents=True,exist_ok=True)
    c=read_candidates(cand);pc,pcs=get_pcs(root,a.pc_file,a.pc_manifest)
    manifest=root/'single_iv_causal'/a.analysis_name/'single_iv_genetic_component_manifest.json'
    if not manifest.exists():raise FileNotFoundError(f'MRI-specific prediction manifest not found: {manifest}; run MRI steps 03/05 first')
    info=json.loads(manifest.read_text())
    if info.get('rewrite_support_only'):raise ValueError('MRI predictions were support-only rewritten, not newly computed')
    if info.get('analysis_name')!=a.analysis_name:raise ValueError('Prediction manifest analysis name mismatch')
    source=Path(info.get('candidate_file',''))
    if source.resolve()!=cand.resolve():raise ValueError(f'MRI prediction candidates {source} != analysis candidates {cand}')
    prepared=root/'single_iv_causal'/a.analysis_name/'phenotype_clean.parquet'
    if not prepared.exists():raise FileNotFoundError(f'MRI prediction visit cache missing: {prepared}')
    mritimes=pd.read_csv(a.mri_file,usecols=['PATNO','time'])
    mritimes['pid']=txtid(mritimes.PATNO);mritimes['time']=pd.to_numeric(mritimes.time,errors='coerce')
    target=set(zip(mritimes.pid.astype(str),mritimes.time))
    pcov=pd.read_parquet(prepared,columns=['participant_id','time'])
    pcov['pid']=txtid(pcov.participant_id);pcov['time']=pd.to_numeric(pcov.time,errors='coerce')
    predicted=set(zip(pcov.pid.astype(str),pcov.time))
    if not target.issubset(predicted):raise ValueError(f'MRI-specific model evaluated at incorrect visits: missing {len(target-predicted)} participant/time keys')
    base,cols=prepare_mri(a.mri_file,pc,pcs,a.component,a.min_mri_gap_years)
    print('[INPUT]',len(c),'IV-gene',len(cols),'DBM outcomes','MRI lag pairs',len(base),'people',base.pid.nunique(),flush=True)
    (out/'input_audit.json').write_text(json.dumps({'candidate_file':str(cand),'n_pairs':len(c),'MRI_source':str(a.mri_file),
      'MRI_pairs':len(base),'MRI_people':int(base.pid.nunique()),'n_rois':len(cols),'PCs':pcs,
      'grex_alignment':'direct_MRI_visit_prediction','note':'PC1/2/3 are region principal components; sign does not imply atrophy direction'},indent=2))
    args=[(r._asdict(),str(root),base,cols,pcs,a.min_participants,a.min_obs,a.analysis_name) for r in c.itertuples(index=False)]
    allres=[];errors=[]
    with ProcessPoolExecutor(max_workers=a.n_jobs) as pool:
        jobs=[pool.submit(task,v) for v in args]
        for i,job in enumerate(as_completed(jobs),1):
            result,error=job.result();allres.extend(result)
            if error:errors.append(error)
            if i%20==0:print('[PROGRESS]',i,'/',len(jobs),'tests',len(allres),'errors',len(errors),flush=True)
    pd.DataFrame(errors,columns=['gene','lead_snp','artifact_id','error']).to_csv(out/'errors.csv',index=False)
    if not allres:raise RuntimeError('No eligible model results. Check GReX exact overlap / pc values; see errors.csv and input_audit.json')
    tab=pd.DataFrame(allres)
    good=tab.p_cluster.notna();tab['q_global_BH']=np.nan
    tab.loc[good,'q_global_BH']=multipletests(tab.loc[good,'p_cluster'],method='fdr_bh')[1]
    tab=tab.sort_values('p_cluster',na_position='last')
    tab.to_csv(out/'MRI_DBM_lag1_ALL_associations.csv',index=False)
    tab.loc[tab.q_global_BH<.05].to_csv(out/'MRI_DBM_lag1_global_FDR05.csv',index=False)
    summ=tab.groupby(['gene','lead_snp']).agg(n_rois_tested=('roi','size'),n_nominal=('p_cluster',lambda x:int((x<.05).sum())),
                                            n_FDR05=('q_global_BH',lambda x:int((x<.05).sum())),min_p=('p_cluster','min')).reset_index()
    summ.to_csv(out/'MRI_DBM_IV_gene_summary.csv',index=False)
    print('[DONE] tests',good.sum(),'global BH<0.05',int((tab.q_global_BH<.05).sum()),'errors',len(errors),'out',out)
if __name__=='__main__':main()
