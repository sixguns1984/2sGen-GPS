#!/usr/bin/env python3
"""CTSH frozen GReX -> same-visit CSF CTSH -> UPDRS-III exploratory path mediation.

Only exploratory associations: not identification of causal mediated effect.
"""
import argparse,json,warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt'
ARTIFACT='cbf36d8c77f5a1dc'

def read(path):
    p=Path(path)
    return pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p,low_memory=False)

def sex_value(s):
    z=s.astype(str).str.lower().str.strip()
    return z.map({'male':1,'m':1,'1':1,'female':0,'f':0,'0':0,'2':0})

def grex_path(root,override,artifact,gene):
    if override:return Path(override)
    matches=list((root/'single_iv_causal').glob(f'*/single_iv_genetic_component/{artifact}.parquet'))
    matches=[p for p in matches if p.is_file()]
    if not matches:raise FileNotFoundError(f'No frozen GReX for {gene} artifact={artifact}; supply --grex-file')
    # Duplicate copies are acceptable only if same realpath; do not silently choose distinct sources.
    canonical=sorted(matches,key=lambda x:(x.parent.parent.name!= 'updrs3',str(x)))
    if len(matches)>1:print('[GREX SOURCES] candidates:',*[str(p) for p in matches],sep='\n')
    return canonical[0]

def merge_grex(m,g,gene,snp):
    required=['cohort','participant_id','time','grex_single_iv']
    miss=set(required)-set(g.columns)
    if miss:raise ValueError(f'GReX missing: {miss}')
    g=g.copy();m=m.copy()
    for d in (m,g):
        d['cohort']=d.cohort.astype(str).str.strip().str.upper()
        d['participant_id']=d.participant_id.astype(str).str.strip()
        d['_timekey']=pd.to_numeric(d.time,errors='coerce').round(8)
    if 'gene' in g:g=g.loc[g.gene.astype(str)==gene]
    if 'lead_snp' in g:g=g.loc[g.lead_snp.astype(str)==snp]
    g=g.loc[g.cohort.isin(['PP','PD'])]
    if g.duplicated(['cohort','participant_id','_timekey']).any():raise ValueError('Duplicate GReX subject/time')
    keep=['cohort','participant_id','_timekey','grex_single_iv']+[v for v in ['within_stage1_followup_support'] if v in g]
    x=m.merge(g[keep],on=['cohort','participant_id','_timekey'],how='left',validate='many_to_one')
    x['GReX']=pd.to_numeric(x.grex_single_iv,errors='coerce')
    return x.drop(columns=['_timekey'])

def prepare(m, use_medication=True, prior=False):
    d=m.copy()
    if 'AGE' not in d:
        for c in ['age_visit','age_visit_clinical']:
            if c in d:d['AGE']=pd.to_numeric(d[c],errors='coerce');break
        if 'AGE' not in d:raise ValueError('AGE unavailable')
    if 'SEX' not in d:
        c='sex' if 'sex' in d else 'sex_clinical'
        d['SEX']=sex_value(d[c])
    d['AGE']=pd.to_numeric(d.AGE,errors='coerce')
    # Center age for numerical stability in cubic adjustment.
    d['age_c']=d.AGE-65.
    d['age_c2']=d.age_c**2;d['age_c3']=d.age_c**3
    d['cluster_id']=d.cohort.astype(str)+'::'+d.participant_id.astype(str)
    # Keep standardization fixed using complete matched rows within the cohort.
    for col,derived in [('GReX','x'),('CTSH_NPX','m')]:
        d[derived]=d.groupby('cohort')[col].transform(lambda q:(q-q.mean())/q.std(ddof=0) if np.isfinite(q.std(ddof=0)) and q.std(ddof=0)>1e-10 else np.nan)
    d['y']=pd.to_numeric(d.UPDRS3,errors='coerce')
    cov=['age_c','age_c2','age_c3','SEX']
    for source in ['diagnosis','diagnosis_clinical','olink_batch_PP_D02']+(['upd23a_medication_for_pd'] if use_medication else []):
        if source not in d:continue
        v=pd.to_numeric(d[source],errors='coerce')
        if v.notna().sum()==0 and source.startswith('diagnosis'):
            v=d[source].astype('category').cat.codes.replace(-1,np.nan)
        if v.nunique(dropna=True)>1:
            safe='adj_'+source
            d[safe]=v
            cov.append(safe)
    # Prior clinical score: strictly earlier clinical visit, not same-time CSF.
    if prior:
        if 'y_prev' not in d:raise ValueError('No previous UPDRS3 available for sensitivity')
        cov+=['y_prev']
    return d,cov

def add_prior_clinical(m,clinical):
    c=clinical.copy()
    c['cohort']=c.cohort.astype(str).str.strip().str.upper()
    c['participant_id']=c.participant_id.astype(str).str.strip()
    c['time']=pd.to_numeric(c.time,errors='coerce')
    c['mds_updrs_part_iii_summary_score']=pd.to_numeric(c.mds_updrs_part_iii_summary_score,errors='coerce')
    c=c.dropna(subset=['time','mds_updrs_part_iii_summary_score']).sort_values(['cohort','participant_id','time'])
    # For every matched visit, find the latest strictly preceding clinical assessment.
    right=c[['cohort','participant_id','time','mds_updrs_part_iii_summary_score']].rename(columns={'time':'prev_visit_time','mds_updrs_part_iii_summary_score':'y_prev'})
    left=m.copy().reset_index(drop=True);left['_row']=np.arange(len(left))
    z=left.merge(right,on=['cohort','participant_id'],how='left')
    z=z[z.prev_visit_time < z.clinical_time-1e-7]
    z=z.sort_values(['_row','prev_visit_time']).drop_duplicates('_row',keep='last')
    m=m.copy();m['y_prev']=np.nan
    m.loc[z['_row'].astype(int),'y_prev']=z.y_prev.to_numpy(float)
    return m

def fit_paths(d,cov,min_people):
    req=['x','m','y','cluster_id']+cov
    d=d.dropna(subset=req).copy()
    if d.cluster_id.nunique()<min_people:return None,d
    if d.x.std()<1e-8 or d.m.std()<1e-8: return None,d
    covtxt=' + '.join(['1']+cov)
    formulas={'a':f'm ~ x + {covtxt}',
              'total':f'y ~ x + {covtxt}',
              'joint':f'y ~ x + m + {covtxt}'}
    fits={name:smf.ols(f,data=d).fit(cov_type='cluster',cov_kwds={'groups':d.cluster_id}) for name,f in formulas.items()}
    a=float(fits['a'].params['x']);b=float(fits['joint'].params['m']);direct=float(fits['joint'].params['x']);total=float(fits['total'].params['x'])
    res={'n_obs':len(d),'n_people':d.cluster_id.nunique(),'a':a,'a_p_cluster':float(fits['a'].pvalues['x']),
         'b':b,'b_p_cluster':float(fits['joint'].pvalues['m']),
         'indirect_ab':a*b,'direct_cprime':direct,'direct_p_cluster':float(fits['joint'].pvalues['x']),
         'total_c':total,'total_p_cluster':float(fits['total'].pvalues['x']),
         'total_minus_direct':total-direct,'decomposition_difference':(total-direct)-a*b,
         'pct_mediated_exploratory':100*a*b/total if abs(total)>1e-8 else np.nan}
    return (res,fits),d

def bootstrap(d,cov,nboot,seed,min_people):
    rng=np.random.default_rng(seed)
    clusters=d.cluster_id.unique()
    if len(clusters)<min_people:return np.array([])
    by={k:d[d.cluster_id==k] for k in clusters}
    effects=[]
    covtxt=' + '.join(['1']+cov)
    for _ in range(nboot):
        chosen=rng.choice(clusters,len(clusters),replace=True)
        sample=pd.concat([by[c] for c in chosen],ignore_index=True)
        try:
            a=smf.ols('m ~ x + '+covtxt,data=sample).fit().params['x']
            b=smf.ols('y ~ x + m + '+covtxt,data=sample).fit().params['m']
            if np.isfinite(a*b):effects.append(a*b)
        except (ValueError,np.linalg.LinAlgError,KeyError):pass
    return np.asarray(effects,float)

