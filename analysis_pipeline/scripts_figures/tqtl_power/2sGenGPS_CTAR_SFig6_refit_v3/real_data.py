"""Adapters for real current-v8.9 Stage-I metadata, expression and PLINK --export A.
Fail clearly on unknown layouts; NEVER manufacture genotypes, observations or covariates.
"""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd


def _norm(s):return re.sub('[^a-z0-9]','',str(s).lower())

def field(cols,aliases,required=True):
    lut={_norm(x):x for x in cols}
    for a in aliases:
        if _norm(a) in lut:return lut[_norm(a)]
    if required:raise ValueError(f'Cannot locate {aliases} among {list(cols)[:85]}; specify a column override')
    return None

SAMPLE=('sample_id','sampleid','rnaseq_sample_id','RNA_sample','RNA_ID','rna_id','sample','Sample_ID')
PERSON=('participant_id','participant','donor_id','individual_id','subject_id','PATNO','subject','IID','patient_id')
COHORT=('cohort','cohort_name','study','study_name','rna_cohort','genotype_cohort','program')
TIME=('followup_time','followup_years','time_from_baseline','years_from_baseline','visit_time','followup','time_year','time_years')

class Metadata:
    def __init__(self,path,cohort,extra=(),cov_cols=None,cohort_col=None,sample_col=None,person_col=None,time_col=None):
        m=pd.read_parquet(path) if str(path).endswith('.parquet') else pd.read_csv(path)
        m=m.reset_index(drop=False) if m.index.name and m.index.name not in m.columns else m
        for p in extra:
            x=pd.read_parquet(p) if str(p).endswith('.parquet') else pd.read_csv(p)
            k=field(m.columns,SAMPLE,False); q=field(x.columns,SAMPLE,False)
            if k and q:
                if x[q].duplicated().any():raise ValueError(f'{p}: duplicate sample IDs in auxiliary covariates')
                cols=[c for c in x if c!=q and c not in m]
                m=m.merge(x[[q]+cols],left_on=k,right_on=q,how='left',validate='many_to_one')
                if q!=k:m=m.drop(columns=q)
            else:
                k=field(m.columns,PERSON);q=field(x.columns,PERSON)
                if x[q].duplicated().any():raise ValueError(f'{p}: duplicate participant IDs in auxiliary covariates')
                cols=[c for c in x if c!=q and c not in m]
                m=m.merge(x[[q]+cols],left_on=k,right_on=q,how='left',validate='many_to_one')
                if q!=k:m=m.drop(columns=q)
        co=cohort_col or field(m.columns,COHORT)
        values=m[co].astype(str).str.lower().str.strip()
        sets={'PP':{'pp','ppmi','parkinson progression markers initiative'},
              'PD':{'pd','pdbp','parkinsons disease biomarkers program','parkinson disease biomarkers program'}}
        m=m.loc[values.isin(sets[cohort])].copy()
        if m.empty:raise ValueError(f'No {cohort} rows found in metadata cohort field {co}; values={values.value_counts().head(15).to_dict()}')
        s=sample_col or field(m.columns,SAMPLE)
        p=person_col or field(m.columns,PERSON)
        t=time_col or field(m.columns,TIME)
        m=m.rename(columns={s:'sample',p:'donor',t:'time'})
        m['sample']=m['sample'].astype(str).str.strip()
        m['donor']=m['donor'].astype(str).str.strip()
        m['time']=pd.to_numeric(m['time'],errors='coerce')
        m=m[m['sample'].ne('')&m.donor.ne('')&m.time.notna()]
        if m['time'].max()>100 or m['time'].min()< -10:
            raise ValueError(f'{cohort}: time variable does not appear to be YEARS. Supply --time-col or preconvert to years')
        if m['sample'].duplicated().any():raise ValueError(f'{cohort}: duplicate RNA sample_id in metadata')
        rin=field(m.columns,('RIN','rin_value','rna_integrity_number'),False)
        if rin:m['_RIN_priority']=pd.to_numeric(m[rin],errors='coerce').fillna(-1)
        else:m['_RIN_priority']=-1.
        m=m.sort_values(['_RIN_priority','sample'],ascending=[False,True])
        m=m.drop_duplicates(['donor','time'],keep='first').drop(columns='_RIN_priority')
        if cov_cols:
            names=cov_cols
            absent=[k for k in names if k not in m.columns]
            if absent:raise ValueError(f'Requested nuisance covariates absent: {absent}')
        else:
            names=self.auto_covariates(m,cohort)
        if not names:raise ValueError('No known covariates found. Supply --cov-cols and --cov-file for production Stage-I matching.')
        for name in names:
            if not pd.api.types.is_numeric_dtype(m[name]):
                vals=m[name].astype(str).str.strip().str.lower()
                if name.lower().startswith('sex') or 'gender' in name.lower():
                    m[name]=vals.map({'m':1,'male':1,'f':0,'female':0,'1':1,'0':0})
                elif 'diagnos' in name.lower() or 'status' in name.lower():
                    # Auto coding is unambiguous for PD vs CT; otherwise require numeric covariate
                    m[name]=vals.map({'pd':1,'parkinson':1,'case':1,'control':0,'ct':0,'healthy':0,'1':1,'0':0})
                else:
                    raise ValueError(f'Covariate {name} is not numeric. Map explicitly before running.')
            m[name]=pd.to_numeric(m[name],errors='coerce')
        m=m.dropna(subset=names)
        if len(m)<100:raise ValueError(f'{cohort}: too few rows ({len(m)}) after covariance completeness; inspect covariates')
        self.data=m;self.cov=names;self.cohort=cohort
        self.summary={'rows':len(m),'donors':m.donor.nunique(),'covariates':names,'time_column':t,
                      'time_min':float(m.time.min()),'time_max':float(m.time.max())}

    @staticmethod
    def auto_covariates(m,cohort):
        order=[]
        for names in [('age_at_baseline','baseline_age','age_baseline'),
                      ('sex_binary','sex','gender_binary'),('diagnosis_binary','diagnosis'),
                      ('RIN','rin_value','rna_integrity_number')]:
            z=field(m.columns,names,False)
            if z:order.append(z)
        # Genome-wide PLINK PCs usually PC1..PC8; expression PCs are study FULL axes.
        for pc in range(1,9):
            z=field(m.columns,(f'genotype_PC{pc}',f'geno_pc{pc}',f'PC{pc}',f'genetic_pc{pc}'),False)
            if z:order.append(z)
        for c in m.columns:
            low=c.lower()
            if ('epc_be_'+('pp_full' if cohort=='PP' else 'pd_full') in low
                or low.startswith('expression_pc_')
                or low.startswith('cibx_clr_')):
                order.append(c)
        return list(dict.fromkeys(order))

class Genotypes:
    def __init__(self,path):
        p=Path(path)
        if not p.exists():raise FileNotFoundError(f'Missing PLINK --export A file {p}')
        data=pd.read_csv(p,sep=r'\s+',low_memory=False)
        fid=field(data.columns,('FID','#FID'));iid=field(data.columns,('IID',))
        data['_id']=data[iid].astype(str).str.strip()
        data['_fid_iid']=data[fid].astype(str).str.strip()+'_'+data['_id']
        if data._id.duplicated().any():
            raise ValueError(f'{p}: IID not unique; use unique per-donor genotypes or update IDs')
        self.data=data.set_index('_id',drop=False)
        self.cols=set(data.columns)
    def vector(self,variant,donor):
        key=str(variant)
        matches=[c for c in self.cols if c==key or c.startswith(key+'_')]
        if len(matches)!=1:raise KeyError(f'{key}: expected one exported dosage column, got {matches[:10]}; check PLINK extraction and variant IDs')
        v=self.data[matches[0]]
        ids=pd.Index(pd.Series(donor).astype(str).str.strip())
        match=ids.isin(self.data.index)
        # Strict ID match. If this fails, do not silently guess a different identifier.
        x=pd.to_numeric(v.reindex(ids),errors='coerce').to_numpy(float)
        x[abs(x-np.round(x))>1e-7]=np.nan
        return x, int(match.sum())

class Expression:
    def __init__(self,path):
        self.path=Path(path)
        if not self.path.exists():raise FileNotFoundError(str(path))
        self.is_csv=self.path.suffix.lower()=='.csv'
        if self.is_csv:
            self.table=pd.read_csv(self.path)
            self.ds=None
            self.cols=list(self.table.columns)
        else:
            import pyarrow.dataset as ds
            self.ds=ds.dataset(str(path),format='parquet')
            self.cols=self.ds.schema.names
        self.sample=field(self.cols,SAMPLE,False)
        self.genecol=field(self.cols,('Ensembl_ID','gene_id','gene','feature_id'),False)
        self.valcol=field(self.cols,('expression_z','expression','z','value','expr','count'),False)
        if self.genecol and self.valcol and self.sample: self.mode='long'
        elif self.sample:self.mode='wide'
        else:
            self.mode='gene_rows'
            if not self.genecol:
                raise ValueError('Unknown expression parquet layout; provide sample-indexed wide or sample_id,gene_id,expression long format')
        self.cache={}
    def get_gene(self,gene):
        gene=str(gene)
        if gene in self.cache:return self.cache[gene]
        if self.mode=='long':
            if self.is_csv:
                q=self.table.loc[self.table[self.genecol].astype(str)==gene,[self.sample,self.valcol]].copy()
            else:
                import pyarrow.dataset as ds
                table=self.ds.to_table(columns=[self.sample,self.valcol],filter=ds.field(self.genecol)==gene)
                q=table.to_pandas()
            q.columns=['sample','y']
        elif self.mode=='wide':
            col=field(self.cols,(gene,gene.split('.')[0]),False)
            if not col:raise KeyError(f'Gene column {gene} not in expression matrix. Check gene-ID versioning')
            q=self.table[[self.sample,col]].copy() if self.is_csv else self.ds.to_table(columns=[self.sample,col]).to_pandas()
            q.columns=['sample','y']
        else:
            # Gene rows and RNA samples across columns. Pull one gene row only,
            # but scanning parquet once per gene can be slow; prefer preprocessing.
            if self.is_csv:row=self.table.loc[self.table[self.genecol].astype(str)==gene].copy()
            else:
                import pyarrow.dataset as ds
                row=self.ds.to_table(filter=ds.field(self.genecol)==gene).to_pandas()
            if row.empty: raise KeyError(f'Gene {gene} not found in expression matrix')
            q=pd.DataFrame({'sample': [c for c in row if c!=self.genecol],
                            'y':[row.iloc[0][c] for c in row if c!=self.genecol]})
        q['sample']=q['sample'].astype(str)
        q['y']=pd.to_numeric(q['y'],errors='coerce')
        if q['sample'].duplicated().any():raise ValueError(f'Expression gene {gene}: duplicate sample IDs')
        self.cache={gene:q.dropna(subset=['y']).set_index('sample')['y']}
        return self.cache[gene]

def build_locus(meta,geno,expression,gene,variant,min_donors=80,min_minor_group=5):
    m=meta.data.copy()
    y=expression.get_gene(gene)
    m['y']=pd.to_numeric(y.reindex(m['sample']).to_numpy(),errors='coerce')
    donor_g,nmatch=geno.vector(variant,m['donor'])
    m['G']=donor_g
    m=m[np.isfinite(m['y'])&np.isfinite(m['G'])]
    if len(m)==0:raise ValueError('No joint expression, genotype and metadata rows')
    counts=m.groupby('donor').size()
    if len(counts)<min_donors:raise ValueError(f'Only {len(counts)} donors; minimum is {min_donors}')
    g=m.drop_duplicates('donor')['G'].to_numpy()
    grp=np.bincount(g.astype(int),minlength=3)
    if grp.min()<min_minor_group:raise ValueError(f'Sparse genotype groups {grp.tolist()}; require each >= {min_minor_group}')
    cov=m[meta.cov].to_numpy(float)
    std=cov.std(axis=0)
    if np.any(std<1e-9):
        bad=np.array(meta.cov)[std<1e-9]
        raise ValueError(f'Constant covariate columns in matched locus: {list(bad)}')
    cov=(cov-cov.mean(axis=0))/std
    time=m['time'].to_numpy(float)
    rho_warning=bool(np.any(np.diff(np.sort(m.time.unique()))<=0))
    return dict(time=time, donor=m['donor'].to_numpy(str),genotype=m['G'].to_numpy(float),
                y=m['y'].to_numpy(float),cov=cov,
                n_donors=len(counts),n_obs=len(m),g_counts=grp.tolist(),
                maf=float(np.sum(g)/(2*len(g))),n_covariates=cov.shape[1])
