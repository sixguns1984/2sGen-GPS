#!/usr/bin/env python3
"""Safe read-only discovery of AMP-PD postmortem donor covariates.

Does not infer age/sex/PD status from VCF genotypes, sample ID, or study-level
75/25 case proportions. Only actual donor-level tables with verified IDs qualify.
"""
from __future__ import annotations
import csv
import os
import re
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

EXTS=('.csv','.tsv','.txt','.csv.gz','.tsv.gz','.parquet','.xlsx','.xls')
TOKENS=('meta','clinic','phenot','donor','patient','subject','sample','participant',
        'demograph','covaria','biospecimen','manifest','individual','inform','annot','cohort')
COLUMNS={
 'id':('iid','donor','donor_id','donorid','participant_id','participantid','sample_id','sampleid',
       'subject','subject_id','subjectid','individual_id','individualid','wgs_id','sample_name','sample'),
 'age':('age','age_at_death','ageatdeath','age_at_death_years','age_death','age_death_years',
        'age_death_yrs','age_of_death','age_when_died','age_years','age_in_years','age_at_autopsy','age_death_y'),
 'sex':('sex','sex_at_birth','biological_sex','biologicalsex','gender','gender_at_birth','sex_birth'),
 'case':('case','case_control','casecontrol','pd_status','pd_diagnosis','disease_status',
         'diseasestatus','diagnosis','diagnosis_group','disease','condition','group',
         'phenotype','clinical_diagnosis','subject_diagnosis','disease_group','study_group'),
}

def norm_col(x): return re.sub(r'[^a-z0-9]+','_',str(x).strip().lower()).strip('_')

def detect(cols, field):
    lookup={norm_col(c):c for c in cols}
    for name in COLUMNS[field]:
        if name in lookup:return lookup[name]
    return None

def norm_sample(s):
    s=str(s).strip()
    s=re.sub(r'^0_', '',s)
    s=re.sub(r'-(?:GPI|DMNX|PFC|PMC|PVC)-DWGS$', '',s, flags=re.I)
    return s

def aliases(donor):
    donor=norm_sample(donor)
    out={donor}
    # Explicit AMP-PD donor alias only; not arbitrary substring matching.
    m=re.fullmatch(r'PM-HA_(S\d+)-BLM\d+',donor,re.I)
    if m:
        s=m.group(1).upper()
        out.update((s,'PM-HA_'+s))
    return out

def donor_map(donors):
    lut=defaultdict(set)
    for donor in donors:
        for a in aliases(donor):lut[a.casefold()].add(donor)
    return lut

def normalize_case(x):
    if pd.isna(x):return None
    y=str(x).strip().lower().replace('_',' ').replace('-',' ')
    y=' '.join(y.split())
    if y in ('pd','pd case','case','parkinson disease','parkinsons disease',
             "parkinson's disease",'parkinson disease patient','parkinsons','patient','1','true','yes','affected'):
        return 'Case'
    if y in ('control','ct','hc','ctrl','healthy','healthy control','normal',
             'neurologically normal','neurologically healthy','no pd','non pd',
             'unaffected','0','false','no'):
        return 'Control'
    return None

def normalize_sex(x):
    if pd.isna(x):return None
    s=str(x).strip().lower()
    if s in ('m','male','man','1'):return 'M'
    if s in ('f','female','woman','2'):return 'F'
    # 0 ambiguous across sources; never guess.
    return None

def normalize_age(x):
    if pd.isna(x):return None
    z=pd.to_numeric(pd.Series([x]),errors='coerce').iloc[0]
    if not np.isfinite(z) or z<18 or z>120:return None
    return float(z)

def candidate_paths(roots, output_root, max_depth=4, max_file_bytes=200_000_000, max_paths=3000):
    seen=set();total=0
    outroot=Path(output_root).resolve()
    for root in roots:
        root=Path(root).expanduser()
        if not root.exists():continue
        if root.is_file():
            if root.name.lower().endswith(EXTS):
                resolved=str(root.resolve())
                if resolved not in seen and root.stat().st_size<=max_file_bytes:
                    seen.add(resolved);yield root
            continue
        root=root.resolve()
        for base,dirs,files in os.walk(root):
            here=Path(base)
            depth=len(here.relative_to(root).parts)
            dirs[:]=[d for d in sorted(dirs) if not d.startswith('.') and
                     d.lower() not in ('pseudobulk','de','raw','counts','fastq','bam','bams','vcf','plink','tmp','cache','logs')]
            if depth>=max_depth:dirs[:]=[]
            for name in sorted(files):
                if not name.lower().endswith(EXTS):continue
                p=here/name
                rp=str(p.resolve())
                if rp in seen or (outroot==p.resolve() or outroot in p.resolve().parents):continue
                if not any(tok in (name+' '+str(here.name)).lower() for tok in TOKENS):continue
                try:
                    if p.stat().st_size>max_file_bytes:continue
                except OSError:continue
                seen.add(rp);yield p
                total+=1
                if total>=max_paths:return

def read_table(path):
    n=path.name.lower()
    if n.endswith(('.csv','.csv.gz')):return pd.read_csv(path,nrows=250000,low_memory=False)
    if n.endswith(('.tsv','.tsv.gz')):return pd.read_csv(path,sep='\t',nrows=250000,low_memory=False)
    if n.endswith('.txt'):
        with open(path,encoding='utf-8',errors='replace') as f:head=f.readline()
        if '\t' in head:return pd.read_csv(path,sep='\t',nrows=250000,low_memory=False)
        if ',' in head:return pd.read_csv(path,nrows=250000,low_memory=False)
        return None
    if n.endswith('.parquet'):
        # pandas will report missing pyarrow rather than invent columns.
        return pd.read_parquet(path)
    if n.endswith(('.xlsx','.xls')):return pd.read_excel(path,nrows=250000)
    return None

def evaluate(path, donors, lut):
    diag={'file':str(path),'status':'SKIPPED','n_rows':0,'n_matched_donors':0,
          'n_valid_age':0,'n_valid_sex':0,'n_valid_case':0,'n_complete':0,
          'columns':'','id_column':'','age_column':'','sex_column':'','case_column':'','reason':''}
    try:
        x=read_table(path)
        if x is None:
            diag['reason']='unsupported text delimiter';return diag,{}
    except Exception as ex:
        diag['reason']=f'read_error:{type(ex).__name__}:{str(ex)[:120]}';return diag,{}
    diag['n_rows']=len(x);diag['columns']='|'.join(map(str,x.columns[:100]))
    idc=detect(x.columns,'id')
    if idc is None:
        diag['reason']='no_exact_ID_column';return diag,{}
    field_cols={field:detect(x.columns,field) for field in ('age','sex','case')}
    diag['id_column']=str(idc)
    for k,v in field_cols.items():diag[k+'_column']=str(v or '')
    if not any(field_cols.values()):
        diag['reason']='no_usable_covariate_column';return diag,{}
    records=defaultdict(lambda:defaultdict(set));methods=defaultdict(set)
    for row in x.itertuples(index=False,name=None):
        # Integer index lookup for reliable whitespace-containing column names.
        item=dict(zip(x.columns,row))
        raw=item[idc]
        if pd.isna(raw):continue
        sid=norm_sample(raw)
        matches=lut.get(sid.casefold(),set())
        if len(matches)!=1:continue
        donor=next(iter(matches))
        methods[donor].add('exact' if sid==donor else 'AMP_PD_sample_alias')
        for field,c in field_cols.items():
            if c is None:continue
            val=(normalize_age(item[c]) if field=='age' else
                 normalize_sex(item[c]) if field=='sex' else normalize_case(item[c]))
            if val is not None:records[donor][field].add(val)
    diag['n_matched_donors']=len(methods)
    conflicts=[];clean={}
    for donor,vals in records.items():
        for field,choices in vals.items():
            if len(choices)>1:conflicts.append(f'{donor}:{field}:{choices}')
        clean[donor]={field:next(iter(z)) for field,z in vals.items() if len(z)==1}
    if conflicts:
        diag['status']='CONFLICT';diag['reason']='within_file_conflicts: '+str(conflicts[:5]);return diag,{}
    for field in ('age','sex','case'):
        diag['n_valid_'+field]=sum(field in z for z in clean.values())
    diag['n_complete']=sum(all(f in z for f in ('age','sex','case')) for z in clean.values())
    if len(methods)<5 or sum(diag['n_valid_'+f] for f in ('age','sex','case'))==0:
        diag['reason']='insufficient_ID_or_covariate_overlap';return diag,{}
    diag['status']='MATCHED'
    return diag,clean

def discover_metadata(donors, roots, out, min_complete=5):
    """Save an auditable canonical CSV only if non-conflicting, genuine donor values found."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    donors=sorted(set(map(str,donors)))
    if len(donors)!=len(set(donors)):raise ValueError('Duplicate normalized donors')
    lut=donor_map(donors)
    alias_ambiguities={k:list(v) for k,v in lut.items() if len(v)>1}
    if alias_ambiguities:raise ValueError('Ambiguous AMP-PD sample aliases: '+str(list(alias_ambiguities.items())[:5]))
    summaries=[]; candidates=[]
    for p in candidate_paths(roots,out):
        diag,records=evaluate(p,donors,lut)
        summaries.append(diag)
        if records:candidates.append((diag,records))
    df=pd.DataFrame(summaries)
    if df.empty:df=pd.DataFrame(columns=['file','status','reason'])
    df.to_csv(out/'candidate_table_audit.csv',index=False)
    (out/'searched_roots.json').write_text(json.dumps([str(p) for p in roots],ensure_ascii=False,indent=2))
    if not candidates:
        raise RuntimeError('No donor-level metadata table with age/sex/PD status was found by ID. See '+
            str(out/'candidate_table_audit.csv')+'; provide an actual clinical source with --donor-metadata PATH; never invent covariates.')
    # All observed values across eligible source tables must agree. Do not pick a
    # high-scoring but contradictory table or infer case status by 75/25 totals.
    vals=defaultdict(lambda:defaultdict(set));source=defaultdict(lambda:defaultdict(set))
    for d,records in candidates:
        for donor,fields in records.items():
            for field,val in fields.items():
                vals[donor][field].add(val)
                source[donor][field].add(d['file'])
    bad=[(donor,field,str(choices)) for donor,fields in vals.items() for field,choices in fields.items() if len(choices)>1]
    if bad:
        pd.DataFrame(bad,columns=['donor','field','conflicting_values']).to_csv(out/'conflicting_clinical_values.csv',index=False)
        raise RuntimeError('Contradictory clinical values across candidate tables: '+str(bad[:5])+'. See '+str(out/'conflicting_clinical_values.csv'))
    rows=[]
    for donor in donors:
        v=vals[donor]
        row={'donor':donor, 'age':next(iter(v['age'])) if v['age'] else np.nan,
             'sex':next(iter(v['sex'])) if v['sex'] else np.nan,
             'case':next(iter(v['case'])) if v['case'] else np.nan}
        for f in ('age','sex','case'):
            row[f+'_sources']='|'.join(sorted(source[donor][f]))
        rows.append(row)
    final=pd.DataFrame(rows)
    final.to_csv(out/'candidate_donor_coverage.csv',index=False)
    complete=final.dropna(subset=['age','sex','case'])
    if len(complete)<min_complete or complete['case'].nunique()<2:
        raise RuntimeError('Donor-level clinical metadata found but insufficient jointly complete case/control records: '+
                           f'{len(complete)} complete donors, case labels={complete.case.value_counts().to_dict()}. '
                           'See '+str(out/'candidate_donor_coverage.csv'))
    # Use distinct source metadata for case/control; never impute missing clinical values.
    outfile=out/'canonical_donor_metadata.csv'
    final[['donor','age','sex','case']].to_csv(outfile,index=False)
    provenance={'n_wgs_donors':len(donors),'n_full_covariates':len(complete),
                'case_counts':complete['case'].value_counts().to_dict(),
                'source_files':sorted({d['file'] for d,_ in candidates}),
                'note':'Donor ID matched using explicit aliases only; clinical values never inferred from genotype or ID.'}
    (out/'metadata_provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2))
    return outfile,provenance

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--vcf-genotypes',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--roots',nargs='+',required=True)
    a=p.parse_args()
    g=pd.read_csv(a.vcf_genotypes)
    f,diag=discover_metadata(g.donor.dropna().unique(),a.roots,a.out)
    print(f,diag)
