#!/usr/bin/env python3
"""Rank evidence-backed, high-MAF PD manuscript Figure2 examples (no data fabrication)."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

TOPICAL = {
    'SNCA':6,'LRRK2':6,'GBA':6,'GBA1':6,'VPS35':6,'PRKN':6,
    'PINK1':6,'MAPT':5,'TMEM175':5,'GPNMB':5,'CTSB':5,'ATP13A2':5,
    'SQSTM1':4,'PIK3C3':4,'MERTK':4,'BACE2':4,'PHLPP1':3,
    'SPART':4,'COQ8A':4,'GLRX':3,'CA2':3,'HLA-DQB1':3,
}

def freq_map(path):
    f = pd.read_csv(path,sep=r'\s+',dtype=str)
    idcol = next((s for s in ['ID','SNP','Variant_ID'] if s in f.columns),None)
    frcol = next((s for s in ['MAF','ALT_FREQS','ALT_FREQ','A1_FREQ'] if s in f.columns),None)
    if not idcol or not frcol:
        raise ValueError('Need PLINK .afreq / .frq with ID/SNP and MAF or allele frequency. Available: '+str(f.columns.tolist()[:20]))
    chosen=list(dict.fromkeys([idcol,frcol]+[c for c in ('A1','A2') if c in f.columns]))
    f=f[chosen].copy().rename(columns={idcol:'lead_snp'})
    f['MAF']=pd.to_numeric(f[frcol].str.split(',').str[0],errors='coerce')
    f['MAF']=np.minimum(f['MAF'],1-f['MAF'])
    return f[['lead_snp','MAF']+[c for c in ('A1','A2') if c in f.columns]].drop_duplicates('lead_snp')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--table',required=True)
    p.add_argument('--freq-file',required=True)
    p.add_argument('--output',default='Figure2_ranked_examples.csv')
    p.add_argument('--min-maf',type=float,default=.05)
    p.add_argument('--min-participants',type=int,default=50)
    p.add_argument('--max-rep-p',type=float,default=.05)
    a=p.parse_args()
    data=pd.read_csv(a.table,low_memory=False)
    required=['gene','lead_snp','phenotype']
    miss=[x for x in required if x not in data.columns]
    if miss:raise ValueError(f'Missing {miss}; found {data.columns.tolist()[:40]}')
    data=data.merge(freq_map(a.freq_file),on='lead_snp',how='inner')
    data['pheno_norm']=data['phenotype'].astype(str).str.lower().replace({
      'mds-updrs iii':'updrs3','mds_updrs_iii':'updrs3','mds_updrs_part_iii_summary_score':'updrs3',
      'mds-updrs ii':'updrs2','mds_updrs_ii':'updrs2','mds_updrs_part_ii_patient_questionnaire_summary_score':'updrs2',
      'mds-updrs i':'updrs1','moca_total_score':'moca','moca_score':'moca'
    })
    if 'descriptive_evidence_tier' in data:
        tier=data['descriptive_evidence_tier'].astype(str).str.upper().str.extract(r'([ABC])',expand=False)
        data['evidence_class']=tier.fillna('other')
        data=data[data.evidence_class.isin(['A','B','C'])].copy()
    if 'primary_downstream_candidate' in data:
        truth=data['primary_downstream_candidate'].astype(str).str.lower()
        data=data[truth.isin(['true','1','yes'])].copy()
    if 'selected_replication_pvalue' in data:
        data['rep_p']=pd.to_numeric(data['selected_replication_pvalue'],errors='coerce')
        data=data[data['rep_p'].lt(a.max_rep_p)].copy()
    else:data['rep_p']=np.nan
    obs_col=next((c for c in ['PD_n_participants','PPMI_PD_n_participants','n_participants'] if c in data.columns),None)
    if obs_col:
        data['participants']=pd.to_numeric(data[obs_col],errors='coerce')
        data=data[data['participants'].ge(a.min_participants)].copy()
    else:data['participants']=np.nan
    data=data[data['MAF'].ge(a.min_maf)].copy()
    data['topic_score']=data['gene'].astype(str).str.upper().map(TOPICAL).fillna(0)
    data['log10_rep_p']=-np.log10(data['rep_p'].clip(lower=1e-300))
    data['rank_score']=data['topic_score']*1.7+data['MAF']*15+data['log10_rep_p'].fillna(0).clip(upper=6)+data['evidence_class'].map({'A':12,'B':5,'C':0}).fillna(0)
    data=data.sort_values(['rank_score','participants'],ascending=False)
    data['story_example']=data['gene'].astype(str)+':'+data['pheno_norm'].astype(str)
    cols=['story_example','gene','lead_snp','pheno_norm','MAF','participants','rep_p','topic_score','evidence_class','rank_score']+[c for c in ('A1','A2') if c in data.columns]
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    data[cols].to_csv(a.output,index=False)
    print('Ranked candidates:',len(data),'->',a.output)
    for phenotype in ['updrs3','updrs2','updrs1','moca']:
        z=data[data.pheno_norm.eq(phenotype)].head(8)
        print('\n'+phenotype+'\n'+z[cols].to_string(index=False))
    print('\nImportant: ranking uses genotype frequency and study association table, not observed trajectory completeness at each time. Confirm chosen examples with the plotting output.')

if __name__=='__main__':main()
