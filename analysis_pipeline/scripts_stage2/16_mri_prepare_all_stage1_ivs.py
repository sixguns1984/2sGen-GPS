#!/usr/bin/env python3
"""Prepare all outcome-independent Stage-I eligible IVs for genes with frozen MRI FDR hits."""
import argparse,json
from pathlib import Path
import pandas as pd

def yes(s):
    return s.astype('string').str.lower().str.strip().isin(('true','1','yes'))

def select_all_ivs(hits, stage1):
    keys=['gene','lead_snp']
    for label,df in [('MRI FDR',hits),('Stage-I',stage1)]:
        missing=set(keys)-set(df.columns)
        if missing:raise ValueError(f'{label} missing {missing}')
    need=['artifact_id','basis_json']
    if any(c not in stage1 for c in need) or not any(c.startswith('beta_g_') for c in stage1):
        raise ValueError('Stage-I eligible table must have artifact_id, basis_json, beta_g_*')
    stage1=stage1.copy()
    for col in keys:
        stage1[col]=stage1[col].astype('string').str.strip()
        hits[col]=hits[col].astype('string').str.strip()
    if 'iv_primary_eligible' in stage1: stage1=stage1[yes(stage1.iv_primary_eligible)].copy()
    if stage1.duplicated(keys).any():raise ValueError('Duplicate Stage-I eligible gene + SNP pairs')
    genes=set(hits.gene.dropna())
    out=stage1[stage1.gene.isin(genes)].copy()
    if out.empty:raise ValueError('No eligible Stage-I IVs for MRI hit genes')
    lead=hits[keys].drop_duplicates()
    missing=lead.merge(out[keys],on=keys,how='left',indicator=True)
    missing=missing[missing['_merge'].eq('left_only')][keys]
    if len(missing):raise ValueError(f'MRI FDR lead IVs missing from eligible Stage-I pool:\n{missing.head(12)}')
    if 'artifact_id' in hits:
        hm=hits[keys+['artifact_id']].drop_duplicates()
        x=hm.merge(out[keys+['artifact_id']],on=keys,suffixes=('_MRI','_stage1'))
        bad=x[x.artifact_id_MRI.astype(str).ne(x.artifact_id_stage1.astype(str))]
        if len(bad):raise ValueError('Stage-I artifact IDs disagree with MRI FDR leads: '+str(bad.head()))
    return out.sort_values(keys).reset_index(drop=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--component',type=int,default=1)
    a.add_argument('--mri-fdr',default='');a.add_argument('--stage1-eligible',default='');a.add_argument('--out-dir',default='')
    args=a.parse_args();root=Path(args.root)
    f=Path(args.mri_fdr) if args.mri_fdr else root/f'single_iv_causal/mri_dbm_lag1/pc{args.component}/MRI_DBM_lag1_global_FDR05.csv'
    st=Path(args.stage1_eligible) if args.stage1_eligible else root/'single_iv_causal/iv_validation/single_iv_primary_eligible.csv'
    out=Path(args.out_dir) if args.out_dir else root/f'single_iv_causal/mri_dbm_lag1/pc{args.component}/evidence_v1';out.mkdir(parents=True,exist_ok=True)
    hits=pd.read_csv(f,low_memory=False,dtype={'gene':'string','lead_snp':'string','artifact_id':'string'})
    if hits.empty:raise ValueError(f'No MRI global FDR hits: {f}')
    allivs=select_all_ivs(hits,pd.read_csv(st,low_memory=False,dtype={'gene':'string','lead_snp':'string','artifact_id':'string'}))
    p=out/'MRI_hit_genes_ALL_eligible_stage1_IVs.csv';allivs.to_csv(p,index=False)
    manifest={'MRI_hits':int(len(hits)),'MRI_hit_gene_IV_pairs':int(hits[['gene','lead_snp']].drop_duplicates().shape[0]),'MRI_hit_genes':int(hits.gene.nunique()),'all_eligible_IVs':int(len(allivs)),'gene_IV_counts':allivs.groupby('gene').size().astype(int).to_dict(),'candidate_file':str(p),'MRI_FDR_source':str(f),'Stage1_source':str(st),'role':'Auxiliary IVs are selected based on Stage-I eligibility, NOT MRI p-values, and are for post-selection robustness only'}
    (out/'all_IV_preparation_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
