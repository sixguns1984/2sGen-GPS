#!/usr/bin/env python3
"""Combine validated IV-gene pairs across UPDRS-I/II/III and MoCA.

Uses only primary_downstream_candidate==True pairs from each final master.
No downstream association can extend this predeclared union.
"""
import argparse,json
from pathlib import Path
import pandas as pd

PHENOS=('updrs1','updrs2','updrs3','moca')
ID=['gene','lead_snp']
KEEP=['PPMI_lag_order','PPMI_pvalue','PPMI_q_bh_global','PPMI_coef_sum',
      'PD_lag_order','PD_pvalue','PD_coef_sum','SU_lag_order','SU_pvalue','SU_coef_sum',
      'PD_PLUS_SU_pvalue','PD_PLUS_SU_coef_sum','PPMI_PD_direction_concordant',
      'PPMI_SU_direction_concordant','both_replication_directions_concordant',
      'n_independent_stage1_IVs','multiIV_p_q_BH','descriptive_evidence_tier',
      'robust_PPMI_ar_pvalue','robust_PD_ar_pvalue','robust_SU_ar_pvalue',
      'primary_downstream_candidate','selection_rule']

def is_true(s):
    return s.astype(str).str.strip().str.lower().isin(['true','1','yes'])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--out-dir',default=None)
    ap.add_argument('--allow-empty',action='store_true')
    a=ap.parse_args()
    root=Path(a.root); out=Path(a.out_dir) if a.out_dir else root/'single_iv_causal/cross_phenotype_summary'
    # Validate every input before writing output; never silently skip incomplete analyses.
    tables={}
    for name in PHENOS:
        f=root/'single_iv_causal'/name/'standard_final'/f'{name}_FINAL_causal_IV_gene_master.csv'
        if not f.is_file():raise FileNotFoundError(f'Missing finalized master: {f}')
        df=pd.read_csv(f,low_memory=False,dtype={'gene':'string','lead_snp':'string'})
        missing=set(ID+['primary_downstream_candidate'])-set(df.columns)
        if missing:raise ValueError(f'{name}: missing columns {sorted(missing)}')
        df=df.loc[is_true(df.primary_downstream_candidate)].copy()
        if df.duplicated(ID).any():raise ValueError(f'{name}: duplicated gene+lead_snp among candidates')
        if df.empty and not a.allow_empty:print(f'[WARN] {name}: zero primary candidates')
        tables[name]=df
    long=[]; wide=[]
    for name,df in tables.items():
        d=df[[*ID,*[c for c in KEEP if c in df.columns]]].copy()
        d.insert(0,'phenotype',name)
        d.insert(1,'replication_design','single_PD' if name=='moca' else 'dual_PD_SU')
        long.append(d)
        rename={c:f'{name}_{c}' for c in d.columns if c not in ID}
        wide.append(d.rename(columns=rename).set_index(ID))
    out.mkdir(parents=True,exist_ok=True)
    long=pd.concat(long,ignore_index=True,sort=False)
    long.to_csv(out/'PD_four_phenotypes_validated_IV_gene_long.csv',index=False)
    w=pd.concat(wide,axis=1,join='outer').reset_index()
    for name in PHENOS:
        c=f'{name}_primary_downstream_candidate'
        w[f'significant_{name}']=is_true(w[c]) if c in w else False
    flags=[f'significant_{p}' for p in PHENOS]
    w['n_significant_phenotypes']=w[flags].sum(axis=1)
    w['n_updrs_significant']=w[[f'significant_{p}' for p in PHENOS[:3]]].sum(axis=1)
    w['phenotypes_significant']=w.apply(lambda x:';'.join(p for p in PHENOS if x[f'significant_{p}']),axis=1)
    w['has_dual_replication']=w.n_updrs_significant.gt(0)
    w['moca_single_replication_only']=w.significant_moca & ~w.has_dual_replication
    w=w.sort_values(['n_significant_phenotypes','n_updrs_significant','gene','lead_snp'],ascending=[False,False,True,True])
    w.to_csv(out/'PD_four_phenotypes_validated_IV_gene_wide.csv',index=False)
    eligible_path=root/'single_iv_causal/iv_validation/single_iv_primary_eligible.csv'
    if not eligible_path.is_file():raise FileNotFoundError(f'Stage1 eligibility file missing: {eligible_path}')
    eligible=pd.read_csv(eligible_path,low_memory=False,dtype={'gene':'string','lead_snp':'string'})
    if not set(ID+['artifact_id']).issubset(eligible.columns):raise ValueError('Stage1 eligible file requires gene, lead_snp and artifact_id')
    if 'iv_primary_eligible' in eligible:eligible=eligible.loc[is_true(eligible.iv_primary_eligible)].copy()
    eligible=eligible.drop_duplicates(ID+['artifact_id'])
    if eligible.duplicated(ID).any():raise ValueError('Stage1 eligible file has multiple artifact_id for a gene+lead_snp')
    candidates=eligible.merge(w[ID+['n_significant_phenotypes','n_updrs_significant','phenotypes_significant','has_dual_replication','moca_single_replication_only']],on=ID,how='inner',validate='one_to_one')
    missing=w[ID].merge(candidates[ID],on=ID,how='left',indicator=True).query('_merge == "left_only"').drop(columns='_merge')
    missing.to_csv(out/'PD_validated_IV_gene_missing_stage1_artifacts.csv',index=False)
    candidates.to_csv(out/'PD_validated_IV_gene_stage1_candidates.csv',index=False)
    gene=w.groupby('gene',as_index=False).agg(n_unique_IVs=('lead_snp','nunique'),max_n_significant_phenotypes=('n_significant_phenotypes','max'),any_dual_replication=('has_dual_replication','max'))
    for name in PHENOS:
        present=w.loc[w[f'significant_{name}'],'gene'].dropna().unique()
        gene[f'gene_significant_{name}']=gene.gene.isin(present)
    gene['n_phenotypes_with_gene_support']=gene[[f'gene_significant_{name}' for name in PHENOS]].sum(axis=1)
    gene.to_csv(out/'PD_four_phenotypes_validated_gene_summary.csv',index=False)
    counts={k:{'n_iv_gene_pairs':len(v),'n_genes':int(v.gene.nunique())} for k,v in tables.items()}
    m={'input_files':{p:str(root/'single_iv_causal'/p/'standard_final'/f'{p}_FINAL_causal_IV_gene_master.csv') for p in PHENOS},'cohort_evidence':{'updrs1':'PPMI+PD+SU','updrs2':'PPMI+PD+SU','updrs3':'PPMI+PD+SU','moca':'PPMI+PD'},'by_phenotype':counts,'union_pairs':len(w),'union_genes':int(w.gene.nunique()),'stage1_candidate_rows':len(candidates),'missing_stage1_artifacts':len(missing),'note':'Only already validated pairs are included; gene-level support can involve different IVs across phenotypes.'}
    (out/'PD_four_phenotypes_summary.json').write_text(json.dumps(m,indent=2,ensure_ascii=False))
    print(json.dumps(m,indent=2,ensure_ascii=False));print('Results:',out)
    if len(missing):raise RuntimeError('Some validated pairs missing Stage1 artifacts. See missing CSV before downstream modeling.')
if __name__=='__main__':main()
