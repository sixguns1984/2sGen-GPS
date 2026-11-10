#!/usr/bin/env python3
"""Audit that selected Stage-II cohort coef_sum signs agree, without refitting."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def check(df, a, b, label):
    for c in (a,b):
        if c not in df.columns:raise ValueError(f'{label}: missing {c}')
    x=pd.to_numeric(df[a],errors='coerce');y=pd.to_numeric(df[b],errors='coerce')
    bad=~x.mul(y).gt(0)
    if bad.any():
        raise ValueError(f'{label}: {int(bad.sum())} coef_sum-discordant rows! '+str(df.loc[bad,['gene','lead_snp',a,b]].head(8).to_dict('records')))
    return int(len(df)),int(df.gene.nunique()),int(df[['gene','lead_snp']].drop_duplicates().shape[0])

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--require-power',action='store_true');a=p.parse_args();root=Path(a.root)
rows=[]
for ph in ('updrs1','updrs2','updrs3'):
    f=root/'single_iv_causal'/ph/'standard_final'/f'{ph}_FINAL_causal_IV_gene_master_FULL.csv'
    d=pd.read_csv(f,low_memory=False)
    n,g,iv=check(d,'PPMI_coef_sum','PD_PLUS_SU_coef_sum',ph)
    rows.append((ph,n,g,iv))
f=root/'single_iv_causal/ad_mmse/standard_final/AD_MMSE_FINAL_evidence_graded.csv'
if f.is_file():
    d=pd.read_csv(f,low_memory=False);n,g,iv=check(d,'MAP_coef_sum','ADNI_coef_sum','AD MMSE');rows.append(('ad_mmse',n,g,iv))
print(pd.DataFrame(rows,columns=['phenotype','rows','genes','unique_gene_snp']).to_string(index=False))
if a.require_power:
    for ph, f in [('PD',root/'single_iv_causal/cross_phenotype_summary_lag1/PD_validated_IV_gene_stage1_candidates.csv'),('AD',root/'single_iv_causal/ad_mmse/standard_final/posthoc_evidence/Supplement_AD_MMSE_IV_gene_publication_POWER80.csv')]:
        d=pd.read_csv(f,low_memory=False)
        if not len(d):raise ValueError(f'{ph} power-filtered table is empty')
        for col in ('bootstrap_power_PP','bootstrap_power_PD'):
            if col not in d or pd.to_numeric(d[col],errors='coerce').lt(.8).any():raise ValueError(f'{ph}: invalid {col}')
        print(f'[POWER80] {ph}: {len(d)} rows')
