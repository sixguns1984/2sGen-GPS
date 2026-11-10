#!/usr/bin/env python3
"""Select EVERY frozen primary AD MMSE IV-gene for PP/PD CTAR bootstrap.

No MAP/ADNI Stage-II refit. Stage-I locus matching and raw BH cutoff reuse
scripts_figures/tqtl_power/ctar_replication_v5/00_prepare_replicated_loci.py.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve()
REPO=HERE.parents[2]
V5=REPO/'scripts_figures/tqtl_power/ctar_replication_v5'
if str(V5) not in sys.path:sys.path.insert(0,str(V5))

KEY=['gene','lead_snp','artifact_id']

def load_final(path):
    if not path.is_file():raise FileNotFoundError(f'Existing AD MMSE FINAL missing: {path}; do not rerun main Stage-II')
    d=pd.read_csv(path,low_memory=False,dtype={k:str for k in KEY})
    if not set(KEY).issubset(d):raise ValueError(f'FINAL missing {set(KEY)-set(d)}')
    if 'PRIMARY_replicated_same_lag_direction' in d.columns:
        p=d.PRIMARY_replicated_same_lag_direction.astype('string').str.lower().isin(['true','1','yes'])
        if not p.all():raise ValueError('FINAL contains non-replicated rows; use AD_MMSE_FINAL_evidence_graded.csv')
    if d.duplicated(KEY).any() or d.duplicated(KEY[:2]).any():raise ValueError('Duplicate AD MMSE exact IV-gene pairs')
    return d

def select(root,final):
    from importlib import import_module
    prep=import_module('00_prepare_replicated_loci')
    stage=root/'single_iv_causal/iv_validation/single_iv_primary_eligible.csv'
    st=pd.read_csv(stage,low_memory=False,dtype={k:str for k in KEY})
    if not set(KEY).issubset(st):raise ValueError('Stage-I eligible table lacks exact gene, lead_snp, artifact_id')
    if 'iv_primary_eligible' in st:
        st=st.loc[st.iv_primary_eligible.astype('string').str.lower().isin(['true','1','yes'])]
    st=st[KEY].drop_duplicates()
    if st.duplicated(KEY[:2]).any():raise ValueError('Stage-I gene/SNP maps to multiple artifact IDs')
    ck=final[KEY].merge(st,on=KEY,how='left',indicator=True,validate='one_to_one')
    if not ck._merge.eq('both').all():
        bad=ck.loc[ck._merge.ne('both'),KEY].head(15).to_dict('records')
        raise ValueError(f'{(~ck._merge.eq("both")).sum()} AD pairs not in exact Stage-I eligible IV set: {bad}')
    pool=final[KEY[:2]].rename(columns={'lead_snp':'variant'}).copy()
    disc=root/'qtl/PP/submitted_followup_ctar_mptga/final/significant_tqtl.parquet'
    rep=root/'qtl/PD/submitted_followup_ctar_mptga/final/all_results.parquet'
    selected,nall,bh=prep.read_discovery(disc,pool)
    selected=prep.match_rep(rep,selected)
    bad=selected.rep_match.ne('both')|~selected.pd_nominal_replicated.fillna(False)
    if bad.any():raise ValueError(f'{bad.sum()} AD IV-gene pairs fail Stage-I PP/PD replication check. Examples: '+str(selected.loc[bad,['gene','variant','pd_pvalue']].head(15).to_dict('records')))
    return selected,nall,bh,disc,rep,stage

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--outdir',required=True)
    ap.add_argument('--final-file',default='')
    a=ap.parse_args();root=Path(a.root);out=Path(a.outdir)
    final_path=Path(a.final_file) if a.final_file else root/'single_iv_causal/ad_mmse/standard_final/AD_MMSE_FINAL_evidence_graded.csv'
    f=load_final(final_path)
    if (out/'locus_bootstrap_results.csv').exists() or (out/'_shards').exists():
        raise FileExistsError(f'Power output already exists at {out}; never mix results from different universes')
    selected,nall,bh,disc,rep,stage=select(root,f)
    out.mkdir(parents=True,exist_ok=True)
    selected.to_csv(out/'selected_loci.csv',index=False)
    f[KEY].to_csv(out/'AD_MMSE_primary_exact_IV_gene_pairs.csv',index=False)
    audit={'mode':'ad-mmse-all-final','n_primary_ad_mmse_IV_gene_pairs':len(f),
           'selected_PPMI':len(selected),'matched_PDBP':len(selected),
           'pdbp_nominal_replicated':int(selected.pd_nominal_replicated.sum()),
           'observed_max_raw_p_of_all_hits':bh,
           'total_significant':nall,'stage1_eligible_source':str(stage),
           'AD_final_source':str(final_path),'PP_discovery_file':str(disc),
           'PD_replication_file':str(rep),
           'method':'conditional observed-donor bootstrap redetection of 8df overall ctAR temporal tQTL',
           'selection':'ALL AD MMSE MAP->ADNI replicated same-lag coef_sum-direction IV-gene pairs; NOT random sampling',
           'note':'PP/PD are Stage-I tQTL cohorts; MAP/ADNI are Stage-II MMSE clinical cohorts.'}
    (out/'selection_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
