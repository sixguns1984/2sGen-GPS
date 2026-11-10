#!/usr/bin/env python3
"""Fast, no-refit AD MMSE FINAL v1.13 correction using v1.12 saved master.

Does not rerun discovery, ADNI replication, PCA, genotype processing,
GReX prediction, or reduced-form models. Backups before overwriting.
"""
from __future__ import annotations
import argparse, json, shutil, sys, hashlib
from pathlib import Path
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
from importlib import import_module
m=import_module('20_finalize_ad_mmse')

OUTPUTS=['AD_MMSE_SUPPLEMENT_all_MAP_FDR_IV_gene_pairs.csv',
         'AD_MMSE_FINAL_replicated_same_lag_direction.csv',
         'AD_MMSE_FINAL_evidence_graded.csv',
         'AD_MMSE_FINAL_gene_summary.csv','AD_MMSE_FINAL_summary.json']


def assert_primary(master):
    required=['MAP_q_bh_global','MAP_coef_lag1','ADNI_coef_lag1','ADNI_pvalue',
              'selected_lag1_direction_concordant','PRIMARY_replicated_same_lag_direction',
              'MAP_FDR_lags','selected_lag_order']
    absent=set(required)-set(master.columns)
    if absent:raise ValueError(f'Master missing fields for replication audit: {sorted(absent)}')
    if master[m.KEY].duplicated().any():raise ValueError('Duplicate IV-gene-artifact rows')
    primary=m.valid_bool(master.PRIMARY_replicated_same_lag_direction)
    selected=master.loc[primary]
    valid=(m.as_num(selected.MAP_q_bh_global).lt(.05) & m.as_num(selected.ADNI_pvalue).lt(.05)
           & m.valid_bool(selected.selected_lag1_direction_concordant)
           & (m.as_num(selected.MAP_coef_lag1)*m.as_num(selected.ADNI_coef_lag1)).gt(0))
    if not bool(valid.all()):raise ValueError('Selected PRIMARY includes wrong lag/sign/FDR/P: cannot regrade')
    lag_ok=[str(int(v)) in str(lags).split(';') for v,lags in
            zip(selected.selected_lag_order,selected.MAP_FDR_lags)]
    if not all(lag_ok):raise ValueError('Selected PRIMARY lag not in MAP FDR significant lags')
    return primary


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    ap.add_argument('--final-dir',default='')
    a=ap.parse_args()
    f=Path(a.final_dir) if a.final_dir else Path(a.root)/'single_iv_causal'/'ad_mmse'/'standard_final'
    src=f/OUTPUTS[0];summary=f/OUTPUTS[-1]
    if not src.exists() or not summary.exists():raise FileNotFoundError(f'Missing previous completed FINAL: {src} / {summary}')
    prior=json.loads(summary.read_text())
    if prior.get('stage2_discovery')!='MAP' or prior.get('stage2_replication')!='ADNI':
        raise ValueError('Refusing to regrade data not labeled MAP discovery and ADNI replication')
    if prior.get('status')!='COMPLETE':
        raise ValueError(f'Previous FINAL status={prior.get("status")} not COMPLETE; review errors before changing final files')
    original_master_sha256=hashlib.sha256(src.read_bytes()).hexdigest()
    raw=pd.read_csv(src,low_memory=False)
    primarymask=assert_primary(raw)
    print(f'[AUDIT] MAP FDR pairs={len(raw)}, replicated MAP->ADNI same-lag direction={int(primarymask.sum())}',flush=True)
    corrected=m.evidence(raw)
    must_not=[c for c in corrected if ('first_stage' in c.lower() or 'firststage' in c.lower() or 'stage1_strength' in c.lower())]
    if must_not:raise ValueError(f'First-stage strength fields remain: {must_not}')
    final=corrected.loc[primarymask].copy()
    final=final.sort_values(['evidence_tier','ADNI_pvalue','MAP_q_bh_global','gene'],kind='mergesort')
    corrected=corrected.sort_values(['PRIMARY_replicated_same_lag_direction','MAP_q_bh_global','gene'],ascending=[False,True,True],kind='mergesort')
    backup=f/'backup_before_v113_no_first_stage_F';backup.mkdir(exist_ok=True)
    for filename in OUTPUTS:
        source=f/filename;dest=backup/filename
        if source.is_file() and not dest.exists():shutil.copy2(source,dest)
    corrected.to_csv(f/OUTPUTS[0],index=False)
    final.to_csv(f/OUTPUTS[1],index=False)
    final.to_csv(f/OUTPUTS[2],index=False)
    m.gene_aggregate(corrected).to_csv(f/OUTPUTS[3],index=False)
    counts=corrected.evidence_tier.value_counts().to_dict()
    prior.update({
        'status':'COMPLETE','final_revision':'v1.13_remove_first_stage_F_strength',
        'n_MAP_FDR_IV_gene_pairs':len(corrected),
        'n_ADNI_primary_same_lag_direction_pairs':len(final),
        'n_ADNI_primary_genes':int(final.gene.nunique()),
        'evidence_tier_counts':counts,
        'first_stage_F_used_in_grading':False,
        'first_stage_F_computed_in_finalization':False,
        'RF_BH_family':'selected lag of all MAP FDR IV-gene pairs, post-selection descriptive only',
        'evidence_tier_definitions':{
            'A':'MAP discovery + ADNI same lag/sign; MAP RF selected-pair BH q<0.05; ADNI RF p<0.05',
            'B':'MAP discovery + ADNI same lag/sign; MAP+ADNI RF nominal p<0.05 but MAP RF BH criterion not met',
            'C':'MAP discovery + ADNI same lag/sign; dual RF support not established or not evaluated',
            'D':'MAP discovery only; ADNI same-lag sign-concordant replication not met'},
        'causal_identification':'NOT ESTABLISHED; no pleiotropy/colocalization causal validation',
        'regrade_source_master_sha256':original_master_sha256,
        'regrade_backup_dir':str(backup),
    })
    # Delete obsolete and misleading old text wherever it appears.
    prior['true_causal_identification']='NOT ESTABLISHED; exclusion restriction, pleiotropy, colocalization untested; first-stage F not used or computed'
    (f/OUTPUTS[-1]).write_text(json.dumps(prior,ensure_ascii=False,indent=2))
    print(f'[FINAL v1.13] graded {len(corrected)} MAP-discovered pairs; primary replicated={len(final)}; counts={counts}',flush=True)
    print(f'[FINAL v1.13] no first-stage F criterion; previous tables saved in {backup}',flush=True)
    print(f'[DONE] {f/OUTPUTS[2]}',flush=True)

if __name__=='__main__': main()
