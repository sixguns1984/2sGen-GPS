#!/usr/bin/env python3
"""Compare PD+SU replication selection direction on lag-1 vs coef_sum.

Use the SAME discovery-FDR lag & replication nominal P<0.05 and SAME best-BIC
lag BEFORE applying either direction gate. Join original PP/PD ctAR bootstrap
results. No models are refit. Unknown bootstrap power is NOT treated as <80%.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

PHENOS = ('updrs1', 'updrs2', 'updrs3')
PAIR = ['gene', 'lead_snp']
LAGKEY = PAIR + ['lag_order']


def read_csv(path):
    if not Path(path).is_file():
        raise FileNotFoundError(str(path))
    return pd.read_csv(path, low_memory=False,
                       dtype={'gene': str, 'lead_snp': str, 'artifact_id': str,
                              'variant': str})


def require(df, cols, source):
    absent = set(cols) - set(df.columns)
    if absent:
        raise ValueError(f'{source} missing columns: {sorted(absent)}')


def pair_counts(df):
    return {'rows': len(df), 'iv_gene_pairs': len(df[PAIR].drop_duplicates()),
            'genes': int(df.gene.nunique())}


def read_power(files, min_valid):
    x = pd.concat([read_csv(p) for p in files], ignore_index=True)
    if 'bootstrap_valid' not in x.columns:
        for k in ['n_valid', 'valid_bootstraps', 'bootstrap_n_valid']:
            if k in x.columns:
                x = x.rename(columns={k: 'bootstrap_valid'})
                break
    require(x, ['gene', 'variant', 'cohort', 'bootstrap_power', 'bootstrap_valid'], 'Power files')
    x = x[x.cohort.isin(['PP', 'PD'])].copy()
    x['bootstrap_power'] = pd.to_numeric(x.bootstrap_power, errors='coerce')
    x['bootstrap_valid'] = pd.to_numeric(x.bootstrap_valid, errors='coerce')
    if ((x.bootstrap_power.dropna() < 0) | (x.bootstrap_power.dropna() > 1)).any():
        raise ValueError('bootstrap_power must be in [0, 1].')
    # Allow additional power files only if duplicate locus/cohort results are numerically identical.
    cols = ['gene', 'variant', 'cohort']
    conflicting = x.groupby(cols, dropna=False)[['bootstrap_power', 'bootstrap_valid']].nunique(dropna=False)
    if (conflicting > 1).any(axis=None):
        raise ValueError('Conflicting duplicate Power records across inputs. Provide one consistent source.')
    x = x.drop_duplicates(cols)
    wide = x.pivot(index=['gene', 'variant'], columns='cohort',
                   values=['bootstrap_power', 'bootstrap_valid'])
    wide.columns = [f'{name}_{cohort}' for name, cohort in wide.columns]
    wide = wide.reset_index().rename(columns={'variant': 'lead_snp'})
    for co in ['PP', 'PD']:
        for c in [f'bootstrap_power_{co}', f'bootstrap_valid_{co}']:
            if c not in wide: wide[c] = np.nan
    wide['power_evaluable'] = (
        wide.bootstrap_power_PP.notna() & wide.bootstrap_power_PD.notna()
        & wide.bootstrap_valid_PP.ge(min_valid) & wide.bootstrap_valid_PD.ge(min_valid))
    wide['power80_pass'] = (
        wide.power_evaluable & wide.bootstrap_power_PP.ge(0.8) & wide.bootstrap_power_PD.ge(0.8))
    return wide


def load_predirection(stage_dir, phenotype):
    dis_file = stage_dir/'discovery_significant_IV_gene_lag_models.csv'
    rep_file = stage_dir/'replication_PD_PLUS_SU_all_lags_annotated.csv.gz'
    if not rep_file.is_file():
        rep_file = rep_file.with_suffix('')  # .csv
    disc = read_csv(dis_file)
    rep = read_csv(rep_file)
    require(disc, LAGKEY+['coef_lag1', 'coef_sum'], dis_file)
    require(rep, LAGKEY+['coef_lag1', 'coef_sum', 'pvalue', 'bic'], rep_file)
    for name, frame in [('discovery', disc), ('replication', rep)]:
        if frame.duplicated(LAGKEY).any():
            raise ValueError(f'{phenotype} duplicated IV-gene-lag in {name} results')
    dcols = LAGKEY+['coef_lag1', 'coef_sum']
    if 'artifact_id' in disc.columns:
        dcols.append('artifact_id')
    disc = disc[dcols].rename(columns={
        'coef_lag1': 'PPMI_coef_lag1', 'coef_sum': 'PPMI_coef_sum'})
    rcols = LAGKEY+['coef_lag1', 'coef_sum', 'pvalue', 'bic']
    rep = rep[rcols].rename(columns={
        'coef_lag1': 'PD_PLUS_SU_coef_lag1', 'coef_sum': 'PD_PLUS_SU_coef_sum',
        'pvalue': 'PD_PLUS_SU_pvalue', 'bic': 'PD_PLUS_SU_bic'})
    z = disc.merge(rep, on=LAGKEY, how='inner', validate='one_to_one')
    z['PD_PLUS_SU_pvalue'] = pd.to_numeric(z.PD_PLUS_SU_pvalue, errors='coerce')
    z['PD_PLUS_SU_bic'] = pd.to_numeric(z.PD_PLUS_SU_bic, errors='coerce')
    z['lag_order'] = pd.to_numeric(z.lag_order, errors='coerce')
    z = z[z.PD_PLUS_SU_pvalue.lt(0.05)].copy()
    # Exactly matches the approved 12_summarize... selection order:
    # first choose best BIC from nominally replicated discovery-significant lag orders,
    # THEN assess either coefficient sign.
    z = z.sort_values(PAIR+['PD_PLUS_SU_bic', 'PD_PLUS_SU_pvalue', 'lag_order'],
                      na_position='last').drop_duplicates(PAIR).copy()
    for c in ('PPMI_coef_lag1', 'PD_PLUS_SU_coef_lag1', 'PPMI_coef_sum', 'PD_PLUS_SU_coef_sum'):
        z[c] = pd.to_numeric(z[c], errors='coerce')
    z['lag1_concordant'] = z.PPMI_coef_lag1.mul(z.PD_PLUS_SU_coef_lag1).gt(0)
    z['coefsum_concordant'] = z.PPMI_coef_sum.mul(z.PD_PLUS_SU_coef_sum).gt(0)
    z.insert(0, 'phenotype', phenotype)
    return z


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    parser.add_argument('--power-results', nargs='+', default=None,
                        help='One or more existing locus_bootstrap_results.csv paths')
    parser.add_argument('--min-valid-bootstrap', type=int, default=800)
    parser.add_argument('--current-table', default='', help='Original final POWER80 3-phenotype publication table')
    parser.add_argument('--out-dir', default='')
    args = parser.parse_args()
    root = Path(args.root)
    out = Path(args.out_dir) if args.out_dir else root/'single_iv_causal'/'cross_phenotype_summary_lag1'/'coefsum_direction_comparison'
    out.mkdir(parents=True, exist_ok=True)
    power_files = args.power_results or [str(root/'manuscript_figures'/'tqtl_benchmark_v4'/'clinical_IV_ctar_power_v5'/'locus_bootstrap_results.csv')]
    power = read_power(power_files, args.min_valid_bootstrap)

    rows = []
    for ph in PHENOS:
        stage = root/'single_iv_causal'/ph/'temporal_causal_agecubic_full_followup_fast'
        rows.append(load_predirection(stage, ph))
    all_results = pd.concat(rows, ignore_index=True)
    all_results = all_results.merge(power, on=PAIR, how='left', validate='many_to_one')
    all_results['power_evaluable'] = all_results.power_evaluable.eq(True)
    all_results['power80_pass'] = all_results.power80_pass.eq(True)
    all_results['power_status'] = np.select(
        [all_results.power80_pass, all_results.power_evaluable],
        ['pass_both_80pct', 'below_80pct'], default='not_evaluable')
    all_results['direction_category'] = np.select(
        [all_results.lag1_concordant & all_results.coefsum_concordant,
         all_results.lag1_concordant & ~all_results.coefsum_concordant,
         ~all_results.lag1_concordant & all_results.coefsum_concordant],
        ['both', 'lag1_only', 'coefsum_only'], default='neither')
    all_results['coefsum_new_vs_lag1'] = all_results.direction_category.eq('coefsum_only')
    all_results = all_results.sort_values(['phenotype', 'gene', 'lead_snp'])
    all_results.to_csv(out/'ALL_predirection_sameBIC_with_power.csv', index=False)

    old = all_results[all_results.lag1_concordant]
    new = all_results[all_results.coefsum_concordant]
    new_add = all_results[all_results.coefsum_new_vs_lag1]
    summaries = []
    for ph in [*PHENOS, 'ALL']:
        d = all_results if ph == 'ALL' else all_results[all_results.phenotype.eq(ph)]
        summaries.append({
            'phenotype': ph,
            'pre_direction_nominal_pairs': len(d),
            'lag1_direction_pairs': int(d.lag1_concordant.sum()),
            'coefsum_direction_pairs': int(d.coefsum_concordant.sum()),
            'both_directions_pairs': int((d.lag1_concordant & d.coefsum_concordant).sum()),
            'lag1_only_pairs': int((d.lag1_concordant & ~d.coefsum_concordant).sum()),
            'coefsum_only_pairs': int((~d.lag1_concordant & d.coefsum_concordant).sum()),
            'lag1_power80_known_pass': int((d.lag1_concordant & d.power80_pass).sum()),
            'coefsum_power80_known_pass': int((d.coefsum_concordant & d.power80_pass).sum()),
            'coefsum_new_power80_known_pass': int((~d.lag1_concordant & d.coefsum_concordant & d.power80_pass).sum()),
            'coefsum_new_power_unknown': int((~d.lag1_concordant & d.coefsum_concordant & ~d.power_evaluable).sum()),
            'coefsum_power_unknown': int((d.coefsum_concordant & ~d.power_evaluable).sum()),
            'coefsum_power_evaluable_below80': int((d.coefsum_concordant & d.power_evaluable & ~d.power80_pass).sum()),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv(out/'direction_comparison_summary.csv', index=False)
    new.to_csv(out/'coefsum_direction_ALL_candidates_with_power.csv', index=False)
    new[new.power80_pass].to_csv(out/'coefsum_direction_POWER80_confirmed.csv', index=False)
    new_add.to_csv(out/'coefsum_only_new_candidates.csv', index=False)
    new_add[~new_add.power_evaluable].to_csv(out/'coefsum_only_new_need_power.csv', index=False)
    new_add[~new_add.power_evaluable][[c for c in ['gene','lead_snp','artifact_id'] if c in new_add.columns]].drop_duplicates().to_csv(
        out/'coefsum_only_new_IV_gene_for_power.csv', index=False)

    current = args.current_table or str(root/'single_iv_causal'/'cross_phenotype_summary_lag1'/'Supplement_PD_IV_gene_publication_3phenotypes_POWER80.csv')
    if Path(current).is_file():
        c = read_csv(current)
        require(c, ['phenotype']+PAIR, current)
        saved = set(map(tuple, c[['phenotype']+PAIR].astype(str).to_numpy()))
        reconstructed = set(map(tuple, old[old.power80_pass][['phenotype']+PAIR].astype(str).to_numpy()))
        print(f'[AUDIT] saved current POWER80: {len(saved)}; reconstructed lag1 POWER80: {len(reconstructed)}')
        if saved != reconstructed:
            diff = pd.DataFrame([{'phenotype': k[0], 'gene': k[1], 'lead_snp': k[2],
                                  'where': ('only_saved' if k in saved else 'only_reconstructed')}
                                 for k in sorted(saved.symmetric_difference(reconstructed))])
            diff.to_csv(out/'current_vs_reconstructed_discrepancies.csv', index=False)
            print(f'[WARNING] Current table and raw reconstruction differ on {len(diff)} records. '
                  'Inspect current_vs_reconstructed_discrepancies.csv before interpreting totals.')

    print('\n=== Direction comparison at the SAME best-BIC lag; Power80 PP and PD ===')
    print(summary.to_string(index=False))
    print('\n[NOTE] coefsum_power80_known_pass is the MINIMUM confirmed count, not the final count when new loci lack power.')
    print('[NOTE] Compute missing PP/PD bootstrap power for coefsum_only_new_need_power.csv before quoting final numbers.')
    print('[OUTPUT]', out)
    (out/'comparison_manifest.json').write_text(json.dumps({
        'selection': 'PPMI discovery BH-significant lag + PD+SU p<0.05 at SAME lag; select best BIC BEFORE testing signs',
        'baseline_direction': 'discovery_coef_lag1 * replication_coef_lag1 > 0',
        'alternative_direction': 'discovery_coef_sum * replication_coef_sum > 0',
        'power': 'PP/PD conditional donor-bootstrap >= 80% and >= min_valid_bootstrap each',
        'min_valid_bootstrap': args.min_valid_bootstrap,
        'power_sources': power_files,
        'cohorts': ['PP', 'PD', 'SU', 'PD+SU'],
        'caveat': 'Only pooled PD+SU is used as the primary replication cohort, consistent with the selected UPDRS pipeline; no models are refitted.'}, indent=2))

if __name__ == '__main__':
    main()
