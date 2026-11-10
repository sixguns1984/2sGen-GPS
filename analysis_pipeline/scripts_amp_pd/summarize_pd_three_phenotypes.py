#!/usr/bin/env python3
"""Integrate ALL existing UPDRS I/II/III Stage-II FINAL evidence + Stage-I/Power.

No clinical p-value recomputation; source descriptive tiers and replication labels
are copied intact, with phenotype prefixes in pair-level wide tables. MoCA excluded.
Version 7, 2026-10-31: reviewer-focused compact and extended reports; FULL remains untouched.
"""
from __future__ import annotations
import argparse
import json
import warnings
from pathlib import Path
import pandas as pd

PHENOS = ('updrs1', 'updrs2', 'updrs3')
KEY = ['gene', 'lead_snp']
CORE_EVIDENCE = (
    'descriptive_evidence_tier', 'AR_discoveryFDR_repNominal_support',
    'AR_all3_nominal_support', 'AR_all3_FDR_stringent_annotation',
    'robust_PPMI_ar_pvalue', 'robust_PPMI_ar_q_BH',
    'robust_PD_ar_pvalue', 'robust_PD_ar_q_BH',
    'robust_SU_ar_pvalue', 'robust_SU_ar_q_BH',
    'robust_PD_PLUS_SU_ar_pvalue', 'robust_PD_PLUS_SU_ar_q_BH',
    'multiIV_n', 'multiIV_p', 'multiIV_p_q_BH', 'multiIV_Q',
    'multiIV_Q_p', 'multiIV_Q_p_q_BH', 'multiIV_I2',
    'multiIV_FDR_support', 'multiIV_LOO_stable',
    'LOO_n', 'LOO_max_p', 'LOO_all_nominal_p05',
    'cross_gene_max_r2', 'cross_gene_ld_flag', 'cross_gene_LD_clean',
)

def yes(s):
    return s.astype('string').str.strip().str.lower().isin(('true','1','yes'))

def clean_key(d, label):
    for k in KEY:
        if k not in d:
            raise ValueError(f'{label} missing {k}')
        d[k] = d[k].astype('string').str.strip()
        if d[k].isna().any() or d[k].eq('').any():
            raise ValueError(f'{label} has empty {k}')
    return d

def load_master(root, ph):
    base = root/'single_iv_causal'/ph/'standard_final'
    full = base/f'{ph}_FINAL_causal_IV_gene_master_FULL.csv'
    concise = base/f'{ph}_FINAL_causal_IV_gene_master.csv'
    path = full if full.is_file() else concise
    if not path.is_file():
        raise FileNotFoundError(f'Missing {ph} final master; checked {full} and {concise}')
    d = clean_key(pd.read_csv(path, low_memory=False, dtype={k:'string' for k in KEY}), str(path))
    if 'primary_downstream_candidate' not in d:
        raise ValueError(f'{path} missing primary_downstream_candidate')
    d = d.loc[yes(d['primary_downstream_candidate'])].copy()
    if d.duplicated(KEY).any():
        raise ValueError(f'{path} has duplicated gene + lead_snp')
    if 'phenotype' in d:
        bad = d['phenotype'].dropna().astype(str).str.lower().str.strip().ne(ph)
        if bad.any():
            raise ValueError(f'{path} has other phenotype labels')
    d['phenotype'] = ph
    if 'replication_design' not in d:
        d['replication_design'] = 'not_recorded_in_master'
    # Do not synthesize missing robustness statistics or change frozen evidence tier.
    if 'descriptive_evidence_tier' not in d:
        warnings.warn(f'{path} lacks descriptive_evidence_tier; check whether FULL results were finalized')
    for cohort in ('PPMI', 'PD_PLUS_SU'):
        if f'{cohort}_coef_x_lag1' not in d:
            warnings.warn(f'{path} lacks {cohort}_coef_x_lag1; only existing source values will be reported (NA otherwise)')
    return d, path

def read_power(path, min_valid):
    d = pd.read_csv(path, low_memory=False, dtype={'gene':'string','variant':'string'})
    if 'bootstrap_valid' not in d:
        for field in ('n_valid','valid_bootstraps','bootstrap_n_valid'):
            if field in d:
                d = d.rename(columns={field:'bootstrap_valid'})
                break
    required = {'gene','variant','cohort','bootstrap_power','bootstrap_valid'}
    if not required.issubset(d):
        raise ValueError(f'{path}: Power results missing {sorted(required - set(d.columns))}')
    d = d.rename(columns={'variant':'lead_snp'})
    d = clean_key(d, str(path))
    if not d['cohort'].isin(('PP','PD')).all():
        raise ValueError(f'{path}: unexpected cohort in Power table')
    if d.duplicated(KEY+['cohort']).any():
        raise ValueError(f'{path}: duplicate Power rows per gene + lead_snp + cohort')
    for field in ('bootstrap_power','bootstrap_valid'):
        d[field] = pd.to_numeric(d[field], errors='coerce')
    if d['bootstrap_power'].dropna().lt(0).any() or d['bootstrap_power'].dropna().gt(1).any():
        raise ValueError('Bootstrap detection fraction must be 0..1')
    p = d.pivot(index=KEY, columns='cohort', values=['bootstrap_power','bootstrap_valid'])
    p.columns = [f'{what}_{cohort}' for what, cohort in p.columns]
    p = p.reset_index()
    for cohort in ('PP','PD'):
        for field in (f'bootstrap_power_{cohort}', f'bootstrap_valid_{cohort}'):
            if field not in p: p[field] = float('nan')
    # Wilson confidence intervals for Monte Carlo uncertainty, not model uncertainty.
    z = 1.959963984540054
    for co in ('PP','PD'):
        prop = p[f'bootstrap_power_{co}']
        n = p[f'bootstrap_valid_{co}'].where(p[f'bootstrap_valid_{co}'].gt(0))
        den = 1 + z*z/n
        ctr = (prop + z*z/(2*n))/den
        half = z*((prop*(1-prop)/n+z*z/(4*n*n))**0.5)/den
        p[f'bootstrap_mc_ci_low_{co}'] = ctr-half
        p[f'bootstrap_mc_ci_high_{co}'] = ctr+half
    p['power80_borderline_uncertainty'] = (
        (p['bootstrap_mc_ci_low_PP'].le(.8) & p['bootstrap_mc_ci_high_PP'].ge(.8)) |
        (p['bootstrap_mc_ci_low_PD'].le(.8) & p['bootstrap_mc_ci_high_PD'].ge(.8))
    ).fillna(False).astype(bool)
    return p

def attach_power(pairs, power, min_valid):
    x = pairs.merge(power, on=KEY, how='left', validate='one_to_one')
    evaluable = (
        x['bootstrap_power_PP'].notna() & x['bootstrap_power_PD'].notna() &
        x['bootstrap_valid_PP'].ge(min_valid).fillna(False) &
        x['bootstrap_valid_PD'].ge(min_valid).fillna(False)
    ).fillna(False).astype(bool)
    passing = (
        evaluable & x['bootstrap_power_PP'].ge(.8).fillna(False) &
        x['bootstrap_power_PD'].ge(.8).fillna(False)
    ).fillna(False).astype(bool)
    x['power_evaluable_both'] = evaluable
    x['tqtl_power_both_ge80'] = passing
    x['power_screen_status'] = 'not_evaluable'
    x.loc[evaluable,'power_screen_status'] = 'below_80pct'
    x.loc[passing,'power_screen_status'] = 'pass_both_80pct'
    return x

def annotate_long(long, pair):
    return long.merge(pair, on=KEY, how='left', validate='many_to_one', suffixes=('','_pair'))

def aggregate_best_tier(x):
    cols = [f'{p}_descriptive_evidence_tier' for p in PHENOS]
    def choose(row):
        found=[]
        for ph, col in zip(PHENOS, cols):
            s=row.get(col)
            if pd.isna(s) or not str(s).strip(): continue
            lead = str(s).strip().upper()[0]
            if lead in ('A','B','C'): found.append((lead,ph,str(s)))
        if not found: return ('not_recorded','','')
        found.sort(key=lambda z:'ABC'.index(z[0]))
        return found[0]
    result = x.apply(choose, axis=1).tolist()
    x['best_evidence_tier_level'] = [z[0] for z in result]
    x['best_evidence_tier_source_phenotype'] = [z[1] for z in result]
    x['best_evidence_tier_original_label'] = [z[2] for z in result]
    return x

def make_pair_wide(full):
    # All phenotype master columns are included, not just selection flags.
    parts=[]
    for ph in PHENOS:
        d = full.loc[full.phenotype.eq(ph)].drop(columns=['phenotype']).copy()
        d = d.rename(columns={c:f'{ph}_{c}' for c in d.columns if c not in KEY})
        parts.append(d)
    x = parts[0]
    for more in parts[1:]:
        x = x.merge(more, on=KEY, how='outer', validate='one_to_one')
    for ph in PHENOS:
        x[f'significant_{ph}'] = yes(x[f'{ph}_primary_downstream_candidate']).fillna(False).astype(bool)
    x['n_significant_phenotypes'] = x[[f'significant_{p}' for p in PHENOS]].sum(axis=1)
    x['n_updrs_significant'] = x['n_significant_phenotypes']
    x['phenotypes_significant'] = x.apply(
        lambda r:';'.join(ph for ph in PHENOS if bool(r[f'significant_{ph}'])),axis=1)
    x['has_pooled_PD_SU_replication'] = pd.concat([
        x[f'{ph}_replication_design'].eq('pooled_PD_PLUS_SU') for ph in PHENOS
    ],axis=1).any(axis=1)
    x = aggregate_best_tier(x)
    x = x.sort_values(['n_significant_phenotypes']+KEY,ascending=[False,True,True]).reset_index(drop=True)
    return x

def aggregate_genes(pairs):
    d = pairs.groupby('gene',as_index=False).agg(n_unique_IVs=('lead_snp','nunique'))
    by_gene = []
    for gene,g in pairs.groupby('gene'):
        present = [p for p in PHENOS if g[f'significant_{p}'].any()]
        by_gene.append({'gene':gene,'n_supported_phenotypes':len(present),'phenotypes_supported':';'.join(present)})
    return d.merge(pd.DataFrame(by_gene),on='gene',validate='one_to_one')


# Publication policy: short, source-faithful tables; complete data remain in
# Supplement_PD_IV_gene_ALL_statistics_3phenotypes.csv and Stage-II FULL masters.
# This is a COLUMN SELECTION ONLY: no p-values, FDR, AR, IV tiers, or power are recalculated.
REVIEW_FIELDS = (
    'gene', 'lead_snp', 'phenotype', 'replication_design',
    'selected_lag_order', 'PPMI_lag_order', 'PD_lag_order', 'SU_lag_order', 'PD_PLUS_SU_lag_order',
    'PPMI_pvalue', 'PPMI_q_bh_global',
    'PD_pvalue', 'SU_pvalue', 'PD_PLUS_SU_pvalue',
    'PPMI_n_participants', 'PD_PLUS_SU_n_participants',
    'stage1_ppmi_tqtl_p', 'stage1_pdbp_tqtl_p',
    'stage1_ppmi_first_stage_F', 'stage1_pdbp_first_stage_F',
    'robust_PPMI_ar_q_BH', 'robust_PD_PLUS_SU_ar_q_BH',
    'AR_discoveryFDR_repNominal_support',
    'multiIV_n', 'multiIV_p_q_BH', 'multiIV_Q_p_q_BH', 'multiIV_I2',
    'multiIV_LOO_stable', 'cross_gene_max_r2',
    'descriptive_evidence_tier',
    'bootstrap_power_PP', 'bootstrap_power_PD', 'power_screen_status',
)
EXTENDED_FIELDS = (
    'gene', 'lead_snp', 'phenotype', 'replication_design',
    'selected_lag_order', 'PPMI_lag_order', 'PD_lag_order',
    'SU_lag_order', 'PD_PLUS_SU_lag_order',
    'PPMI_pvalue', 'PPMI_q_bh_global', 'PD_pvalue', 'SU_pvalue',
    'PD_PLUS_SU_pvalue', 'PPMI_n_participants', 'PD_n_participants',
    'SU_n_participants', 'PD_PLUS_SU_n_participants',
    'stage1_ppmi_tqtl_p', 'stage1_pdbp_tqtl_p',
    'stage1_ppmi_first_stage_F', 'stage1_pdbp_first_stage_F',
    'robust_PPMI_ar_pvalue', 'robust_PPMI_ar_q_BH',
    'robust_PD_ar_pvalue', 'robust_PD_ar_q_BH',
    'robust_SU_ar_pvalue', 'robust_SU_ar_q_BH',
    'robust_PD_PLUS_SU_ar_pvalue', 'robust_PD_PLUS_SU_ar_q_BH',
    'AR_discoveryFDR_repNominal_support',
    'multiIV_n', 'multiIV_beta', 'multiIV_se', 'multiIV_p',
    'multiIV_p_q_BH', 'multiIV_Q', 'multiIV_Q_df', 'multiIV_Q_p',
    'multiIV_Q_p_q_BH', 'multiIV_I2',
    'multiIV_direction_concordance_fraction',
    'LOO_n', 'LOO_max_p', 'LOO_all_nominal_p05',
    'multiIV_LOO_stable',
    'cross_gene_max_r2', 'cross_gene_ld_flag',
    'descriptive_evidence_tier',
    'bootstrap_power_PP', 'bootstrap_power_PD',
    'bootstrap_valid_PP', 'bootstrap_valid_PD',
    'bootstrap_mc_ci_low_PP', 'bootstrap_mc_ci_high_PP',
    'bootstrap_mc_ci_low_PD', 'bootstrap_mc_ci_high_PD',
    'power_screen_status',
)

# Default reviewer-wide: cross-phenotype comparison at a glance, not all raw
# phenotype-specific covariates. All columns are retained in the audit FULL files.
WIDE_PHENO_FIELDS = (
    'PPMI_q_bh_global', 'PD_PLUS_SU_pvalue',
    'robust_PPMI_ar_q_BH', 'robust_PD_PLUS_SU_ar_q_BH',
    'multiIV_p_q_BH', 'descriptive_evidence_tier',
)
WIDE_PHENO_EXTENDED_FIELDS = (
    'replication_design', 'selected_lag_order', 'PPMI_pvalue',
    'PPMI_q_bh_global', 'PD_pvalue', 'SU_pvalue', 'PD_PLUS_SU_pvalue',
    'robust_PPMI_ar_pvalue', 'robust_PPMI_ar_q_BH',
    'robust_PD_PLUS_SU_ar_pvalue', 'robust_PD_PLUS_SU_ar_q_BH',
    'AR_discoveryFDR_repNominal_support',
    'multiIV_n', 'multiIV_p', 'multiIV_p_q_BH',
    'multiIV_Q_p', 'multiIV_Q_p_q_BH', 'multiIV_I2',
    'LOO_max_p', 'LOO_all_nominal_p05', 'multiIV_LOO_stable',
    'cross_gene_max_r2', 'cross_gene_ld_flag', 'descriptive_evidence_tier',
)
# Original selected-model clinical GReX coefficients, not frozen Stage-I
# genotype-time basis beta_g_*. No coefficient is recomputed or reoriented.
# NAs for lag>selected_lag_order are expected; no zero imputation.
LAG_COHORTS = ('PPMI', 'PD', 'SU', 'PD_PLUS_SU')
LAG_COEF_FIELDS = tuple(
    field for cohort in LAG_COHORTS
    for field in (*[f'{cohort}_coef_x_lag{lag}' for lag in range(1, 5)],
                  f'{cohort}_coef_sum')
)
# Compact WIDE: primary discovery + the pooled PD/SU replication design
# (cohort-specific PD and SU remain in the concise LONG and extended WIDE).
WIDE_LAG_COHORTS = ('PPMI', 'PD_PLUS_SU')
WIDE_LAG_FIELDS = tuple(
    field for cohort in WIDE_LAG_COHORTS
    for field in (*[f'{cohort}_coef_x_lag{lag}' for lag in range(1, 5)],
                  f'{cohort}_coef_sum')
)

WIDE_COMMON_FIELDS = (
    'gene', 'lead_snp', 'phenotypes_significant', 'n_significant_phenotypes',
    'best_evidence_tier_level', 'best_evidence_tier_original_label', 'best_evidence_tier_source_phenotype',
    'bootstrap_power_PP', 'bootstrap_power_PD', 'power_screen_status',
)

# Missing columns are explicitly filled with NA, not fabricated from other
# tests or from the source's arbitrary choice of replication design.
def project_columns(df, fields):
    if not set(KEY).issubset(df.columns):
        raise ValueError('Reviewer table must preserve gene and lead_snp')
    present = [c for c in fields if c in df.columns]
    result = df.loc[:, present].copy()
    missing = [c for c in fields if c not in df.columns]
    for col in missing:
        result[col] = pd.NA
    return result.loc[:, list(fields)]

def reviewer_long(df, extended=False):
    # Column-only projection: use the final FULL master coefficients verbatim.
    # PPMI discovery, PD, SU and pooled PD+SU each retain four lags and coef_sum.
    fields = list(EXTENDED_FIELDS if extended else REVIEW_FIELDS)
    # Keep exposure coefficients beside original cohort discovery/replication
    # P values, rather than hiding the most interpretable data at the far right.
    position = (fields.index('PD_PLUS_SU_pvalue')+1 if 'PD_PLUS_SU_pvalue' in fields
                else len(fields))
    fields[position:position] = list(LAG_COEF_FIELDS)
    return project_columns(df, fields).sort_values(
        ['phenotype', 'gene', 'lead_snp'], kind='stable'
    ).reset_index(drop=True)

def reviewer_wide(df, extended=False):
    phen_fields = ((WIDE_PHENO_EXTENDED_FIELDS + LAG_COEF_FIELDS) if extended
                   else (WIDE_PHENO_FIELDS + WIDE_LAG_FIELDS))
    cols = list(WIDE_COMMON_FIELDS)
    cols += [f'significant_{ph}' for ph in PHENOS]
    for ph in PHENOS:
        cols += [f'{ph}_{field}' for field in phen_fields]
    return project_columns(df, cols).sort_values(
        ['n_significant_phenotypes', 'gene', 'lead_snp'],
        ascending=[False, True, True], kind='stable', na_position='last'
    ).reset_index(drop=True)

FIELD_EXPLANATIONS = {
    'gene': 'Gene symbol; not unique when multiple lead variants occur',
    'lead_snp': 'Exact genetic instrument (variant ID); combine with gene for the IV-gene key',
    'phenotype': 'updrs1, updrs2, updrs3; MoCA excluded',
    'replication_design': 'Copied unchanged from Stage-II final master (e.g. pooled_PD_PLUS_SU)',
    'selected_lag_order': 'Lag order selected by original Stage-II procedure',
    'PPMI_pvalue': 'PPMI discovery joint lag-block association p-value',
    'PPMI_q_bh_global': 'PPMI original global BH adjusted q-value (not recomputed after selection)',
    'PD_pvalue': 'PDBP PD cohort nominal association p-value at designated lag, when available',
    'SU_pvalue': 'SU cohort nominal association p-value at designated lag, when available',
    'PD_PLUS_SU_pvalue': 'Pooled PDBP PD+SU nominal association p-value',
    'PPMI_n_participants': 'Number of participants in the selected PPMI model',
    'PD_PLUS_SU_n_participants': 'Number of participants in the pooled replication model',
    'stage1_ppmi_tqtl_p': 'Stage-I PPMI overall temporal tQTL test p-value',
    'stage1_pdbp_tqtl_p': 'Stage-I PDBP overall temporal tQTL replication p-value',
    'stage1_ppmi_first_stage_F': 'Descriptive PPMI first-stage strength F statistic; NOT an eligibility gate',
    'stage1_pdbp_first_stage_F': 'Descriptive PDBP first-stage strength F statistic; NOT an eligibility gate',
    'robust_PPMI_ar_q_BH': 'Stage-II discovery AR-type robustness BH q-value as stored in final master',
    'robust_PD_PLUS_SU_ar_q_BH': 'Stage-II pooled replication AR-type robustness BH q-value as stored in final master',
    'AR_discoveryFDR_repNominal_support': 'Original AR robustness support label; missing is not treated as false',
    'multiIV_n': 'Number of Stage-I eligible independent IVs evaluated for this gene',
    'multiIV_p_q_BH': 'Original BH q-value for common-lag gene-level multi-IV analysis',
    'multiIV_Q_p_q_BH': 'Original BH q-value for Cochran Q heterogeneity check',
    'multiIV_I2': 'Cochran I-squared heterogeneity (fraction 0..1)',
    'multiIV_LOO_stable': 'Original leave-one-IV-out stability annotation',
    'cross_gene_max_r2': 'Maximum observed LD r-squared between prioritised IVs of different genes',
    'descriptive_evidence_tier': 'Original Stage-II descriptive evidence tier; never reclassified by cross summary',
    'bootstrap_power_PP': 'PPMI conditional donor-bootstrap tQTL re-detection rate (fraction 0..1)',
    'bootstrap_power_PD': 'PDBP conditional donor-bootstrap tQTL re-detection rate (fraction 0..1)',
    'power_screen_status': 'pass_both_80pct, below_80pct or not_evaluable; not prospective power',
}


def reviewer_power_audit(df):
    fields = list(WIDE_COMMON_FIELDS)
    fields += [f'significant_{ph}' for ph in PHENOS]
    fields += ['bootstrap_valid_PP','bootstrap_valid_PD',
               'bootstrap_mc_ci_low_PP','bootstrap_mc_ci_high_PP',
               'bootstrap_mc_ci_low_PD','bootstrap_mc_ci_high_PD',
               'power_evaluable_both','tqtl_power_both_ge80',
               'power80_borderline_uncertainty']
    return project_columns(df, fields)

def machine_candidate(stage, clinical, out=None):
    # All stage columns stay intact, including artifact_id, basis_json, beta_g_*.
    # Add only essential phenotype evidence rather than every intermediate model column.
    clinical_cols=reviewer_wide(clinical, extended=True)
    out=stage.merge(clinical_cols,on=KEY,how='inner',validate='one_to_one')
    required={'gene','lead_snp','artifact_id','basis_json'}
    if not required.issubset(out.columns) or not any(c.startswith('beta_g_') for c in out):
        raise ValueError('Machine candidate lost frozen Stage-I model parameters')
    return out

def write_reviewer_dictionary(out):
    rows = [{'field': field, 'description': FIELD_EXPLANATIONS.get(field,
             'Original Stage-II statistical summary / cohort-specific evidence; see FULL audit for definition')}
            for field in EXTENDED_FIELDS + LAG_COEF_FIELDS]
    for row in rows:
        f=row['field']
        if f.endswith('_coef_sum') and f.split('_coef_sum')[0] in LAG_COHORTS:
            row['description'] = 'Original sum of selected-model lagged GReX coefficients from this cohort; not the joint-lag test statistic or a causal direction test'
        elif any(f.startswith(co+'_coef_x_lag') for co in LAG_COHORTS):
            row['description'] = 'Original selected-model coefficient for this specific lagged GReX term, taken verbatim from Stage-II FULL master; missing if lag is not included/estimated'
    pd.DataFrame(rows).to_csv(out/'PD_three_phenotypes_review_table_dictionary.csv', index=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--out-dir',required=True)
    parser.add_argument('--power-results')
    parser.add_argument('--min-valid-bootstrap',type=int,default=800)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.prepare_only and args.power_results:
        raise ValueError('--prepare-only and --power-results cannot both be used')
    if not args.prepare_only and not args.power_results:
        raise ValueError('Specify --prepare-only or --power-results')
    root=Path(args.root);out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    sources={};tables=[]
    for ph in PHENOS:
        d,src=load_master(root,ph)
        tables.append(d);sources[ph]=str(src)
    full=pd.concat(tables,ignore_index=True,sort=False)
    if full.duplicated(['phenotype']+KEY).any():raise ValueError('Repeated phenotype/gene/SNP in final masters')
    full=full.sort_values(['phenotype']+KEY).reset_index(drop=True)
    wide=make_pair_wide(full)
    # This is the unfiltered pair-level evidence table, with all full source statistics.
    stagefile=root/'single_iv_causal/iv_validation/single_iv_primary_eligible.csv'
    if not stagefile.exists():raise FileNotFoundError(stagefile)
    stage=clean_key(pd.read_csv(stagefile,low_memory=False,dtype={k:'string' for k in KEY}),str(stagefile))
    if 'iv_primary_eligible' in stage:
        stage=stage.loc[yes(stage['iv_primary_eligible'])].copy()
    if not {'artifact_id','basis_json'}.issubset(stage):
        raise ValueError(f'{stagefile} missing artifact_id / basis_json')
    if not any(c.startswith('beta_g_') for c in stage):
        raise ValueError(f'{stagefile} missing frozen beta_g_* columns')
    if stage.duplicated(KEY).any():raise ValueError(f'{stagefile} duplicate gene + SNP')
    st = stage.merge(wide,on=KEY,how='inner',validate='one_to_one')
    if len(st) != len(wide):
        absent=wide[KEY].merge(st[KEY],on=KEY,how='left',indicator=True)
        raise ValueError(f'{len(wide)-len(st)} clinical pairs lack eligible Stage-I record: '+
                         str(absent.loc[absent._merge.eq('left_only'),KEY].head(8).to_dict('records')))
    # Do not permit silent artifact mismatch when the source masters carry artifact IDs.
    for ph in PHENOS:
        f=f'{ph}_artifact_id'
        if f in st:
            both=st[f].notna()
            if ((st.loc[both,f].astype(str)!=st.loc[both,'artifact_id'].astype(str))).any():
                raise ValueError(f'{ph} artifact_id mismatch vs Stage-I manifest')
    power=None
    if args.power_results:
        power=read_power(Path(args.power_results),args.min_valid_bootstrap)
        wide=attach_power(wide,power,args.min_valid_bootstrap)
        # Save the complete audit before any failure on the power gate.
        reviewer_power_audit(wide).to_csv(out/'PD_three_phenotypes_IV_gene_power_audit_ALL.csv',index=False)
        wide.to_csv(out/'PD_three_phenotypes_IV_gene_power_audit_FULL.csv',index=False)
        if not wide['power_evaluable_both'].any():
            raise RuntimeError('No clinical candidate is evaluable in both cohorts; see IV_gene_power_audit_ALL.csv')
    # Generate complete long and wide evidence tables with power annotations (if supplied).
    long=full.copy()
    if power is not None:
        append_cols = KEY+[c for c in wide if c.startswith(('bootstrap_','power_','tqtl_power_'))]
        long=long.merge(wide[append_cols].drop_duplicates(KEY),on=KEY,how='left',validate='many_to_one')
    # Three tiers of output: reviewer-short, statistical extended, and complete audit.
    # Preserve all original FULL statistics and all Power annotations without loss.
    long.to_csv(out/'Supplement_PD_IV_gene_ALL_statistics_3phenotypes.csv',index=False)
    wide.to_csv(out/'PD_three_phenotypes_full_evidence_WIDE_AUDIT.csv',index=False)
    reviewer_long(long).to_csv(out/'Supplement_PD_IV_gene_publication_3phenotypes.csv',index=False)
    reviewer_long(long,extended=True).to_csv(out/'PD_three_phenotypes_robustness_evidence_LONG.csv',index=False)
    reviewer_wide(wide).to_csv(out/'PD_three_phenotypes_publication_wide.csv',index=False)
    reviewer_wide(wide,extended=True).to_csv(out/'PD_three_phenotypes_robustness_evidence_WIDE.csv',index=False)
    write_reviewer_dictionary(out)
    # Original tier labels have intentionally NOT been recomputed from filtered Power subset.
    tier_rows=full.groupby(['phenotype','descriptive_evidence_tier'],dropna=False).size().reset_index(name='n_IV_gene_pairs') if 'descriptive_evidence_tier' in full else pd.DataFrame(columns=['phenotype','descriptive_evidence_tier','n_IV_gene_pairs'])
    tier_rows.to_csv(out/'PD_three_phenotypes_evidence_tier_counts_UNFILTERED.csv',index=False)
    reviewer_wide(make_pair_wide(full)).to_csv(out/'PD_three_phenotypes_IV_gene_pairs_UNFILTERED.csv',index=False)
    stage_unfiltered=machine_candidate(stage,make_pair_wide(full))
    stage_unfiltered.to_csv(out/'PD_three_phenotypes_stage1_candidates_UNFILTERED.csv',index=False)
    aggregate_genes(make_pair_wide(full)).to_csv(out/'PD_three_phenotypes_gene_summary_UNFILTERED.csv',index=False)
    stats={'phenotypes':list(PHENOS),'excluded_phenotypes':['moca'],
           'source_masters':sources,'source_stage1':str(stagefile),
           'n_unfiltered_pairs':len(st),'n_unfiltered_genes':int(st.gene.nunique()),
           'by_phenotype':{ph:int((full.phenotype==ph).sum()) for ph in PHENOS},
           'source_evidence_tiers':tier_rows.to_dict('records'),
           'power_gate_applied':power is not None,'n_power_qualified_pairs':None}
    if power is not None:
        stats['power_results']=str(args.power_results)
        stats['min_valid_bootstrap']=args.min_valid_bootstrap
        stats['power_screen_status_counts']=wide.power_screen_status.value_counts(dropna=False).to_dict()
        qualified=wide.loc[wide.tqtl_power_both_ge80].copy()
        reviewer_wide(qualified).to_csv(out/'PD_three_phenotypes_IV_gene_pairs_POWER80.csv',index=False)
        qualified.to_csv(out/'PD_three_phenotypes_IV_gene_pairs_POWER80_FULL_AUDIT.csv',index=False)
        passed_long=long.merge(qualified[KEY],on=KEY,how='inner',validate='many_to_one')
        reviewer_long(passed_long).to_csv(out/'Supplement_PD_IV_gene_publication_3phenotypes_POWER80.csv',index=False)
        reviewer_long(passed_long,extended=True).to_csv(out/'Supplement_PD_IV_gene_statistics_extended_3phenotypes_POWER80.csv',index=False)
        passed_long.to_csv(out/'Supplement_PD_IV_gene_ALL_statistics_3phenotypes_POWER80.csv',index=False)
        qualified.groupby(['best_evidence_tier_level','best_evidence_tier_original_label'],dropna=False).size().reset_index(name='n_IV_gene_pairs').to_csv(out/'PD_three_phenotypes_evidence_tier_counts_POWER80.csv',index=False)
        stats['n_power_qualified_pairs']=len(qualified)
        stats['n_power_qualified_genes']=int(qualified.gene.nunique())
        # Write candidate files atomically once all joins and checks have succeeded.
        if qualified.empty:
            for name in ('PD_validated_IV_gene_stage1_candidates.csv','PD_three_phenotypes_IV_gene_pairs.csv','PD_three_phenotypes_gene_summary.csv'):
                (out/name).unlink(missing_ok=True)
            raise RuntimeError('Zero clinical IV pairs passed both PP and PD >=80%; do not reuse old downstream files')
        # Keep a complete audit of the old all-columns join; emit a compact
        # machine-safe manifest for the MRI/CSF entrypoints.
        stqualified_full=stage.merge(qualified,on=KEY,how='inner',validate='one_to_one')
        stqualified_full.to_csv(out/'PD_validated_IV_gene_stage1_candidates_FULL_AUDIT.csv',index=False)
        stqualified=machine_candidate(stage,qualified)
        if len(stqualified)!=len(qualified):raise ValueError('Qualified pair missing Stage-I parameters')
        for name,table in (("PD_validated_IV_gene_stage1_candidates.csv",stqualified),
                           ("PD_three_phenotypes_IV_gene_pairs.csv",qualified),
                           ("PD_three_phenotypes_gene_summary.csv",aggregate_genes(qualified))):
            target=out/name
            tmp=target.with_suffix(target.suffix+'.tmp')
            table.to_csv(tmp,index=False)
            tmp.replace(target)
    stats['reviewer_tables'] = {'short_long_n_columns':len(REVIEW_FIELDS)+len(LAG_COEF_FIELDS), 'extended_long_n_columns':len(EXTENDED_FIELDS)+len(LAG_COEF_FIELDS), 'short_wide_n_columns':len(WIDE_COMMON_FIELDS)+len(PHENOS)+len(PHENOS)*(len(WIDE_PHENO_FIELDS)+len(WIDE_LAG_FIELDS)), 'full_raw_columns':int(len(long.columns)), 'full_wide_columns':int(len(wide.columns)), 'machine_candidate_preserves_frozen_model':True}
    (out/'PD_three_phenotypes_summary.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False,default=str))
    print(json.dumps({k:v for k,v in stats.items() if k!='source_evidence_tiers'},indent=2,ensure_ascii=False,default=str))

if __name__=='__main__':main()
