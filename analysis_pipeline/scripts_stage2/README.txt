Compare the PD Stage-II lag-1 vs summed-lag directional gate, WITHOUT model refitting.

1) Copy compare_coefsum_vs_lag1_power80.py to your server, e.g.:
   /public2/data/luojunfeng/2sGenGPS_revised_chatgpt/compare_coefsum_vs_lag1_power80.py

2) Run:
   ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
   python "$ROOT/compare_coefsum_vs_lag1_power80.py" --root "$ROOT"

Input Stage-II files per phenotype:
  $ROOT/single_iv_causal/{updrs1,updrs2,updrs3}/temporal_causal_agecubic_full_followup_fast/discovery_significant_IV_gene_lag_models.csv
  $ROOT/single_iv_causal/{updrs1,updrs2,updrs3}/temporal_causal_agecubic_full_followup_fast/replication_PD_PLUS_SU_all_lags_annotated.csv.gz
Power input:
  $ROOT/manuscript_figures/tqtl_benchmark_v4/clinical_IV_ctar_power_v5/locus_bootstrap_results.csv

Decision: among exact discovery-significant lags with PD+SU P < 0.05,
select best-BIC lag PER SNP/gene and only then apply lag1 or coef_sum sign consistency.
Never choose different lag models under the two policies.

Outputs:
  cross_phenotype_summary_lag1/coefsum_direction_comparison/direction_comparison_summary.csv
  .../ALL_predirection_sameBIC_with_power.csv
  .../coefsum_direction_POWER80_confirmed.csv
  .../coefsum_only_new_candidates.csv
  .../coefsum_only_new_need_power.csv
  .../coefsum_only_new_IV_gene_for_power.csv
  .../current_vs_reconstructed_discrepancies.csv  [only if old counts disagree]

Critical: Previously tested clinical Power results DO NOT necessarily include new SNP-gene
pairs that failed the old lag-1 direction rule. The confirmed POWER80 count is a lower
bound until the new IV-gene set gets the same PPMI/PDBP donor-bootstrap evaluation.

If the new source files are different from the ones used for the previous 116-row table,
this script prints an AUDIT WARNING and writes an explicit mismatch file rather than
silently claiming the counts are identical.
