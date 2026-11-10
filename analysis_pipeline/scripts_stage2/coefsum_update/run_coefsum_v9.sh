#!/usr/bin/env bash
# Existing Stage-II fitted models reused. This script only refinalizes selected
# IVs, refreshes post-selection robustness, and recomputes PP/PD bootstrap power.
set -euo pipefail
ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
CODE=${CODE:-$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized}
PDBASE=$ROOT/single_iv_causal
SUMMARY=$PDBASE/cross_phenotype_summary_lag1
POWER_RUN=$CODE/scripts_figures/tqtl_power/ctar_replication_v5/run_replicated_power.sh
AD_RUN=$CODE/scripts_stage2/ad_mmse_posthoc/run_ad_mmse_posthoc.sh
PD_POWER_OUT=${PD_POWER_OUT:-$ROOT/manuscript_figures/tqtl_benchmark_v4/clinical_IV_ctar_power_coefsum_v6}
AD_POWER_OUT=${AD_MMSE_POWER_OUT:-$ROOT/manuscript_figures/tqtl_benchmark_v4/AD_MMSE_ctar_power_coefsum_v2}
N_JOBS=${N_JOBS:-8}
mkdir -p "$ROOT/logs"
cmd=${1:-help}; shift || true
case "$cmd" in
  pd-postprocess)
    for ph in updrs1 updrs2 updrs3; do
      case "$ph" in
        updrs1) outcome=mds_updrs_part_i_summary_score;;
        updrs2) outcome=mds_updrs_part_ii_summary_score;;
        updrs3) outcome=mds_updrs_part_iii_summary_score;;
      esac
      cov=/public2/data/luojunfeng/2sTGCS_binary/outcome/$ph/updrs.cov.csv
      std=$PDBASE/$ph/standard_final
      [[ -s "$cov" ]] || { echo "[ERROR] Missing existing phenotype: $cov" >&2;exit 2; }
      python "$CODE/scripts_stage2/07_finalize_replicated_candidates.py" \
        --root "$ROOT" --analysis-name "$ph" --replication-cohorts PD,SU --replication-pooled-groups PD+SU
      OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python "$CODE/scripts_stage2/08_candidate_robustness.py" \
        --root "$ROOT" --analysis-name "$ph" --phenotype-cov "$cov" \
        --outcome-col "$outcome" --time-varying-covariates upd23a_medication_for_pd \
        --use-genotype-pcs --adjust-current-age-cubic --no-require-stage1-followup-support \
        --min-participants 50 --n-jobs "$N_JOBS"
      python "$CODE/scripts_stage2/09_cross_gene_ld_audit.py" \
        --root "$ROOT" --analysis-name "$ph" \
        --candidate-file "$std/${ph}_candidate_exact_pairs_clean_stage1_input.csv" \
        --raw-bfile "${PPMI_BFILE:-$ROOT/data/genotype/PP/amp_pd_PP_qc}" \
        --plink "${PLINK:-/usr/bin/plink1.9}" --window-kb 2000 --report-r2 0.01 --flag-r2 0.10 \
        --out-dir "$std/ld_audit"
      python "$CODE/scripts_stage2/10_assemble_final_master.py" \
        --root "$ROOT" --analysis-name "$ph" --ld-dir "$std/ld_audit" --ld-threshold 0.10
    done
    ;;
  pd-power-prepare|pd-power-cache|pd-power-audit|pd-power-preflight|pd-power-full|pd-power-merge)
    case "$cmd" in
      pd-power-prepare) sub=prepare;;
      pd-power-cache) sub=cache-expression;;
      pd-power-audit) sub=audit-inputs;;
      pd-power-preflight) sub=preflight;;
      pd-power-full) sub=full;;
      pd-power-merge) sub=merge;;
    esac
    POWER_MODE=phenotype-final POWER_OUT="$PD_POWER_OUT" bash "$POWER_RUN" "$sub" "$@"
    ;;
  pd-publish)
    [[ -s "$PD_POWER_OUT/locus_bootstrap_results.csv" ]] || { echo "[ERROR] Missing PD coef_sum clinical Power: $PD_POWER_OUT" >&2;exit 2; }
    python "$CODE/scripts_amp_pd/summarize_pd_three_phenotypes.py" \
      --root "$ROOT" --out-dir "$SUMMARY" \
      --power-results "$PD_POWER_OUT/locus_bootstrap_results.csv" --min-valid-bootstrap "${MIN_VALID_BOOTSTRAP:-800}"
    ;;
  ad-finalize)
    python "$CODE/scripts_stage2/20_finalize_ad_mmse.py" --root "$ROOT" --n-jobs "$N_JOBS" "$@"
    ;;
  ad-power-prepare|ad-power-cache|ad-power-audit|ad-power-preflight|ad-power-full|ad-power-merge)
    case "$cmd" in
      ad-power-prepare) sub=power-prepare;;
      ad-power-cache) sub=power-cache;;
      ad-power-audit) sub=power-audit;;
      ad-power-preflight) sub=power-preflight;;
      ad-power-full) sub=power-full;;
      ad-power-merge) sub=power-merge;;
    esac
    AD_MMSE_POWER_OUT="$AD_POWER_OUT" bash "$AD_RUN" "$sub" "$@"
    ;;
  ad-multiiv|ad-ld|ad-publish)
    sub=${cmd#ad-}
    AD_MMSE_POWER_OUT="$AD_POWER_OUT" N_JOBS="$N_JOBS" bash "$AD_RUN" "$sub" "$@"
    ;;
  audit)
    python "$CODE/scripts_stage2/coefsum_update/verify_coefsum_pipeline.py" --root "$ROOT" "$@"
    ;;
  help|*)
    cat <<EOF
Usage:
  bash $0 pd-postprocess
  bash $0 pd-power-prepare|pd-power-cache|pd-power-audit|pd-power-preflight|pd-power-full|pd-power-merge|pd-publish
  bash $0 ad-finalize|ad-power-prepare|ad-power-cache|ad-power-audit|ad-power-preflight|ad-power-full|ad-power-merge|ad-multiiv|ad-ld|ad-publish
  bash $0 audit
EOF
    [[ "$cmd" == help ]] || exit 2;;
esac
