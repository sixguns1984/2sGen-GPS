#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
CODE=${CODE:-$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CORE=$CODE/scripts_figures/tqtl_power/2sGenGPS_CTAR_SFig6_refit_v3
POWER_OUT=${AD_MMSE_POWER_OUT:-$ROOT/manuscript_figures/tqtl_benchmark_v4/AD_MMSE_ctar_power_coefsum_v2}
STD=$ROOT/single_iv_causal/ad_mmse/standard_final
EVIDENCE_OUT=${AD_MMSE_EVIDENCE_OUT:-$STD/posthoc_evidence}
FINAL=$STD/AD_MMSE_FINAL_evidence_graded.csv
export ROOT CODE OUT="$POWER_OUT" PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$ROOT/logs" "$EVIDENCE_OUT"
command=${1:-help};shift || true
case "$command" in
  power-prepare)
    python "$HERE/01_prepare_ad_mmse_power.py" --root "$ROOT" --outdir "$POWER_OUT" --final-file "$FINAL" "$@" ;;
  power-cache)
    bash "$CORE/run_sfigure6.sh" cache-expression "$@" ;;
  power-audit)
    bash "$CORE/run_sfigure6.sh" audit-inputs "$@" ;;
  power-preflight)
    bash "$CORE/run_sfigure6.sh" preflight "$@" ;;
  power-full)
    bash "$CORE/run_sfigure6.sh" full "$@" ;;
  power-merge)
    bash "$CORE/run_sfigure6.sh" merge "$@" ;;
  multiiv)
    python "$HERE/02_ad_mmse_multiiv.py" --root "$ROOT" --n-jobs "${N_JOBS:-8}" --out-dir "$EVIDENCE_OUT" "$@" ;;
  ld)
    MAP_BFILE=${MAP_LD_BFILE:-$ROOT/single_iv_causal/ad_mmse/prep/MAP_common_IVs}
    for ext in .bed .bim .fam; do
      [[ -s "$MAP_BFILE$ext" ]] || { echo "[ERROR] No MAP discovery-cohort LD reference $MAP_BFILE$ext" >&2;exit 1; }
    done
    python "$CODE/scripts_stage2/09_cross_gene_ld_audit.py" \
      --root "$ROOT" --analysis-name ad_mmse \
      --candidate-file "$FINAL" --raw-bfile "$MAP_BFILE" \
      --plink "${PLINK:-/usr/bin/plink1.9}" --window-kb 2000 \
      --report-r2 0.1 --flag-r2 0.1 --out-dir "$EVIDENCE_OUT/ld_audit" "$@" ;;
  publish)
    python "$HERE/03_ad_mmse_publish.py" --root "$ROOT" --final-file "$FINAL" \
      --power-results "$POWER_OUT/locus_bootstrap_results.csv" \
      --multiiv-file "$EVIDENCE_OUT/multi_iv_gene_consistency_summary.csv" \
      --ld-dir "$EVIDENCE_OUT/ld_audit" --out-dir "$EVIDENCE_OUT" \
      --min-valid-bootstrap "${MIN_VALID_BOOTSTRAP:-800}" "$@" ;;
  help|*)
    echo "Usage: bash $0 {power-prepare|power-cache|power-audit|power-preflight|power-full|multiiv|ld|publish}" >&2
    [[ "$command" == help ]] || exit 2;;
esac
