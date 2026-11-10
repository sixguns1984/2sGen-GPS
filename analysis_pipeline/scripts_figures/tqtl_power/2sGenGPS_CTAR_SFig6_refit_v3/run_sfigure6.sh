#!/usr/bin/env bash
set -euo pipefail
# One command launcher; the original Stage-I and Stage-II directories are never modified.
ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
CODE=${CODE:-${ROOT}/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:-${ROOT}/manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_refit_v3}
LOG=${ROOT}/logs
mkdir -p "$OUT" "$LOG"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CMD=${1:-help};shift || true
case "$CMD" in
  prepare)
    python "$HERE/01_prepare_real_inputs.py" --root "$ROOT" --outdir "$OUT" \
        --n-pairs "${N_PAIRS:-5000}" --seed "${SEED:-20261030}" \
        --plink "${PLINK:-/usr/bin/plink1.9}" --skip-plink "$@" ;;
  cache-expression)
    python "$HERE/05_cache_selected_expression.py" --root "$ROOT" --outdir "$OUT" "$@" ;;
  audit-inputs)
    python "$HERE/06_audit_stage1_inputs.py" --root "$ROOT" --outdir "$OUT" "$@" ;;
  preflight)
    EXTRA_EXPR=()
    [[ -s "$OUT/expression_selected.parquet" ]] && EXTRA_EXPR=(--expression "$OUT/expression_selected.parquet")
    # Separate output: a quick low-rep audit MUST NOT contaminate the final 1,000-rep results.
    AUDIT="$OUT/_preflight"
    mkdir -p "$AUDIT"
    # preflight is independent of final checkpoint files; safe to restart from scratch
    rm -f "$AUDIT/locus_bootstrap_results.csv" "$AUDIT/locus_failures.csv" "$AUDIT/donor_sample_size_sensitivity.csv"
    python "$HERE/02_run_power.py" --root "$ROOT" --outdir "$AUDIT" \
      --selected-file "$OUT/selected_loci.csv" \
      --pp-raw "$OUT/genotypes_PP.raw" --pd-raw "$OUT/genotypes_PD.raw" \
      --alpha-disc "$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["observed_max_raw_p_of_all_hits"])' "$OUT/selection_audit.json")" \
      --max-pairs "${PREFLIGHT_PAIRS:-30}" --bootstrap-reps "${PREFLIGHT_REPS:-100}" \
      --audit-pairs "${PREFLIGHT_AUDIT_PAIRS:-10}" --sensitivity-reps 0 "${EXTRA_EXPR[@]}" "$@"
    python "$HERE/04_audit_refit.py" --results "$AUDIT/locus_bootstrap_results.csv" \
      --output "$AUDIT/fit_audit.json" ;;
  full|full-parallel)
    # Each worker writes to an independent checkpoint directory.  Defaults to 4
    # CPU workers for a conservative memory footprint. Same JOBS on resume.
    python "$HERE/07_run_parallel.py" run --root "$ROOT" --out "$OUT" \
      --jobs "${POWER_JOBS:-4}" --reps "${BOOTSTRAPS:-1000}" \
      --sensitivity-reps "${SUBSAMPLE_REPS:-100}" \
      --sensitivity-pairs "${SUBSAMPLE_PAIRS:-400}" \
      --audit-pairs "${FULL_AUDIT_PAIRS:-10}" \
      --batch-size "${POWER_BATCH_SIZE:-32}" \
      --backend "${POWER_BACKEND:-fast}" "$@" ;;
  merge)
    python "$HERE/07_run_parallel.py" merge --root "$ROOT" --out "$OUT" \
      --jobs "${POWER_JOBS:-4}" "$@" ;;
  full-serial)
    EXTRA_EXPR=()
    [[ -s "$OUT/expression_selected.parquet" ]] && EXTRA_EXPR=(--expression "$OUT/expression_selected.parquet")
    python "$HERE/02_run_power.py" --root "$ROOT" --outdir "$OUT" \
      --bootstrap-reps "${BOOTSTRAPS:-1000}" --sensitivity-reps "${SUBSAMPLE_REPS:-100}" \
      --sensitivity-pairs "${SUBSAMPLE_PAIRS:-400}" --audit-pairs "${FULL_AUDIT_PAIRS:-10}" --resume "${EXTRA_EXPR[@]}" "$@" ;;
  plot)
    if [[ ! -s "$OUT/locus_bootstrap_results.csv" && -d "$OUT/_shards" ]]; then
      python "$HERE/07_run_parallel.py" merge --root "$ROOT" --out "$OUT" \
        --jobs "${POWER_JOBS:-4}"
    fi
    python "$HERE/04_audit_refit.py" --results "$OUT/locus_bootstrap_results.csv" \
      --output "$OUT/fit_audit.json"
    python "$HERE/03_plot_power.py" --root "$ROOT" --outdir "$OUT" "$@" ;;
  *)
    echo "Usage: bash run_sfigure6.sh {prepare|cache-expression|audit-inputs|preflight|full|full-parallel|full-serial|merge|plot} [flags]";exit 2;;
esac
