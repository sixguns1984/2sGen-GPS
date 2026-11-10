#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
CODE=${CODE:-$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized}
FIGROOT=${FIGROOT:-$ROOT/manuscript_figures/tqtl_benchmark_v4}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MPLBACKEND=cairo
mkdir -p "$FIGROOT/power_analysis_empirical_v4" "$FIGROOT/supplementary" "$ROOT/logs"
echo '[STEP 1/5] Full PPMI/PDBP Stage-I replication (existing result tables, no fits)'
"$PYTHON" -u "$HERE/09_full_cohort_replication.py" --root "$ROOT" --figroot "$FIGROOT"
echo '[STEP 2/5] Empirical CTAR 8-parameter trajectory calibration using real PPMI genotypes, RNA and covariates'
"$PYTHON" -u "$HERE/06_calibrate_real_ctar_effects.py" --root "$ROOT" --code "$CODE" --figroot "$FIGROOT" \
    --n-calibration "${CALIBRATION_N:-260}" --min-valid "${CALIBRATION_MIN_VALID:-35}" \
    --seed "${POWER_SEED:-20261029}"
echo '[STEP 3/5] Fast approximate CTAR power grid with real covariate/visit design'
"$PYTHON" -u "$HERE/07_empirical_ctar_power.py" --root "$ROOT" --code "$CODE" --figroot "$FIGROOT" \
    --mafs "${POWER_MAFS:-.05,.10,.20,.30}" \
    --n-grid "${POWER_N_GRID:-100,150,200,250,300,400,500,700,900,1200,1400,1600,2000,2600,3500,5000,7000}" \
    --rho "${POWER_RHO:-.6}" --profile "${POWER_PROFILE:-PD}" \
    --draws "${POWER_GENO_DRAWS:-1500}" --seed "${POWER_SEED:-20261029}"
if [[ "${RUN_SUBSAMPLING:-1}" == "1" ]];then
  echo '[STEP 4/5] Representatively sampled PPMI candidate pairs; without-replacement PDBP CTAR refits'
  # v4_subsampling is deliberately isolated from old v2/v3 checkpoints.
  "$PYTHON" -u "$HERE/10_ctar_subsample_existing_model.py" \
    --root "$ROOT" --out-root "$FIGROOT/v4_subsampling" \
    --n-pairs "${N_SUBSAMPLE_PAIRS:-300}" --resamples "${SUBSAMPLE_REPS:-8}" \
    --fractions "${SUBSAMPLE_FRACTIONS:-0.5,0.75,0.9,1.0}" \
    --rho-points "${RHO_POINTS:-100}" --device "${DEVICE:-cpu}" \
    --seed "${POWER_SEED:-20261029}"
  "$PYTHON" -u "$HERE/11_summarize_subsamples.py" --root "$ROOT" --out-root "$FIGROOT/v4_subsampling"
else
  echo '[STEP 4/5] Subsampling skipped by RUN_SUBSAMPLING=0; full-PDBP replication still shown'
fi
echo '[STEP 5/5] Export vector PDF / SVG and optional PNG'
"$PYTHON" -u "$HERE/08_plot_sfigure6_empirical.py" --figroot "$FIGROOT" --max-n "${PLOT_N_MAX:-5000}"
echo '[COMPLETE] Empirically calibrated CTAR power and Supplementary Figure 6' 
