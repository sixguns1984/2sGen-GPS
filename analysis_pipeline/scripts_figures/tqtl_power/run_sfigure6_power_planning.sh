#!/usr/bin/env bash
# Prospective CTAR 8-df planning power + existing independent replication curve.
# No genome-wide or Stage II model reruns. Launch with nohup.
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
mkdir -p "$FIGROOT/power_analysis_planning_v3" "$FIGROOT/supplementary" "$ROOT/logs"
echo '[STEP 1] CTAR-informed analytic prospective power / HWE finite-sample genotype representation'
"$PYTHON" -u "$HERE/03_ctar_prospective_power.py" \
  --root "$ROOT" --code "$CODE" --figroot "$FIGROOT" \
  --rho "${POWER_RHO:-0.6}" --shape "${POWER_SHAPE:-constant}" \
  --cohort-profile "${POWER_PROFILE:-PD}" \
  --nsim-genotype "${POWER_GENO_DRAWS:-2500}" \
  --seed "${POWER_SEED:-20261029}" \
  --mafs "${POWER_MAFS:-0.05,0.10,0.20,0.30}" \
  --effects "${POWER_EFFECTS:-0.25,0.50,0.75,1.00}" \
  --n-grid "${POWER_N_GRID:-100,150,200,250,300,400,500,700,900,1200,1600,2000,3000,4000,6000}"
echo '[STEP 2] Reuse already-completed PDBP no-replacement sensitivity; create vector PDF'
"$PYTHON" -u "$HERE/04_plot_sfigure6_planning.py" --root "$ROOT" --figroot "$FIGROOT"
echo '[DONE] New Supplementary Figure 6'
