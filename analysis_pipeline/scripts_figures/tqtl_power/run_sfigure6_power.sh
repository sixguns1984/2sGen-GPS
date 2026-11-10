#!/usr/bin/env bash
# New independent-replication power/sensitivity SFig6. Assumes launch by nohup in user's shell.
set -euo pipefail
ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
CODE=${CODE:-$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized}
FIGROOT=${FIGROOT:-$ROOT/manuscript_figures/tqtl_benchmark_v4}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}
N_PAIRS=${N_PAIRS:-100}
RESAMPLES=${RESAMPLES:-24}
FRACTIONS=${FRACTIONS:-0.5,0.75,0.9,1.0}
SEED=${SEED:-20261029}
RHO_POINTS=${RHO_POINTS:-100}
DEVICE=${DEVICE:-cpu}
mkdir -p "$FIGROOT/power_analysis_independent_v2/checkpoints" "$FIGROOT/supplementary" "$ROOT/logs"
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MPLBACKEND=cairo
printf '[POWER V2] PPMI-only candidate selection; PDBP without-replacement subsampling.\n'
"$PYTHON" -u "$HERE/01_ctar_independent_subsample.py" --root "$ROOT" --out-root "$FIGROOT" \
  --n-pairs "$N_PAIRS" --resamples "$RESAMPLES" --fractions "$FRACTIONS" \
  --seed "$SEED" --device "$DEVICE" --rho-points "$RHO_POINTS"
printf '[PLOT] Generate manuscript vector figure\n'
"$PYTHON" -u "$HERE/02_plot_sfigure6_independent.py" --root "$ROOT" --out-root "$FIGROOT"
