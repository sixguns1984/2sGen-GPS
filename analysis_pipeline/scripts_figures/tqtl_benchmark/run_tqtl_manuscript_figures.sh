#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
CODE=${CODE:-$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized}
FIGROOT=${FIGROOT:-$ROOT/manuscript_figures/tqtl_benchmark_v4}
SIM=${SIM:-$ROOT/simulation_qtl_benchmark_rho_covariate/simulation_summary.csv}
REAL=${REAL:-$ROOT/qtl/replication_summary.csv}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$FIGROOT"/{main,supplementary,tables,logs}
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
[[ -s "$SIM" ]] || { echo "Missing simulation file: $SIM" >&2; exit 2; }
[[ -s "$REAL" ]] || { echo "Missing replication file: $REAL" >&2; exit 2; }
echo "[1/2] Main Figure 1: a-d real, e-g covariate-adjusted simulation"; date -Is
python -u "$HERE/03_manuscript_figure1.py" --real "$REAL" --simulation "$SIM" --out-root "$FIGROOT"
echo "[2/2] Real verified tQTL: CTAR rho distribution and three observed expression trajectories"; date -Is
python -u "$HERE/04_rep_tqtl_ar_supplementary.py" --root "$ROOT" --code "$CODE" --out-root "$FIGROOT" \
  --search-per-quantile "${SEARCH_PER_QUANTILE:-40}"
echo "[DONE] $FIGROOT"; date -Is
