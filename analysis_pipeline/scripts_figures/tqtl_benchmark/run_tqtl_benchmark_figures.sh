#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
CODE=${CODE:-$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SIM=${SIM:-$ROOT/simulation_qtl_benchmark_rho_covariate/simulation_summary.csv}
REAL=${REAL:-$ROOT/qtl/replication_summary.csv}
FIGROOT=${FIGROOT:-$ROOT/manuscript_figures/tqtl_benchmark_v1}
mkdir -p "$FIGROOT"/{main,supplementary,reviewer,tables,logs}
export MPLBACKEND=Agg
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
[[ -s "$SIM" ]] || { echo "[ERROR] Missing simulation summary: $SIM" >&2; exit 2; }
[[ -s "$REAL" ]] || { echo "[ERROR] Missing replication summary: $REAL" >&2; exit 2; }
echo "[1/2] Simulation figures from $SIM"; date -Is
python -u "$HERE/01_simulation_figures.py" --input "$SIM" --out-root "$FIGROOT"
echo "[2/2] PPMI-PDBP real data figures from $REAL"; date -Is
python -u "$HERE/02_realdata_figures.py" --input "$REAL" --out-root "$FIGROOT"
echo "[DONE] Figures saved to $FIGROOT"; date -Is
