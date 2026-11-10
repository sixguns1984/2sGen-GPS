#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
CODE=${CODE:-$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized}
OUT=${POWER_OUT:-$ROOT/manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_REPLICATED_v5}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}
for fn in locus_bootstrap_results.csv matched_pair_power.csv power_summary.json; do
    [[ -s "$OUT/$fn" ]] || { echo "Missing completed v5 result: $OUT/$fn" >&2; exit 2; }
done
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
echo "[PLOT ONLY] Use finalized CTAR_REPLICATED_v5 directory: $OUT"
"$PYTHON" -u "$HERE/03_plot_replication_conditioned.py" --root "$ROOT" --outdir "$OUT" --plot-only "$@"
echo "[PLOT ONLY] SVG/PDF/PNG updated in: $OUT"
