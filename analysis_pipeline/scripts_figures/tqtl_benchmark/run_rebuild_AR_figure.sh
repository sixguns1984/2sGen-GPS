#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
CODE=${CODE:-$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized}
FIGROOT=${FIGROOT:-$ROOT/manuscript_figures/tqtl_benchmark_v4}
SELECTION=${SELECTION:-fixed}
SEARCH_PER_QUANTILE=${SEARCH_PER_QUANTILE:-80}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$FIGROOT/supplementary" "$FIGROOT/tables"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -u "$HERE/04_rep_tqtl_ar_supplementary.py" \
    --root "$ROOT" --code "$CODE" --out-root "$FIGROOT" \
    --selection "$SELECTION" --search-per-quantile "$SEARCH_PER_QUANTILE"
