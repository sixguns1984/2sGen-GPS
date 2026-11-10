#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
CODE=${CODE:-$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized}
FIGROOT=${FIGROOT:-$ROOT/manuscript_figures/ctar_sample_cohorts_v1}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u "$HERE/01_make_cohort_figures.py" --root "$ROOT" --code "$CODE" --out-root "$FIGROOT" --csf-analysis-name "${CSF_ANALYSIS_NAME:-csf_olink_product_adjusted_D01_preferred_v3}" "$@"
