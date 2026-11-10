#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
CODE=${CODE:-$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized}
MRI_FILE=${MRI_FILE:-/public2/data/luojunfeng/2sTGCS_binary/outcome/DBM/confound_pass_snp/MRI_DBM.csv}
HIT_FILE=${HIT_FILE:-$ROOT/single_iv_causal/mri_dbm_lag1/pc1/MRI_DBM_lag1_global_FDR05.csv}
OUT=${OUT:-$ROOT/manuscript_figures/tqtl_benchmark_v4/mri_dbm_examples}
mkdir -p "$OUT"
python "$HERE/01_plot_mri_dbm_examples.py" \
  --root "$ROOT" --mri-file "$MRI_FILE" --hit-file "$HIT_FILE" \
  --outdir "$OUT" "$@"
