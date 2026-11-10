#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}"
CODE="${CODE:-$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized}"
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
echo '[RE-GRADE] Existing FINAL only; NO model, genotype, PCA, GReX or RF rerun.'
python -u "$CODE/scripts_stage2/21_regrade_ad_mmse_existing_final.py" --root "$ROOT" "$@"
