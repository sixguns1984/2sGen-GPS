#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}"
CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
exec python "$CODE/scripts_amp_pd/06d_stage1_final_tqtl_vqtl_sensitivity.py" \
  --root "$ROOT" \
  --n-jobs "${N_JOBS:-8}" \
  --genes-per-task "${GENES_PER_TASK:-10}" \
  --n-folds "${N_FOLDS:-5}" \
  --fold-seed "${FOLD_SEED:-1}" \
  --min-fluctuation-visits "${MIN_VISITS:-3}" \
  --min-genotype-group-n "${MIN_GROUP_N:-10}" \
  "$@"
