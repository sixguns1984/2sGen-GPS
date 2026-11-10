#!/usr/bin/env bash
set -euo pipefail

CODE=${CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
QTL_LABEL=${QTL_LABEL:-submitted_followup_ctar_mptga}
PPMI_BFILE=${PPMI_BFILE:-$ROOT/data/genotype/PP/amp_pd_PP_qc}
PLINK=${PLINK:-/usr/bin/plink1.9}
LD_R2=${LD_R2:-0.1}
STRENGTH_JOBS=${STRENGTH_JOBS:-4}
STRENGTH_BATCH_GENES=${STRENGTH_BATCH_GENES:-8}
PDBP_QTL_TABLE=${PDBP_QTL_TABLE:-}

cd "$CODE"
mkdir -p "$ROOT/logs"

DIR_ARGS=(--root "$ROOT" --qtl-label "$QTL_LABEL")
if [[ -n "$PDBP_QTL_TABLE" ]]; then DIR_ARGS+=(--pdbp-qtl-table "$PDBP_QTL_TABLE"); fi
python scripts_stage2/00_trajectory_direction.py "${DIR_ARGS[@]}"

CAND_ARGS=(--root "$ROOT" --qtl-label "$QTL_LABEL" --ppmi-bfile "$PPMI_BFILE" --plink "$PLINK" --ld-r2-threshold "$LD_R2")
if [[ -n "$PDBP_QTL_TABLE" ]]; then CAND_ARGS+=(--pdbp-qtl-table "$PDBP_QTL_TABLE"); fi
python scripts_stage2/01_prepare_temporal_iv_candidates.py "${CAND_ARGS[@]}"

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python scripts_stage2/02_compute_iv_strength.py \
  --root "$ROOT" \
  --qtl-label "$QTL_LABEL" \
  --n-jobs "$STRENGTH_JOBS" \
  --blas-threads 1 \
  --batch-genes "$STRENGTH_BATCH_GENES"

printf '\nStage-II shared temporal-IV preparation complete.\n'
printf 'Primary IV table: %s\n' "$ROOT/single_iv_causal/iv_validation/single_iv_primary_eligible.csv"
printf 'Strength table:   %s\n' "$ROOT/single_iv_causal/iv_strength_all/all_IV_first_stage_strength.csv"
