#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}"
CODE="${CODE:-$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized}"
N_JOBS="${N_JOBS:-12}"
BASE="$ROOT/single_iv_causal/ad_mmse"
SOURCE="$BASE/temporal_causal_agecubic_full_followup_fast"
if [[ ! -s "$SOURCE/discovery_all_IV_gene_lag_models.csv.gz" || ! -s "$SOURCE/replication_ADNI_all_lags_for_discovery_significant_IV_gene_pairs.csv.gz" ]]; then
  echo '[ERROR] Completed MAP/ADNI original Stage2 model CSVs missing. Cannot finalize.' >&2
  exit 1
fi
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
echo "[FINALIZE] Existing run only: $SOURCE"
echo "[FINALIZE] No new IV QC, no new MAP/ADNI screening, no genotype PCA or GReX prediction."
echo "[FINALIZE] Target: $BASE/standard_final; N_JOBS=$N_JOBS"
python -u "$CODE/scripts_stage2/20_finalize_ad_mmse.py" \
    --root "$ROOT" --analysis-name ad_mmse --n-jobs "$N_JOBS" "${@}"
