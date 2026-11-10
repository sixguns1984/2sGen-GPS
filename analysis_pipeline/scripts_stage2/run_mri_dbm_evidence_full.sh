#!/usr/bin/env bash
set -euo pipefail
MODE=${1:-all}
ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
CODE=${CODE:-$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized}
MRI_FILE=${MRI_FILE:-/public2/data/luojunfeng/2sTGCS_binary/outcome/DBM/confound_pass_snp/MRI_DBM.csv}
COMPONENT=${COMPONENT:-1}
N_JOBS=${N_JOBS:-8}
PLINK=${PLINK:-/usr/bin/plink1.9}
RAW_BFILE=${RAW_BFILE:-/public/labdata/public_data/AMP_PD_WGS/plink/all_chrs_merged}
MAIN=$ROOT/single_iv_causal/mri_dbm_lag1
BASE=$MAIN/pc${COMPONENT}
OUT=$BASE/evidence_v1
CLINICAL=$ROOT/single_iv_causal/cross_phenotype_summary_lag1/PD_three_phenotypes_publication_wide.csv
ALL_CAND=$OUT/MRI_hit_genes_ALL_eligible_stage1_IVs.csv
AUX_ANALYSIS=mri_dbm_lag1_multiiv
AUXDIR=$ROOT/single_iv_causal/$AUX_ANALYSIS
COV=$MAIN/mri_visit_covariates.csv
mkdir -p "$OUT" "$ROOT/logs"
if [[ "$MODE" != publish ]]; then
for f in "$BASE/MRI_DBM_lag1_global_FDR05.csv" "$CLINICAL" "$MAIN/single_iv_genetic_component_manifest.json"; do
  [[ -s "$f" ]] || { echo "[ERROR] Required source missing: $f" >&2; exit 1; }
done
fi
if [[ $MODE == prepare || $MODE == all ]]; then
  python "$CODE/scripts_stage2/16_mri_prepare_all_stage1_ivs.py" --root "$ROOT" --component "$COMPONENT" --out-dir "$OUT"
  python "$CODE/scripts_stage2/14a_prepare_mri_prediction_cov.py" --mri-file "$MRI_FILE" --out "$COV"
  python "$CODE/scripts_stage2/03_prepare_phenotype_genotype.py" --root "$ROOT" --analysis-name "$AUX_ANALYSIS" --phenotype-cov "$COV" --candidate-file "$ALL_CAND" --raw-bfile "$RAW_BFILE" --plink "$PLINK"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python "$CODE/scripts_stage2/05_materialize_temporal_component.py" \
    --root "$ROOT" --analysis-name "$AUX_ANALYSIS" --phenotype-cov "$COV" --candidate-file "$ALL_CAND" --plink "$PLINK" \
    --stage1-followup-min 0 --stage1-followup-max "${STAGE1_FOLLOWUP_MAX:-3}" --n-jobs "$N_JOBS"
  python - "$AUXDIR/single_iv_genetic_component_manifest.json" <<'PY'
import sys,json
m=json.load(open(sys.argv[1]))
assert m['n_component_errors']==0, f"Auxiliary IV predictions failed: {m['n_component_errors']}"
print('[PASS] All auxiliary Stage-I frozen GReX predictions materialized.')
PY
fi
if [[ $MODE == run || $MODE == all ]]; then
  [[ -s "$ALL_CAND" ]] || { echo "[ERROR] Run prepare first: $ALL_CAND" >&2; exit 1; }
  [[ -s "$AUXDIR/single_iv_genetic_component_manifest.json" ]] || { echo "[ERROR] Run prepare first: auxiliary GReX missing" >&2; exit 1; }
  # Respect the original AR computation and its singular_reduced_form failures.
  # Do not re-estimate AR with a different rank / model just to force a P value.
  if [[ ! -s "$BASE/publication_robustness/MRI_DBM_IV_reduced_form_AR_type_diagnostics.csv" ]]; then
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python "$CODE/scripts_stage2/15_mri_dbm_robustness_master.py" \
      --root "$ROOT" --mri-file "$MRI_FILE" --component "$COMPONENT" --clinical-master "$CLINICAL" \
      --output-dir "$BASE/publication_robustness"
  else
    echo '[REUSE] Original MRI AR-type diagnostics; no recalculation'
  fi
  # Original Stage-II LD audit; if LD data are unavailable, leave LD metrics NA.
  if ! python "$CODE/scripts_stage2/09_cross_gene_ld_audit.py" \
    --root "$ROOT" --analysis-name mri_dbm_lag1 \
    --candidate-file "$BASE/MRI_DBM_lag1_global_FDR05.csv" \
    --raw-bfile "$RAW_BFILE" --plink "$PLINK" --report-r2 0.10 --flag-r2 0.10 \
    --out-dir "$OUT/ld_audit"; then
    echo '[WARN] LD audit unavailable; other diagnostics continue, LD kept NA' >&2
    rm -f "$OUT/ld_audit/lead_iv_ld_pairs.csv"
  fi
  if [[ -s "$OUT/ld_audit/lead_iv_missing_from_raw_bim.csv" ]]; then
    if [[ $(wc -l < "$OUT/ld_audit/lead_iv_missing_from_raw_bim.csv") -gt 1 ]]; then
      echo '[WARN] LD SNPs missing in BIM; suppressing incomplete LD annotations' >&2
      rm -f "$OUT/ld_audit/lead_iv_ld_pairs.csv"
    fi
  fi
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python "$CODE/scripts_stage2/17_mri_evidence_master.py" \
    --root "$ROOT" --component "$COMPONENT" --mri-file "$MRI_FILE" --aux-analysis-name "$AUX_ANALYSIS" \
    --clinical-wide "$CLINICAL" --aux-candidates "$ALL_CAND" --out-dir "$OUT"
  # Do not change AR: only merge metrics actually computed with original tests.
  python "$CODE/scripts_stage2/18_mri_publish_evidence.py" --root "$ROOT" --component "$COMPONENT"
fi
if [[ $MODE == publish ]]; then
  # Safe for partially unavailable AR/multi-IV: missing metrics remain NA.
  python "$CODE/scripts_stage2/18_mri_publish_evidence.py" --root "$ROOT" --component "$COMPONENT"
fi
case "$MODE" in prepare|run|all|publish) ;; *) echo 'Usage: run_mri_dbm_evidence_full.sh [prepare|run|all|publish]' >&2; exit 2;; esac
