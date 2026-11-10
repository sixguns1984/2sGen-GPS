#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
CODE=${CODE:-$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized}
MRI_FILE=${MRI_FILE:-/public2/data/luojunfeng/2sTGCS_binary/outcome/DBM/confound_pass_snp/MRI_DBM.csv}
CAND=${CAND:-$ROOT/single_iv_causal/cross_phenotype_summary_lag1/PD_validated_IV_gene_stage1_candidates.csv}
ANALYSIS_NAME=mri_dbm_lag1
RAW_BFILE=${RAW_BFILE:-/public/labdata/public_data/AMP_PD_WGS/plink/all_chrs_merged}
PLINK=${PLINK:-/usr/bin/plink1.9}
N_JOBS=${N_JOBS:-8}
STAGE1_FOLLOWUP_MAX=${STAGE1_FOLLOWUP_MAX:-3}
DIR=$ROOT/single_iv_causal/$ANALYSIS_NAME
COV=$DIR/mri_visit_covariates.csv
mkdir -p "$DIR" "$ROOT/logs"
# Strictly use full frozen Stage1 model parameters for each of the 255 pairs.
python - "$CAND" <<'PY_CHECK'
import sys,pandas as pd
x=pd.read_csv(sys.argv[1],nrows=5)
needed={'gene','lead_snp','artifact_id','basis_json'}
if not needed.issubset(x.columns) or not any(c.startswith('beta_g_') for c in x.columns):
 raise SystemExit('Candidate file lacks frozen Stage1 basis_json/beta_g_*. Use Stage1-enriched 255-pair manifest; do NOT use UPDRS GReX cache')
PY_CHECK
python "$CODE/scripts_stage2/14a_prepare_mri_prediction_cov.py" --mri-file "$MRI_FILE" --out "$COV"
python "$CODE/scripts_stage2/03_prepare_phenotype_genotype.py" --root "$ROOT" --analysis-name "$ANALYSIS_NAME" --phenotype-cov "$COV" --candidate-file "$CAND" --raw-bfile "$RAW_BFILE" --plink "$PLINK"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python "$CODE/scripts_stage2/05_materialize_temporal_component.py" --root "$ROOT" --analysis-name "$ANALYSIS_NAME" --phenotype-cov "$COV" --candidate-file "$CAND" --plink "$PLINK" --stage1-followup-min 0 --stage1-followup-max "$STAGE1_FOLLOWUP_MAX" --n-jobs "$N_JOBS"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python "$CODE/scripts_stage2/14_mri_dbm_lag1.py" --root "$ROOT" --analysis-name "$ANALYSIS_NAME" --mri-file "$MRI_FILE" --candidates "$CAND" --component "${COMPONENT:-1}" --n-jobs "$N_JOBS" --out-dir "$DIR/pc${COMPONENT:-1}"
