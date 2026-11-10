#!/usr/bin/env bash
set -euo pipefail

CODE=${CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
: "${ROOT:?Set ROOT}"
: "${ANALYSIS_NAME:?Set ANALYSIS_NAME}"
: "${PHENOTYPE_COV:?Set PHENOTYPE_COV}"
: "${OUTCOME_COL:?Set OUTCOME_COL}"

QTL_LABEL=${QTL_LABEL:-submitted_followup_ctar_mptga}
RAW_BFILE=${RAW_BFILE:-/public/labdata/public_data/AMP_PD_WGS/plink/all_chrs_merged}
PPMI_BFILE=${PPMI_BFILE:-$ROOT/data/genotype/PP/amp_pd_PP_qc}
PLINK=${PLINK:-/usr/bin/plink1.9}
TWSTATS=${TWSTATS:-/usr/bin/twstats}
TWTABLE=${TWTABLE:-twtable}
N_JOBS=${N_JOBS:-16}
STRENGTH_JOBS=${STRENGTH_JOBS:-4}
STRENGTH_BATCH_GENES=${STRENGTH_BATCH_GENES:-8}
TASK_CHUNKSIZE=${TASK_CHUNKSIZE:-8}
MAX_LAG=${MAX_LAG:-4}
MIN_PARTICIPANTS=${MIN_PARTICIPANTS:-50}
LD_R2=${LD_R2:-0.1}
STAGE1_FOLLOWUP_MIN=${STAGE1_FOLLOWUP_MIN:-0}
STAGE1_FOLLOWUP_MAX=${STAGE1_FOLLOWUP_MAX:-3}
TIME_VARYING_COVARIATES=${TIME_VARYING_COVARIATES:-upd23a_medication_for_pd}
FORCE_SHARED_PREP=${FORCE_SHARED_PREP:-0}
REPLICATION_COHORTS=${REPLICATION_COHORTS-PD,SU}
REPLICATION_POOLED_GROUPS=${REPLICATION_POOLED_GROUPS-PD+SU}
EXTRA_COVARIATES=${EXTRA_COVARIATES:-}
CANDIDATE_FILE=${CANDIDATE_FILE:-$ROOT/single_iv_causal/iv_validation/single_iv_primary_eligible.csv}
PDBP_QTL_TABLE=${PDBP_QTL_TABLE:-}

STAGE13_DIR=${STAGE13_DIR:-$ROOT/single_iv_causal/$ANALYSIS_NAME/temporal_causal_agecubic_full_followup_fast}
STANDARD_DIR=${STANDARD_DIR:-$ROOT/single_iv_causal/$ANALYSIS_NAME/standard_final}
mkdir -p "$ROOT/logs" "$STANDARD_DIR"
cd "$CODE"

SHARED_IV="$ROOT/single_iv_causal/iv_validation/single_iv_primary_eligible.csv"
SHARED_STRENGTH="$ROOT/single_iv_causal/iv_strength_all/all_IV_first_stage_strength.csv"
if [[ "$FORCE_SHARED_PREP" == "1" || ! -s "$SHARED_IV" || ! -s "$SHARED_STRENGTH" ]]; then
  ROOT="$ROOT" CODE="$CODE" QTL_LABEL="$QTL_LABEL" PPMI_BFILE="$PPMI_BFILE" PLINK="$PLINK" LD_R2="$LD_R2" \
  STRENGTH_JOBS="$STRENGTH_JOBS" STRENGTH_BATCH_GENES="$STRENGTH_BATCH_GENES" PDBP_QTL_TABLE="$PDBP_QTL_TABLE" \
  bash "$CODE/run_stage2_prepare_temporal_ivs.sh"
else
  echo "[REUSE] Shared Stage-II IV eligibility and strength files already exist."
  echo "        Set FORCE_SHARED_PREP=1 to rebuild them."
fi

python scripts_stage2/03_prepare_phenotype_genotype.py \
  --root "$ROOT" \
  --analysis-name "$ANALYSIS_NAME" \
  --phenotype-cov "$PHENOTYPE_COV" \
  --candidate-file "$CANDIDATE_FILE" \
  --raw-bfile "$RAW_BFILE" \
  --plink "$PLINK"

python scripts_stage2/04_prepare_phenotype_pcs.py \
  --root "$ROOT" \
  --analysis-name "$ANALYSIS_NAME" \
  --raw-bfile "$RAW_BFILE" \
  --plink "$PLINK" \
  --twstats "$TWSTATS" \
  --twtable "$TWTABLE" \
  --cohorts "${DISCOVERY_COHORT:-PP}${REPLICATION_COHORTS:+,$REPLICATION_COHORTS}"

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python scripts_stage2/05_materialize_temporal_component.py \
  --root "$ROOT" \
  --analysis-name "$ANALYSIS_NAME" \
  --phenotype-cov "$PHENOTYPE_COV" \
  --qtl-label "$QTL_LABEL" \
  --candidate-file "$CANDIDATE_FILE" \
  --plink "$PLINK" \
  --stage1-followup-min "$STAGE1_FOLLOWUP_MIN" \
  --stage1-followup-max "$STAGE1_FOLLOWUP_MAX" \
  --n-jobs "$N_JOBS"

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python scripts_stage2/06_run_temporal_causal_fast.py \
  --root "$ROOT" \
  --analysis-name "$ANALYSIS_NAME" \
  --phenotype-cov "$PHENOTYPE_COV" \
  --outcome-col "$OUTCOME_COL" \
  --candidate-file "$CANDIDATE_FILE" \
  --replication-cohorts "$REPLICATION_COHORTS" \
  --replication-pooled-groups "$REPLICATION_POOLED_GROUPS" \
  --extra-covariates "$EXTRA_COVARIATES" \
  --max-lag "$MAX_LAG" \
  --min-participants "$MIN_PARTICIPANTS" \
  --time-varying-covariates "$TIME_VARYING_COVARIATES" \
  --use-genotype-pcs \
  --adjust-current-age-cubic \
  --no-require-stage1-followup-support \
  --ar-mode off \
  --detail-mode minimal \
  --task-chunksize "$TASK_CHUNKSIZE" \
  --out-dir "$STAGE13_DIR" \
  --n-jobs "$N_JOBS"

export CODE ROOT ANALYSIS_NAME PHENOTYPE_COV OUTCOME_COL STAGE13_DIR STANDARD_DIR PPMI_BFILE PLINK N_JOBS TASK_CHUNKSIZE MIN_PARTICIPANTS TIME_VARYING_COVARIATES REPLICATION_COHORTS REPLICATION_POOLED_GROUPS EXTRA_COVARIATES CANDIDATE_FILE
bash "$CODE/run_stage2_postprocess_existing_fast.sh"
