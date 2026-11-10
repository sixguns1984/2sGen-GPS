#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
CODE=${CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
OUT=${OUT:-$ROOT/simulation_qtl_benchmark_rho_covariate}
SIM_DEVICE=${SIM_DEVICE:-gpu}   # gpu | cpu
GPUS=${GPUS:-0,1,2}
CPU_PARALLEL=${CPU_PARALLEL:-4}
DTYPE=${DTYPE:-float64}
BATCH_SIZE=${BATCH_SIZE:-2048}
N_PARTICIPANTS=${N_PARTICIPANTS:-300}
N_PAIRS=${N_PAIRS:-1000}
N_REPLICATES=${N_REPLICATES:-50}
RHO_POINTS=${RHO_POINTS:-20}
RHO_VALUES=${RHO_VALUES:-0,0.3,0.6,0.9}
SCHEDULED_TIMES=${SCHEDULED_TIMES:-0,1,2,3,4}
CONFOUNDING_STRENGTH=${CONFOUNDING_STRENGTH:-0.90}
ALT_FRACTION=${ALT_FRACTION:-0.50}
DYNAMIC_FRACTION=${DYNAMIC_FRACTION:-0.50}
STATIC_EFFECT=${STATIC_EFFECT:-0.28}
DYNAMIC_LINEAR=${DYNAMIC_LINEAR:-0.34}
DYNAMIC_QUADRATIC=${DYNAMIC_QUADRATIC:-0.10}
COVARIANCE_MODE=${COVARIANCE_MODE:-discrete_visit}
PLOT_RESULTS=${PLOT_RESULTS:-1}
METHODS=${METHODS:-submitted_followup_mptga,submitted_followup_ctar_mptga,lin2018_mptga,lin2018_regression,lin2018_regression_cov,lin2018_ar,union,fisher,manova}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}

if [[ "$SIM_DEVICE" == "cpu" ]]; then
  GPUS=""
elif [[ "$SIM_DEVICE" != "gpu" ]]; then
  echo "SIM_DEVICE must be cpu or gpu" >&2
  exit 2
fi

cd "$CODE"

python scripts_benchmark/benchmark_simulation.py \
  --out-dir "$OUT" \
  --methods "$METHODS" \
  --gpus "$GPUS" \
  --cpu-parallel "$CPU_PARALLEL" \
  --n-participants "$N_PARTICIPANTS" \
  --n-pairs "$N_PAIRS" \
  --n-replicates "$N_REPLICATES" \
  --batch-size "$BATCH_SIZE" \
  --dtype "$DTYPE" \
  --rho-points "$RHO_POINTS" \
  --rho-values "$RHO_VALUES" \
  --scheduled-times "$SCHEDULED_TIMES" \
  --covariance-mode "$COVARIANCE_MODE" \
  --confounding-strength "$CONFOUNDING_STRENGTH" \
  --alt-fraction "$ALT_FRACTION" \
  --dynamic-fraction "$DYNAMIC_FRACTION" \
  --static-effect "$STATIC_EFFECT" \
  --dynamic-linear "$DYNAMIC_LINEAR" \
  --dynamic-quadratic "$DYNAMIC_QUADRATIC"

if [[ "$PLOT_RESULTS" == "1" ]]; then
  python scripts_benchmark/plot_simulation.py \
    --input "$OUT/simulation_summary_by_replicate.csv" \
    --out-dir "$OUT/figures"
fi
