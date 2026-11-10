#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
CODE=${CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
TMPDIR=${TMPDIR:-$ROOT/tmp}
GPUS=${GPUS:-0,1,2}
GENES_PER_CHUNK=${GENES_PER_CHUNK:-100}
BATCH_SIZE=${BATCH_SIZE:-2048}
GPU_BATCH_MAP=${GPU_BATCH_MAP:-0:4096,1:2048,2:2048}
MIN_GPU_BATCH=${MIN_GPU_BATCH:-256}

mkdir -p "$TMPDIR" "$ROOT/logs"
export TMPDIR TMP="$TMPDIR" TEMP="$TMPDIR"
cd "$CODE"

exec python scripts_amp_pd/04g_run_revised_followup_replication.py \
  --root "$ROOT" \
  --methods submitted_followup_ctar_mptga \
  --gpus "$GPUS" \
  --cpu-workers 1 \
  --genes-per-chunk "$GENES_PER_CHUNK" \
  --batch-size "$BATCH_SIZE" \
  --gpu-batch-map "$GPU_BATCH_MAP" \
  --min-gpu-batch "$MIN_GPU_BATCH" \
  --rho-points 100 \
  --ctar-reference-interval 1.0 \
  --ctar-rho-max 0.99 \
  --legacy-genotype-mode binary \
  --fdr-strategy stagewise_hierarchical \
  --fdr-alpha 0.05 \
  --dtype float64 \
  --tmp-dir "$TMPDIR"
