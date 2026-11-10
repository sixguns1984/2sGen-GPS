#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
CODE=${CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
TMPDIR=${TMPDIR:-$ROOT/tmp}
LOG=${LOG:-$ROOT/logs/stage1_ctar_mptga_genomewide.log}

mkdir -p "$TMPDIR" "$ROOT/logs"
export TMPDIR TMP="$TMPDIR" TEMP="$TMPDIR"
cd "$CODE"

exec python scripts_amp_pd/03g_run_revised_followup_discovery.py \
  --root "$ROOT" \
  --methods submitted_followup_ctar_mptga \
  --gpus 0,1,2 \
  --cpu-workers 1 \
  --genes-per-chunk 100 \
  --batch-size 2048 \
  --gpu-batch-map 0:4096,1:2048,2:2048 \
  --min-gpu-batch 256 \
  --rho-points 100 \
  --ctar-reference-interval 1.0 \
  --ctar-rho-max 0.99 \
  --legacy-genotype-mode binary \
  --fdr-strategy stagewise_hierarchical \
  --fdr-alpha 0.05 \
  --dtype float64 \
  --tmp-dir "$TMPDIR"
