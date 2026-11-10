#!/usr/bin/env bash
set -euo pipefail
# V8.9 revised policy: Stage-I QTL/benchmark uses the manuscript-exact QC'ed
# PPMI/PDBP matrices. Current Stage-II uses the separate current-V8.9 resources.
exec bash "$(dirname "$0")/run_stage1_manuscript_qtl.sh" "$@"
