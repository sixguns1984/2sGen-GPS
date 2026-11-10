#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}"
export ROOT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
if [[ $# -eq 0 ]]; then echo "Usage: bash $0 audit|prepare|fit|cell-lmm|summarize|all [opts]"; exit 2; fi
MODE="$1"; shift
# Compatible with prior legacy 'snp-audit' and 'lmm-summary' entry names.
if [[ "$MODE" == snp-audit ]]; then MODE=audit; fi
if [[ "$MODE" == lmm-summary ]]; then MODE=summarize; fi
exec python "$SCRIPT_DIR/postmortem_scde.py" "$MODE" "$@"
