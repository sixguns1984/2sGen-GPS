#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT="${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}"
export EDGE_RSCRIPT="${EDGE_RSCRIPT:-$HOME/.conda/envs/2sgps_dreamlet_fix/bin/Rscript}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.conda/envs/scvi/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then PYTHON_BIN="$(command -v python3 || true)"; fi
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python interpreter not found. Set PYTHON_BIN=/path/to/python" >&2
    exit 2
fi
if [[ $# -lt 1 ]]; then
    echo 'Usage: run_edger.sh check|fit|summarize|compare [options]' >&2
    exit 2
fi
exec "$PYTHON_BIN" "$SCRIPT_DIR/edger_postmortem.py" "$@"
