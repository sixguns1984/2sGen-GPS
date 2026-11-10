#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# The previous erroneous placeholder variables must never be used.
unset BH_RAW_P ALPHA_INTER
python "$HERE/ctar_practical_sample_size.py" --root "$ROOT" "$@"
