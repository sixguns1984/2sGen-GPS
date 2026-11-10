#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
OUT=${OUT:-$ROOT/manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_N80_final}
DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# Invalid placeholders may linger in a user's exported environment from old README.
# They must NEVER reach argparse (and must NEVER be silently used as a numerical cutoff).
validate_optional_p () {
    local name="$1"
    local val="${!name:-}"
    [[ -z "$val" ]] && return 0
    if ! python -c 'import math,sys;v=float(sys.argv[1]);sys.exit(0 if math.isfinite(v) and 0<v<=.05 else 1)' "$val" 2>/dev/null; then
        echo "[WARN] Ignoring invalid environment variable ${name}=${val}; recovering the true CTAR scan threshold instead." >&2
        unset "$name"
    fi
}
validate_optional_p BH_RAW_P
validate_optional_p ALPHA_INTER

ARGS=(--root "$ROOT" --output "$OUT" --n-pairs 1500 --min-pairs 1000 --bootstraps 200)
if [[ -n "${BH_RAW_P:-}" ]]; then ARGS+=(--alpha-disc "$BH_RAW_P"); fi
if [[ -n "${ALPHA_INTER:-}" ]]; then ARGS+=(--alpha-inter "$ALPHA_INTER"); fi
echo "[INFO] CTAR SFig6 directory: $DIR"
echo "[INFO] Real data root: $ROOT"
python "$DIR/ctar_sfig6_n80.py" "${ARGS[@]}" "$@"
