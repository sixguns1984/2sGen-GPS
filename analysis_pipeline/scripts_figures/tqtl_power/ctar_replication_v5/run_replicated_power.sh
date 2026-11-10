#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
CODE=${CODE:-$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CORE=$CODE/scripts_figures/tqtl_power/2sGenGPS_CTAR_SFig6_refit_v3
MODE=${POWER_MODE:-representative}
if [[ "$MODE" != representative && "$MODE" != phenotype-final ]]; then echo "Invalid POWER_MODE=$MODE" >&2;exit 2;fi
if [[ "$MODE" == representative ]]; then
  OUT=${POWER_OUT:-$ROOT/manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_REPLICATED_v5}
else
  OUT=${POWER_OUT:-$ROOT/manuscript_figures/tqtl_benchmark_v4/clinical_IV_ctar_power_coefsum_v6}
fi
export OUT ROOT CODE
cmd=${1:-help};shift || true
case "$cmd" in
 prepare)
  python "$HERE/00_prepare_replicated_loci.py" --root "$ROOT" --mode "$MODE" --outdir "$OUT" --n-pairs "${N_PAIRS:-5000}" --seed "${SEED:-20261030}"
  ;;
 cache-expression|audit-inputs|preflight|full|full-serial|merge)
  bash "$CORE/run_sfigure6.sh" "$cmd" "$@"
  ;;
 plot)
  python "$HERE/03_plot_replication_conditioned.py" --root "$ROOT" --outdir "$OUT" "$@"
  ;;
 *) echo "Usage: POWER_MODE=representative|phenotype-final bash $(basename "$0") prepare|cache-expression|audit-inputs|preflight|full|plot" >&2;exit 2;;
esac
