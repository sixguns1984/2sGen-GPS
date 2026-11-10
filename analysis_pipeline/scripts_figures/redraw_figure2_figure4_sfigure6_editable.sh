#!/usr/bin/env bash
# Plot-only re-render from the previously computed publication/power inputs.
# Never launches Stage-I/Stage-II, CTAR or power-calibration calculations.
set -euo pipefail
ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
CODE=${CODE:-$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized}
FIGROOT=${FIGROOT:-$ROOT/manuscript_figures/tqtl_benchmark_v4}
WHICH=${1:-all}; if [[ $# -gt 0 ]]; then shift; fi
case "$WHICH" in
  pd)
    bash "$CODE/scripts_figures/pd_figure2/run_figure2_balanced.sh" plot --top-labels-per-phenotype 8 "$@"
    ;;
  ad)
    bash "$CODE/scripts_figures/ad_figure4/run_figure4_ad.sh" plot --top-n 30 --top-labels-per-phenotype 8 --example-cohort MAP "$@"
    ;;
  power|power-v6)
    FIGROOT="$FIGROOT" bash "$CODE/scripts_figures/tqtl_power/run_sfigure6_power_v6.sh" plot "$@"
    ;;
  power-v5)
    FIGROOT="$FIGROOT" bash "$CODE/scripts_figures/tqtl_power/run_sfigure6_CTAR_v5.sh" plot "$@"
    ;;
  all)
    [[ $# == 0 ]] || { echo '[ERROR] For separate extra figure options, use pd/ad/power directly' >&2; exit 2; }
    bash "$0" pd
    bash "$0" ad
    bash "$0" power-v6
    ;;
  *)
    echo 'Usage: bash redraw_figure2_figure4_sfigure6_editable.sh {pd|ad|power|power-v5|all} [figure-specific arguments]' >&2
    exit 2;;
esac
