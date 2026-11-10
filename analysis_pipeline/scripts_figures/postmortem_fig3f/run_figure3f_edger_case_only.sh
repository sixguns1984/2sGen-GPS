#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}"
CODE="${CODE:-$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized}"
PYTHON_BIN="${PYTHON_BIN:-${HOME}/.conda/envs/scvi/bin/python}"
SCRIPT="$CODE/scripts_figures/postmortem_fig3f/01_plot_fig3f_edger_qlf.py"
INPUT="${EDGER_FDR_CSV:-$ROOT/single_iv_causal/postmortem_snp_scDE/carrier_vs_noncarrier_v6_vcf_verified/edger/FDR05_significant_edgeR_QLF_genes.csv}"
OUT="${FIG3F_CASE_OUT:-$ROOT/manuscript_figures/tqtl_benchmark_v4/figure3_postmortem_edger_qlf_case_only}"
if [[ ! -x "$PYTHON_BIN" ]]; then PYTHON_BIN="$(command -v python)"; fi
[[ -s "$INPUT" ]] || { echo "Missing edgeR significant table: $INPUT" >&2; exit 1; }
[[ -f "$SCRIPT" ]] || { echo "Missing Figure 3F script: $SCRIPT" >&2; exit 1; }
exec "$PYTHON_BIN" "$SCRIPT" --root "$ROOT" --input "$INPUT" --outdir "$OUT" --stratum case_only "$@"
