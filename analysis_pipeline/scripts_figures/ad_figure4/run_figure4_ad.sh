#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
OUT=${OUT:-$ROOT/manuscript_figures/tqtl_benchmark_v4/figure4_ad_mmse}
TABLE=${TABLE:-$ROOT/single_iv_causal/ad_mmse/standard_final/posthoc_evidence/Supplement_AD_MMSE_IV_gene_publication_POWER80.csv}
mkdir -p "$OUT" "$ROOT/logs"
CMD=${1:-plot};shift || true
case "$CMD" in
  plot) python "$HERE/01_make_ad_figure4.py" --root "$ROOT" --publication-table "$TABLE" --outdir "$OUT" "$@" ;;
  *) echo 'Usage: bash run_figure4_ad.sh plot [--top-n 30 --min-genotype-n 10 --max-years 10 --story-examples GENE1 GENE2]' >&2; exit 2;;
esac
