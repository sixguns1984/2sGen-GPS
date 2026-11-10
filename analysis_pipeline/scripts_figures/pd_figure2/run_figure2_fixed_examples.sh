#!/usr/bin/env bash
# Figure 2 plot-only, fixed b=COQ8A/rs370085216, c=ARL17B/rs11654422.
# Uses existing Stage-I BIM and frozen Stage-II component; does NOT run QTL/ARX.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
OUT=${OUT:-$ROOT/manuscript_figures/tqtl_benchmark_v4/figure2_pd_power80}
BIM=${BIM:-$ROOT/data/genotype/PP/amp_pd_PP_qc.bim}
mkdir -p "$OUT"
STORY_CSV="$OUT/Figure2_fixed_examples.csv"
cat > "$STORY_CSV" <<'CSV'
gene,lead_snp,phenotype
COQ8A,rs370085216,updrs3
ARL17B,rs11654422,updrs3
CSV
[[ -s "$BIM" ]] || { echo "[ERROR] Stage-I BIM missing: $BIM" >&2; exit 2; }
export ROOT OUT
bash "$HERE/run_figure2_balanced.sh" plot \
  --story-csv "$STORY_CSV" \
  --example-cohort PP \
  --bim "$BIM" \
  --top-labels-per-phenotype "${TOP_LABELS:-8}" \
  "$@"
echo "[DONE] $OUT/Figure2_PD_temporal_association_main_power80.{pdf,svg,png}"
echo "[AUDIT] $OUT/Figure2_examples_allele_MAF_audit.csv"
