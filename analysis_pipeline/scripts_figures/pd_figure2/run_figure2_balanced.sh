#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-/public2/data/luojunfeng/2sGenGPS_revised_chatgpt}
OUT=${OUT:-$ROOT/manuscript_figures/tqtl_benchmark_v4/figure2_pd_power80}
TABLE=${TABLE:-$ROOT/single_iv_causal/cross_phenotype_summary_lag1/Supplement_PD_IV_gene_publication_3phenotypes_POWER80.csv}
ARTIFACT=${ARTIFACT:-$ROOT/single_iv_causal/cross_phenotype_summary_lag1/PD_validated_IV_gene_stage1_candidates.csv}
FREQ=${FREQ:-$OUT/PPMI_plink19.frq}
PLINK=${PLINK:-/usr/bin/plink1.9}
GENO=${GENO:-$ROOT/data/genotype/PP/amp_pd_PP_qc}
mkdir -p "$OUT"
CMD=${1:-help}; shift || true
case "$CMD" in
  freq)
    [[ -f "${GENO}.bed" && -f "${GENO}.bim" && -f "${GENO}.fam" ]] || { echo "[ERROR] Missing genotype prefix $GENO" >&2; exit 2; }
    [[ -x "$PLINK" ]] || { echo "[ERROR] PLINK 1.9 not executable: $PLINK" >&2; exit 2; }
    "$PLINK" --bfile "$GENO" --freq --out "${FREQ%.frq}"
    ;;
  plot)
    python "$HERE/01_make_pd_figure2.py" \
      --root "$ROOT" \
      --publication-table "$TABLE" \
      --artifact-table "$ARTIFACT" \
      --outdir "$OUT" \
      --freq-file "$FREQ" \
      --example-cohort "${EXAMPLE_COHORT:-PP}" \
      ${STORY_EXAMPLES:+--story-examples $STORY_EXAMPLES} "$@"
    ;;
  panel-a)
    python "$HERE/01_make_pd_figure2.py" \
      --root "$ROOT" \
      --publication-table "$TABLE" \
      --artifact-table "$ARTIFACT" \
      --outdir "$OUT" \
      --panel-a-only "$@"
    ;;
  *) echo 'Usage: bash run_figure2_balanced.sh {freq|plot|panel-a}' >&2; exit 2 ;;
esac
