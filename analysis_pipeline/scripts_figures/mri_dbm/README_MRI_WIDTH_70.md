# MRI DBM example plot: narrower width (v1)

## Scope

This patch updates only the display stage of the original project script
`scripts_figures/mri_dbm/01_plot_mri_dbm_examples.py` recovered from the supplied
`scripts_figures.zip`. The original plotting entry point remains
`scripts_figures/mri_dbm/run_mri_dbm_examples.sh`.

- MRI Figure_MRI_DBM_FLOT1_C7orf25: **7.40 in to 5.18 in (-30%)**, height stays **9.40 in**.
- Individual FLOT1/C7orf25 output figures: **7.40 in to 5.18 in**, height stays **5.60 in**.
- Choose a different shrinkage with `--width-scale 0.75` (-25%) or
  `--width-scale 0.67` (-33%). Default is `0.70` (-30%).
- Long titles, MAF annotation, and explanatory footer are minimally formatted
  to fit the narrower canvas; expression/association analysis, genotype groups,
  MRI time points, statistics, colors and plot-axes layout are unchanged.
- Cairo is set before pyplot. Vector PDF and SVG are saved via their dedicated
  backends so SVG text remains editable. PNG uses Agg to avoid Cairo's Path issue.

## Install

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
ZIP=$ROOT/2sGenGPS_MRI_DBM_FLOT1_C7orf25_narrow30_v1.zip
cp -a "$CODE/scripts_figures/mri_dbm" \
  "$CODE/scripts_figures/mri_dbm_backup_$(date +%Y%m%d_%H%M%S)"
unzip -oq "$ZIP" -d "$CODE"
python -m py_compile "$CODE/scripts_figures/mri_dbm/01_plot_mri_dbm_examples.py"
```

## Redraw (existing numerical results; do not rerun MRI causal analysis)

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
bash "$CODE/scripts_figures/mri_dbm/run_mri_dbm_examples.sh" --width-scale 0.70
```

To make it exactly one-third narrower:

```bash
bash "$CODE/scripts_figures/mri_dbm/run_mri_dbm_examples.sh" --width-scale 0.67
```

Outputs (same as the original script):

```text
$ROOT/manuscript_figures/tqtl_benchmark_v4/mri_dbm_examples/
  Figure_MRI_DBM_FLOT1_C7orf25.pdf
  Figure_MRI_DBM_FLOT1_C7orf25.svg
  Figure_MRI_DBM_FLOT1_C7orf25.png
  FLOT1_rs2253588_RightCaudatenucleus7.pc1.pdf/.svg/.png
  C7orf25_rs35162690_LeftSubstantianigra.parsreticulata3.pc1.pdf/.svg/.png
  MRI_DBM_example_genotype_and_model_audit.csv
  MRI_example_figure_manifest.json
```

The script loads completed global-FDR MRI hits and frozen GReX predictions,
plus the original MRI outcome table for plotting means. It does **not** rerun
Stage I, Stage II, bootstrap, or clinical QTL analysis. Later-than-supported
cached GReX displayed in the original chart is still annotated as extrapolation;
this patch does not change its interpretation.

## Verification status

Mock-data layout and rendering tests passed for width scales 0.70 and 0.67:
PNG/PDF/SVG exported, PDF page widths verified, no PDF text lies outside page,
SVG retains actual `<text>` elements; single-panel layout also passed. The
remote server's full real MRI inputs were not available locally, so replotting
real MRI curves remains a server-side action.
