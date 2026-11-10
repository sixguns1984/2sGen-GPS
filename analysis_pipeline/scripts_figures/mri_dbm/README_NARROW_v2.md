# MRI DBM Figure FLOT1/C7orf25 — extra narrow layout v2

This overlays the previously installed `scripts_figures/mri_dbm` plotter. The only changes are visual layout, no Stage-I/Stage-II/MRI models or cached predictions are recomputed.

## Fixes

* `--width-scale` now accepts values **0.30 through 1.00** instead of 0.60–1.00.
* Default 0.45 (7.4 * 0.45 = **3.33 inches**); height remains 9.4 inches for the combined figure and 5.6 inches for a standalone example.
* At widths below 0.60, title and MAF/PLINK annotations wrap over two lines (or three title lines below 0.40), legends use two/one columns, and margins are widened in absolute inches so y labels stay visible.
* The lower example's tGReX panel moves slightly up so explanatory footer no longer overlaps with `Years since baseline`.
* Uses Cairo as the requested interactive renderer and `backend="svg"`, `backend="pdf"` for editable vector text; `backend="agg"` for PNG compatibility.

## Install

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
ZIP=$ROOT/2sGenGPS_MRI_DBM_extra_narrow_v2.zip
cp -a "$CODE/scripts_figures/mri_dbm" "$CODE/scripts_figures/mri_dbm_backup_$(date +%Y%m%d_%H%M%S)"
unzip -oq "$ZIP" -d "$CODE"
python -m py_compile "$CODE/scripts_figures/mri_dbm/01_plot_mri_dbm_examples.py"
```

## Draw only — no new MRI or tQTL inference

Suggested new width (60% narrower than original):

```bash
bash "$CODE/scripts_figures/mri_dbm/run_mri_dbm_examples.sh" --width-scale 0.40
```

Even narrower (70% narrower than original):

```bash
bash "$CODE/scripts_figures/mri_dbm/run_mri_dbm_examples.sh" --width-scale 0.30
```

Other supported choices include 0.35, 0.45, and 0.50. Very small scales can make the labels harder to read in print.

Outputs are overwritten only in the existing figure directory:

`$ROOT/manuscript_figures/tqtl_benchmark_v4/mri_dbm_examples/`

* `Figure_MRI_DBM_FLOT1_C7orf25.pdf/.svg/.png`
* separate FLOT1 and C7orf25 example plot files
* the existing genotype/model audit and plot manifest (based on the same source cache)

The `figure_width_scale` and `figure_width_inches` fields of `MRI_example_figure_manifest.json` record the actual width chosen.

## Offline verification

Synthetic plotting tests passed at width-scale 0.50, 0.45, 0.40, 0.35 and 0.30, including single-example plot; verified PDF page sizes, that no text extends past the page edge, and SVG has editable `<text>` elements. These are plotting tests, not a rerun of actual patient or MRI results.
