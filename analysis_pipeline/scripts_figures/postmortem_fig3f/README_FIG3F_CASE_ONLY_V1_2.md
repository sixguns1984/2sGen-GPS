# 2sGenGPS Fig 3F: edgeR QLF PD case-only heatmap (v1.2)

This is a **plot-only** update to the tested v1.1 Cairo/PNG-compatible Fig 3F script.
It adds the `--stratum case_only` flag and a dedicated one-command runner.
The default `--stratum all` behavior and the pre-existing all-donor runner are retained.

## Data selection

- Source: `FDR05_significant_edgeR_QLF_genes.csv`, the existing edgeR-QLF BH-FDR significant table.
- `stratum == "case_only"`, `padj < 0.05`.
- ALT carrier vs noncarrier (`log2FC > 0` = Up, `< 0` = Down).
- Two SNPs: rs2253588 and rs35162690.
- Regions: DMNX, GPI, PVC, PMC, PFC.
- Cells: Exc, Inh, Oli, Immune, OPC, Ast, Vast.
- Heatmap color: log2(DEG count + 1); counts printed as integers.
- Tested zero is distinguished from unverified or skipped cells using `edger/fit_status.csv` and `edgeR_QLF_group_summary.csv`. Never use `--assume-complete` for publication without independent verification.
- No re-fitting, VCF scanning, or H5AD read.

## Install

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
ZIP=$ROOT/2sGenGPS_Figure3F_edgeR_QLF_case_only_v1_2.zip
cp -a "$CODE/scripts_figures/postmortem_fig3f" "$CODE/scripts_figures/postmortem_fig3f_backup_$(date +%Y%m%d_%H%M%S)"
unzip -oq "$ZIP" -d "$CODE"
python -m py_compile "$CODE/scripts_figures/postmortem_fig3f/01_plot_fig3f_edger_qlf.py"
```

## Plot only (PD cases only)

```bash
bash "$CODE/scripts_figures/postmortem_fig3f/run_figure3f_edger_case_only.sh"
```

Optionally override output location with `FIG3F_CASE_OUT`, source CSV with `EDGER_FDR_CSV` or pass `--dpi 400`.

Default output directory:

```
$ROOT/manuscript_figures/tqtl_benchmark_v4/figure3_postmortem_edger_qlf_case_only/
```

Files created:

- `Figure3F_postmortem_edgeR_QLF_case_only.pdf/.svg/.png`
- `Figure3F_rs2253588_edgeR_QLF_case_only.pdf/.svg/.png`
- `Figure3F_rs35162690_edgeR_QLF_case_only.pdf/.svg/.png`
- `Figure3F_edgeR_QLF_case_only_DEG_count_audit.csv`
- `Figure3F_case_only_source_sha256.txt`

PDF and SVG have editable text; PNG uses Agg for compatibility with older Cairo.
Python source includes `import matplotlib; matplotlib.use('cairo')` before pyplot.

## Original all-donor Figure 3F remains unchanged

```bash
bash "$CODE/scripts_figures/postmortem_fig3f/run_figure3f_edger.sh"
```

## Local verification

```bash
python "$CODE/scripts_figures/postmortem_fig3f/test_fig3f_case_only.py"
python "$CODE/scripts_figures/postmortem_fig3f/test_fig3f_edger.py"
```

Validation on the uploaded significant CSV (without server fit_status):
- `case_only` 589 significant entries, rs2253588 120 Up + 76 Down; rs35162690 24 Up + 369 Down.
- Some absent comparison cells are **unverified** when only the significant CSV is provided; the plot displays `-` for these, not 0. On the server, use the accompanying edgeR fit_status/group summary to resolve these cells.

## Why distinct output directories matter

The `case_only` figure is a PD subgroup comparison of SNP carriers and noncarriers. It is **not** a PD versus control differential-expression analysis. It should not overwrite the all-donor main comparison.
