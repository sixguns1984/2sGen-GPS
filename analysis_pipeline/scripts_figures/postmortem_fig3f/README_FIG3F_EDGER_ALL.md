# 2sGenGPS manuscript Figure 3F: edgeR-QLF DEG-count heatmaps (v1.0)

**Scope:** plot only; **NO edgeR model fitting**, H5AD, VCF, DESeq2 or Dreamlet access.

## Exact source

```
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
FDR=$ROOT/single_iv_causal/postmortem_snp_scDE/carrier_vs_noncarrier_v6_vcf_verified/edger/FDR05_significant_edgeR_QLF_genes.csv
```

Input fields used: `snp`, `region`, `cell_type`, `stratum`, `method`, `gene_id`, `log2FC`, `padj`, `contrast`.

Filter `stratum == "all"` and `padj < 0.05` and `method == "edger_qlf"`. Up = `log2FC > 0` and down = `log2FC < 0` for the **VCF ALT-carrier vs noncarrier** contrast. Counts are **distinct gene IDs** within SNP × region × cell type. Values shown in cells are raw DEG counts; fills are log2(count+1).

Order matches the reference: regions `DMNX, GPI, PVC, PMC, PFC`; cells `Exc, Inh, Oli, Immune, OPC, Ast, Vast`. Important: `Immune` and `Vast` are kept **exactly as in the source annotation**; they are not uncritically relabeled `Mic`/`End`.

The two SNPs have separate rows in the combined Figure 3F, and separate per-SNP files are also exported. Up (yellow-red) and down (yellow-blue) have separate shared color scales, consistent across both SNPs.

## Critical coverage / zero-count rule

`FDR05_significant_edgeR_QLF_genes.csv` alone **cannot** tell zero-DEG fit from a skipped or missing fit. The plot automatically reads these files from the **same `edger/` folder**, if present:

- `fit_status.csv` (primary: `PASS` status for `stratum=all`)
- `edgeR_QLF_group_summary.csv` (fallback for completed groups)

If a successfully fitted comparison has no rows in the FDR CSV, label **0**. If the comparison is absent/skipped/unverified, label **-** on a gray cell. The latter must not be interpreted as a zero-DEG result. `Figure3F_edgeR_QLF_all_DEG_count_audit.csv` records the distinction per group. 

For testing with a partial copy of only the significant CSV, `--assume-complete` explicitly converts unverified cells to 0, **but this mode is not recommended for a final manuscript figure**, and its output is labeled `WARNING: unverified zeros assumed`.

## Installation (archive only adds these files)

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
ZIP=$ROOT/2sGenGPS_Figure3F_edgeR_QLF_all_heatmap_v1.zip
unzip -oq "$ZIP" -d "$CODE"
```

## Plot using existing finalized edgeR results

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
bash "$CODE/scripts_figures/postmortem_fig3f/run_figure3f_edger.sh"
```

If needed, override paths **without editing code**:

```bash
EDGER_FDR_CSV=/absolute/path/FDR05_significant_edgeR_QLF_genes.csv \
FIG3F_OUT=/absolute/output/dir \
bash "$CODE/scripts_figures/postmortem_fig3f/run_figure3f_edger.sh"
```

The launcher calls the existing Python/scvi interpreter with pandas/numpy/matplotlib.

## Outputs

Default location:

```
$ROOT/manuscript_figures/tqtl_benchmark_v4/figure3_postmortem_edger_qlf/
├── Figure3F_postmortem_edgeR_QLF_all.pdf / .svg / .png
├── Figure3F_rs2253588_edgeR_QLF_all.pdf / .svg / .png
├── Figure3F_rs35162690_edgeR_QLF_all.pdf / .svg / .png
├── Figure3F_edgeR_QLF_all_DEG_count_audit.csv
└── Figure3F_source_sha256.txt
```

PDF/SVG contain editable vector text. `matplotlib.use('cairo')` is included before importing pyplot; `savefig(backend='svg')` with `svg.fonttype='none'` preserves real SVG `<text>` objects. PDF is saved through Matplotlib's PDF writer with `pdf.fonttype=42` to preserve embedded editable text.

## Verified from provided CSV

`all`: rs2253588 = 97 up / 86 down (183); rs35162690 = 65 up / 431 down (496). Total = 679 DEG rows. No duplicate SNP/region/cell/gene IDs in the uploaded file. These are **not** counts of globally unique genes across all tissue contexts; DEG counts are within each comparison.

## Suggested figure legend (adapt to final manuscript)

**f, Regional and cell-type-specific transcriptomic associations with rs2253588 and rs35162690.** Heatmaps summarize the number of significantly upregulated (left) and downregulated (right) genes in VCF ALT-allele carriers relative to noncarriers across five brain regions and seven annotated cell classes. Gene-level differential expression was estimated using donor-level pseudobulk counts and edgeR quasi-likelihood models, with adjustment for age, sex and disease status in the combined case-and-control cohort. Significance was defined as BH-FDR < 0.05 within each SNP–region–cell-type comparison. Cell labels show integer gene counts; colors represent log2(count + 1). Gray cells indicate unverified or skipped comparisons, not zero significant genes.

## v1.1 Cairo/PNG compatibility fix (2026-11-04)

On some Matplotlib/Cairo versions `savefig(Path(...), format="png")`
raises `AttributeError: 'PosixPath' object has no attribute 'encode'` inside
`cairocffi.write_to_png`. The code now converts **all output filenames to
`str`** and uses `backend='agg'` explicitly **only for PNG**. It keeps
`matplotlib.use('cairo')` at import time, native `backend='pdf'` with
`pdf.fonttype=42` and native `backend='svg'` with `svg.fonttype='none'`
for Illustrator-editable vector text. No input/output filenames, statistical
filters, heatmap ordering or plot layout were changed.

Installation (backup and overwrite only the Fig3F plotting module):

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
ZIP=$ROOT/2sGenGPS_Figure3F_edgeR_QLF_all_heatmap_v1_1_Cairo_PNG_fix.zip
cp -a "$CODE/scripts_figures/postmortem_fig3f" \
  "$CODE/scripts_figures/postmortem_fig3f_backup_$(date +%Y%m%d_%H%M%S)"
unzip -oq "$ZIP" -d "$CODE"
python -m py_compile "$CODE/scripts_figures/postmortem_fig3f/01_plot_fig3f_edger_qlf.py"
bash "$CODE/scripts_figures/postmortem_fig3f/run_figure3f_edger.sh"
```

This is a plot-only fix, so do **not** re-run edgeR, Dreamlet or pseudobulk.
