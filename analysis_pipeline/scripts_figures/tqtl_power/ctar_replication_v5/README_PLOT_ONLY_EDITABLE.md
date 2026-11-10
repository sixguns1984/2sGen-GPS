# CTAR REPLICATED v5 — redraw-only and editable vector text

This is for the **completed** `sfigure6_ctar_REPLICATED_v5` results, **not** `power_analysis_CTAR_v5` or `power_analysis_CTAR_v6`. No tQTL preparation, power analysis, bootstrap, or sample-size simulation is performed.

## Required existing files (all three, checked before plotting)

- `locus_bootstrap_results.csv`: valid completed PP/PD per-SNP-gene power and model LRT fields; used to create panels a/b/c and paired cohort data.
- `matched_pair_power.csv`: existing PP/PD/joint per-pair powers; read without re-computation.
- `power_summary.json`: existing fractions and CI; read without overwriting.

## Optional existing files (for complete original panels)

- `donor_sample_size_sensitivity.csv`: used for sensitivity panels d/e. If missing, panel d displays a missing-substudy note and panel e only selection-context annotation.
- `bootstrap_LRT_examples/*_PP.csv`: used for panel a LRT distribution curves; if missing, the reference cutoff still appears.

Not read for plotting: `selected_loci.csv`, `expression_selected.parquet`, `locus_failures.csv`, `stage1_input_alignment_audit*`, `_preflight/`, `_shards/`. These belong to preparation/audit/calculation steps, not this redraw.

## Install (server)

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
ZIP=$ROOT/2sGenGPS_SFigure6_CTAR_REPLICATED_v5_plot_only_editable.zip
cp -a "$CODE/scripts_figures/tqtl_power/ctar_replication_v5" "$CODE/scripts_figures/tqtl_power/ctar_replication_v5_backup_$(date +%Y%m%d_%H%M%S)"
unzip -oq "$ZIP" -d "$CODE"
```

## Correct redraw command

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
POWER_OUT=$ROOT/manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_REPLICATED_v5 \
  bash "$CODE/scripts_figures/tqtl_power/ctar_replication_v5/run_replicated_v5_plot_only_editable.sh"
```

Output in same v5 directory:
`Supplementary_Figure_6_CTAR_REPLICATED_v5.pdf`, `.svg`, `.png`.

The plotting script keeps `matplotlib.use('cairo')` before `pyplot`. To preserve editable typography, `svg.fonttype='none'`, `pdf.fonttype=42`, and explicit `backend='svg'/'pdf'` are used for vector saves. SVG should include actual `<text>` elements; PDF uses embedded TrueType text when supported. Check the resulting file in Illustrator, which can vary by font availability.
