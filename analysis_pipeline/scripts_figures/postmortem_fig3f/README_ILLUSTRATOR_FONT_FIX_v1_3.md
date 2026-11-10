# Fig.3F Adobe Illustrator-friendly typography v1.3

Fixes pink Adobe Illustrator substitute-font highlights caused by DejaVu Sans embedded font subsets when PDFs are opened in Illustrator.

## What changes
- PDF: base-14 Helvetica + Helvetica-Bold (native editable PDF text). No DejaVu Sans dependency.
- SVG: native text with font-family `Arial` (editable text; preferred for Illustrator on Mac).
- Colorbar labels: visual subscript 2 rendered as a separate editable text object; avoids DejaVu mathtext.
- PNG: Agg backend for Cairo-PNG compatibility.
- Data, DEG counts, color scales, input files, stratification and default output paths unchanged.

## Install
```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE="$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized"
ZIP="$ROOT/2sGenGPS_Figure3F_Illustrator_fonts_v1_3.zip"
cp -a "$CODE/scripts_figures/postmortem_fig3f" "${CODE}/scripts_figures/postmortem_fig3f_backup_$(date +%Y%m%d_%H%M%S)"
unzip -oq "$ZIP" -d "$CODE"
python -m py_compile "$CODE/scripts_figures/postmortem_fig3f/01_plot_fig3f_edger_qlf.py"
```

## Run
```bash
bash "$CODE/scripts_figures/postmortem_fig3f/run_figure3f_edger_case_only.sh"
# Optional, original all donors:
bash "$CODE/scripts_figures/postmortem_fig3f/run_figure3f_edger.sh"
```

## Open in Illustrator
Prefer `Figure3F_postmortem_edgeR_QLF_case_only.svg` for native, editable Arial text. The PDF is also editable with standard Helvetica. If preexisting pink highlight persists, close/reopen the regenerated file; do not reuse an old Illustrator document's already-substituted font objects. Illustrator Type > Find/Replace Font reveals any remaining missing fonts.

## Validation
Verified with user-provided FDR05_significant_edgeR_QLF_genes.csv. Case-only totals rs2253588 up 120/down 76; rs35162690 up 24/down 369. PDF fonts verified as Helvetica and Helvetica-Bold only; SVG font-family Arial on all text nodes. No Illustrator/macOS UI test was possible here.
