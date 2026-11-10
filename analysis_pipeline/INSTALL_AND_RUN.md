# 2sGenGPS 2026-10-30 hotfix — PLINK 1.9 + Figure 2 shell launcher

This patch **does not change Stage-I or Stage-II analyses**. It edits only
`01_prepare_real_inputs.py`, `run_sfigure6.sh`, `02_figure2_balanced_wrapper.py`,
`03_rank_high_maf_examples.py` (same code + PLINK1.9 compatible `.frq` input), and
adds `run_figure2_balanced.sh` and a standalone mock smoke test.

The ZIP is rooted at `scripts_figures/`, so unpack with **`-d "$CODE"`**, not `-d "$FIG"`.

## Install

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
ZIP=$ROOT/2sGenGPS_20261030_plink19_figure2_hotfix.zip

# Check upload before changing source
ls -lh "$ZIP" /usr/bin/plink1.9

# Backup the small files that this patch overwrites
BACKUP=$ROOT/backup_hotfix_20261030_$(date +%Y%m%d_%H%M%S)
mkdir -p "$BACKUP"
for REL in \
 scripts_figures/tqtl_power/2sGenGPS_CTAR_SFig6_refit_v3/01_prepare_real_inputs.py \
 scripts_figures/tqtl_power/2sGenGPS_CTAR_SFig6_refit_v3/run_sfigure6.sh \
 scripts_figures/pd_figure2/02_figure2_balanced_wrapper.py \
 scripts_figures/pd_figure2/03_rank_high_maf_examples.py; do
  if [ -f "$CODE/$REL" ]; then
    mkdir -p "$BACKUP/$(dirname "$REL")"
    cp -p "$CODE/$REL" "$BACKUP/$REL"
  fi
done

unzip -oq "$ZIP" -d "$CODE"

# Mock PLINK1.9 + mock Figure 2 full CLI and data-interface test (no real data modified)
python "$CODE/scripts_figures/test_20261030_hotfix.py"
```

## A. Power: same working directory, now PLINK1.9

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
POWER=$CODE/scripts_figures/tqtl_power/2sGenGPS_CTAR_SFig6_refit_v3

# The original failed BEFORE genotype export; re-run prepare safely.
# No need to rerun Stage-I.
bash "$POWER/run_sfigure6.sh" prepare
bash "$POWER/run_sfigure6.sh" cache-expression
bash "$POWER/run_sfigure6.sh" preflight

# ONLY if preflight fit audit is valid:
# nohup bash "$POWER/run_sfigure6.sh" full > "$ROOT/logs/sfig6_CTAR_refit_v3.log" 2>&1 &
# bash "$POWER/run_sfigure6.sh" plot
```

PLINK1.9 `.raw` files have genotype header `Variant_ID_A1` and are read by
existing `real_data.py`. This patch uses `--recode A`, NOT PLINK2 `--export A`.
It scans each cohort `.bim` to audit absent variants before extraction and
checks previous `.raw` headers before reuse. Missing cohort variants are
reported in `genotypes_PP_unmatched_variants.txt` or `genotypes_PD_unmatched_variants.txt`.

## B. Figure 2: no `$FIG` environment variable is required

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
F2=$CODE/scripts_figures/pd_figure2/run_figure2_balanced.sh

# PLINK 1.9 --freq -> .frq; build a ranked table with high MAF.
bash "$F2" prepare

# Run installed original plotter through the balanced layout wrapper.
nohup bash "$F2" plot > "$ROOT/logs/figure2_balanced_v1_4.log" 2>&1 &
tail -f "$ROOT/logs/figure2_balanced_v1_4.log"
```

`prepare` finds an existing `Supplement_PD_IV_gene_publication*.csv` in the project.
If your final publication table has another name/location, set **TABLE** in the same command:

```bash
TABLE=/absolute/path/to/Supplement_PD_IV_gene_publication.csv bash "$F2" prepare
```

The wrapper expects your already-installed `01_make_pd_figure2.py`. It first
checks `scripts_figures/pd_figure2/01_make_pd_figure2.py` and then searches
`$CODE/scripts_figures` for that exact filename. If the source has been renamed,
set `ORIGINAL=/absolute/path/to/original_plot_script.py` for `plot`.

**Important:** The old plotter may choose a SNP by gene/phenotype. The new
wrapper verifies the actual plotted `rsID` matches the selected high-MAF
candidate `lead_snp`; if it differs, it stops rather than silently labelling a
wrong variant. To genuinely change exact SNP selection in that case, edit the
original plotter's lookup logic; the source is not bundled in this patch.

The displayed `A1/A2` are PLINK BIM allele labels. They are **not** asserted
to be the dosage-counted effect allele without checking actual original
plotter genotype harmonization. MAF comes from actual PPMI PLINK1.9 `.frq`.

GReX x-range is clipped to the last observed phenotype time in b/c/d.
The 2x2 layout avoids the over-tall panel a. The original output remains;
additional `_balanced` files are produced.

## Validation / limits

- Passed mock PLINK v1.90b6.26 genotype export/reuse/absent-SNP audit and
  mock Figure2 2x2 save using ranked SNP + actual `.frq` and `.bim`.
- 9 existing upstream unit tests passed.
- The actual server source of `01_make_pd_figure2.py` is not in this ZIP;
  project-backed older ZIP could not be materialized in this environment.
- Real PPMI/PDBP genotype, expression, covariate and Stage-I refit have NOT
  been validated on server. **The preflight LRT audit remains mandatory**.
- All power computations concern overall temporal tQTL only, not dynamic tQTL.
