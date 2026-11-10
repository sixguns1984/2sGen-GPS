# 2sGenGPS postmortem edgeR QLF add-on v1.0 (2026-11-03)

## Goal and actual inputs

This **adds** a true R/Bioconductor edgeR 4.x quasi-likelihood (QLF) analysis to the **already prepared** postmortem v6.2/v6.3 pseudobulk. It neither changes v6.3 `postmortem_scde.py` nor Dreamlet, nor recalculates VCF/H5AD/CTAR/Stage-II. Use the existing `2sgps_dreamlet_fix` environment (edgeR was loaded when Dreamlet was installed) and Python from `scvi`.

The module imports the existing `postmortem_scde.py` (v6.3+) for the **exact** `design_for()` model, `prepared_region_state()` checks, SNPs, regions, and cell types. Per region it reads:

- `OUT/pseudobulk/{region}/gene_mapping.csv` (gene ID/name map, v6.2 `gene_name`-collapse already done)
- `OUT/pseudobulk/{region}/{celltype}/counts.npz` (`donor x gene`, **raw integer counts**)
- `OUT/pseudobulk/{region}/{celltype}/donors.csv` (`donor,age,sex,case,n_cells`)
- `OUT/pseudobulk/{region}/.prepared_complete.json` (version and min cells)
- `OUT/vcf_genotypes.csv` (`snp,donor,carrier,alt_dose`)

The clinical metadata were originally loaded by v6.2 from `/public2/data/luojunfeng/2sTGCS_binary/scQTL/AMP_PD_Postmortem_Brain_2024_8_30_dwx.csv` (`IID`, `age_at_baseline`, `SEX`, `case_control_other_latest`). This add-on uses only the **prepared and validated metadata** instead of reading clinical files again. In the original VCF, ALT carrier is **rs2253588 C** and **rs35162690 T**. A noncarrier has ALT dose 0; a carrier has 1/2. This is a carrier association, not a claim that ALT is the original Stage-I effect allele.

## Statistics

For each SNP x brain region x cell type x stratum, samples are **independent donors**, never individual cells. All groups: `~ carrier + ageScaled + sex + status` (omit a constant sex/status column). Case-only: `~ carrier + ageScaled + sex` (omit constant sex). `ageScaled` uses the same group-specific centered/scaled age as v6.3, and gender/status reference levels are F/Control; carrier reference is 0. The tested coefficient is **`carrier1`**, positive log2FC = increased expression among carriers.

- Python applies the same v6.3 prefilter: sum of counts >=10, counts >=2 in >=3 donors, and >=5 donors in each SNP group.
- R edgeR applies `filterByExpr(DGEList, design=..., min.count=5, min.total.count=15, min.prop=0.7)` by default (independently and transparently; can be adjusted by flags).
- edgeR 4.x `calcNormFactors(method='TMM')`; `glmQLFit(design=..., robust=TRUE, legacy=FALSE)`; `glmQLFTest(coef='carrier1')`; within-comparison BH FDR.
- `stat` is the **nonnegative QL F statistic**: do not infer direction from `stat`. Direction comes from `log2FC`.
- No `n_cells` pseudo-replication, no donor random intercept on already aggregated one-row-per-donor matrix; edgeR does not implement Dreamlet's cell-count-specific precision weights.

Official edgeR workflows: https://bioconductor.org/packages/release/bioc/html/edgeR.html and https://bioconductor.org/packages/release/workflows/html/RnaSeqGeneEdgeRQL.html

## Installation (after uploading the ZIP to `$ROOT`)

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
ZIP=$ROOT/2sGenGPS_postmortem_edgeR_QLF_addon_v1.zip
unzip -oq "$ZIP" -d "$CODE"
RUN=$CODE/scripts_snp_scDE/run_edger.sh
OUT=$ROOT/single_iv_causal/postmortem_snp_scDE/carrier_vs_noncarrier_v6_vcf_verified
```

**No conda activation is necessary** if your paths match below: wrapper uses Python `$HOME/.conda/envs/scvi/bin/python` while Rscript defaults to `$HOME/.conda/envs/2sgps_dreamlet_fix/bin/Rscript`. To override, set `PYTHON_BIN=` and `EDGE_RSCRIPT=`. Alternatively pass `--rscript /absolute/path/to/Rscript` after the command.

```bash
# Version check (doesn't fit)
bash "$RUN" check --out "$OUT"

# Offline bridge test (does not require R to run an actual fit)
"$HOME/.conda/envs/scvi/bin/python" "$CODE/scripts_snp_scDE/test_edger_qlf_bridge.py"

# Dry run, no new R model or count export
bash "$RUN" fit --out "$OUT" --regions GPI --celltypes Oli \
  --snps rs2253588 --strata all --dry-run --max-jobs 1

# Real smoke test: just one combination
bash "$RUN" fit --out "$OUT" --regions GPI --celltypes Oli \
  --snps rs2253588 --strata all --max-jobs 1 --n-cpus 1 --fail-fast

# Full set, resuming only outputs matching current data and settings
bash "$RUN" fit --out "$OUT" --resume --n-cpus 2 --fail-fast
bash "$RUN" summarize --out "$OUT"
bash "$RUN" compare --out "$OUT"
```

`--n-cpus` limits numeric-library threads via environment variables; **it does not parallelize multiple edgeR fits**. Only one R process is launched at a time to avoid oversubscribing RAM. To refit, use `--overwrite` (then old per-group outputs are replaced upon success). For help: `bash "$RUN" fit --help`.

## Outputs (never overwrite prior DESeq2/HC3/Dreamlet)

```
OUT/edger/
  fit_status.csv
  de/{stratum}/{snp}/{region}/{celltype}/
      edger_qlf.csv                 # all genes retained by edgeR filterByExpr
      edger_qlf.manifest.json       # data/schema/code fingerprint for resume
      edger_qlf.log                 # R stdout/stderr for debugging
      edgeR_TMM_norm_factors.csv    # donor library sizes & normalization factors
  ALL_genes_edgeR_QLF_results.csv
  FDR05_significant_edgeR_QLF_genes.csv
  Top30_edgeR_QLF_genes_by_group.csv
  edgeR_QLF_group_summary.csv
  edgeR_QLF_vs_other_methods.csv
```

Cross-method comparison matches `gene_id` *within the same SNP/region/cell type/stratum*, reports log2FC Spearman / sign agreement / FDR overlap on shared tested genes. The tests do not falsely share or recompute FDR across distinct methods or different gene universes. A different number of DE genes is **not** by itself evidence of better statistical accuracy.

## Caveat and testing

The Python bridge/metadata/MatrixMarket order, dry run, simulated R invocation, resume, aggregation and comparison have passed local tests. **Actual R edgeR fitting could not be executed in the build environment** because it does not have R. Run the one-comparison smoke test on your existing server R environment first; see the per-comparison `edger_qlf.log` if an actual R interface issue appears. The patch does not modify any previously generated raw/pseudobulk/DE files.
