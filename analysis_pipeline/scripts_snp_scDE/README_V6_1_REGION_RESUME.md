# Postmortem scDE v6.1 — interrupted pseudobulk preparation recovery

This corrects `RuntimeError: Missing prepared region DMNX` without modifying the frozen Stage-I/Stage-II workflows or prior result files. The error occurs when `fit` requests GPI, DMNX, PFC, PMC and PVC but `prepare` completed only a subset (or used a different `--out`). A directory alone is not sufficient to indicate complete preparation; all seven cell-type pseudobulk artifacts are now required.

## Command sequence — full five-region output

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
RUN=$CODE/scripts_snp_scDE/run_snp_scDE.sh
OUT=$ROOT/single_iv_causal/postmortem_snp_scDE/carrier_vs_noncarrier_v6_vcf_verified

# Fast no-data regression tests:
python "$CODE/scripts_snp_scDE/test_postmortem_scde_v6.py"
python "$CODE/scripts_snp_scDE/test_mock_h5ad_v6.py"
python "$CODE/scripts_snp_scDE/test_prepare_resume_v6_1.py"

# Do not re-run 10.75 GB VCF audit when OUT/vcf_genotypes.csv already exists.
# Fill missing or incomplete regions; leave complete GPI/PFC/PMC/PVC untouched.
bash "$RUN" prepare --out "$OUT" --skip-prepared --chunk-size 3000 --min-cells 30

# Full genome-wide pseudobulk DE: all expressed genes tested for both SNPs.
bash "$RUN" fit --out "$OUT" --methods both --n-cpus 4 --min-group 5
bash "$RUN" summarize --out "$OUT"

# Optional cell-level sensitivity only for TOP 30 genes per group:
bash "$RUN" cell-lmm --out "$OUT" --top-n 30 --max-cells-per-donor 30
```

Alternatively **opt in** to repair missing/incomplete regions directly during fit:

```bash
bash "$RUN" fit --out "$OUT" --methods both --n-cpus 4 --auto-prepare-missing
```

By default, fit now checks **all requested regions before fitting any**, and fails with a specific `prepare --regions REGION` suggestion if a region is missing or incomplete. With `--auto-prepare-missing`, the missing region is read from the original count H5AD, aggregated, and fitted. To prepare only DMNX with the original v6 or v6.1 code:

```bash
bash "$RUN" prepare --out "$OUT" --regions DMNX --min-cells 30
bash "$RUN" fit --out "$OUT" --methods both --n-cpus 4
```

The full transcriptome is assessed using pseudobulk PyDESeq2 and OLS-HC3 after gene expression filtering: total count >=10; >=3 donors each with >=2 counts; >=30 cells per donor/cell type. Cell-LMM is deliberately limited to top 30 pseudobulk hits, for donor-random-intercept sensitivity, not genome-wide testing. `ALL_genes_DE_results.csv`, `fit_status.csv` and `DE_group_summary.csv` show actual tested genes and groups.

NOTE: any previously fitted region may be recalculated when fitting again; its results are replaced in the SAME v6 output folder. The release does not overwrite Stage-I/Stage-II or any v5 postmortem output. Do not combine results prepared with different `--min-cells` thresholds.
