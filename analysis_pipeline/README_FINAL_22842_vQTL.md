# Current final 22,842 temporal cis-eQTL → residual-vQTL sensitivity (PPMI)

This patch adds a **new** sensitivity script, without replacing the old historical dynamic-vQTL analysis and **without altering Stage-I/Stage-II IV eligibility or causal association results**.

## Exact final input paths (confirmed from the code bundle)

```text
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
ROOT/single_iv_causal/iv_validation/single_iv_primary_eligible.csv
  gene, lead_snp (22,842 distinct eligible SNP–gene pairs expected)
ROOT/qtl/PP/submitted_followup_ctar_mptga/final/significant_tqtl.parquet
  Ensembl_ID, Variant_ID, rho[, rho_reference_interval]
```

The patch maps `(gene, lead_snp)` to `(Ensembl_ID, Variant_ID)` by **exact pair keys**, retrieves the Stage-I CTAR residual-correlation parameter `rho`, checks that every one of the 22,842 final eligible pairs matched, and refuses to continue if there is a missing pair or duplicate key. This is the *post-replication, trajectory-concordant, LD-pruned* final panel, **not** `significant_dynamic.parquet` or the unfiltered large significant-tQTL discovery pool.

## Statistics

The analysis reuses the original Stage-I donor-cross-fitted residual fluctuation implementation:

1. For each SNP–gene pair, fit the **full** cubic CTAR mean (common cubic time, eight genotype-specific cubic coefficients, fitted nuisance covariates) on 4/5 of PPMI participants at a time; predict the held-out participants, generating out-of-fold raw mean residuals. **The original Stage-I full-model rho is held fixed across folds.**
2. Compute per-participant residual sample variance from at least 3 longitudinal visits; test log(variance + numerical offset) against 0/1/2 genotype dosage with a participant-level HC3-robust Wald test adjusted for participant-mean Stage-I nuisance covariates and visit count. This is the **primary 1-df additive residual-vQTL test**.
3. Additional sensitivities: 2-df genotypic group differences (requires at least 10 participants in each of 0/1/2 dosage groups) and an additive log-MAD variability test.
4. Apply BH correction **across the entire fixed eligible set of 22,842 SNP–gene pairs** for each test separately, with untestable pairs assigned P=1 conservatively. Report both tested and untestable denominators. A targeted smoke subset has only **diagnostic**, not manuscript-level, FDR.

A positive vQTL test is an **additional genotype-associated residual-variability signal**; it does not invalidate the corresponding mean-trajectory tQTL. Conversely, a nonsignificant vQTL test does not prove absence of variance effects. This is a selected, discovery-cohort sensitivity analysis, not an independent PDBP vQTL replication. It does **not** support treating 22,842 tests as a genome-wide unselected family.

## Installation

Copy the three files to the existing code folder, keeping directory structure. **Do not unpack the original 3.7-MB project ZIP again**; no existing files need overwriting.

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE="$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized"
PATCH=/path/to/2sGenGPS_FINAL_22842_vQTL_patch.zip
unzip -o "$PATCH" -d "$CODE"
chmod +x "$CODE/run_stage1_final_22842_vqtl.sh"
```

## Verify inputs only (no fitting)

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE="$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized"
ROOT="$ROOT" bash "$CODE/run_stage1_final_22842_vqtl.sh" --preflight-only
```

Expected: `[PREFLIGHT PASS] matched_pairs=22,842, unique_genes=7,570` (if the server's final files match the revised manuscript).

## Small smoke (never use its FDR in a paper)

```bash
ROOT="$ROOT" N_JOBS=2 GENES_PER_TASK=2 bash "$CODE/run_stage1_final_22842_vqtl.sh" \
  --max-genes 3 \
  --out-dir "$ROOT/qtl/PP/submitted_followup_ctar_mptga/final/vqtl_final_22842_SMOKE"
```

## Complete analysis (resumable; 8 workers)

```bash
mkdir -p "$ROOT/logs"
nohup env ROOT="$ROOT" N_JOBS=8 GENES_PER_TASK=10 \
  bash "$CODE/run_stage1_final_22842_vqtl.sh" \
  > "$ROOT/logs/stage1_final_22842_vqtl.log" 2>&1 &
# Progress:
tail -f "$ROOT/logs/stage1_final_22842_vqtl.log"
```

Restarting the **same command** resumes finished gene-batch checkpoints; `--restart` intentionally clears old batches. A run fingerprint prevents accidentally resuming against different 22,842 inputs, different stage-I CTAR files, or different cross-fitting settings. Choose `N_JOBS` to suit available RAM; each worker loads expression/genotype metadata.

Optional explicit source paths:

```bash
bash "$CODE/run_stage1_final_22842_vqtl.sh" \
  --eligible-csv "$ROOT/single_iv_causal/iv_validation/single_iv_primary_eligible.csv" \
  --ctar-parquet "$ROOT/qtl/PP/submitted_followup_ctar_mptga/final/significant_tqtl.parquet"
```

## Outputs

```text
ROOT/qtl/PP/submitted_followup_ctar_mptga/final/vqtl_final_22842/
├── final_22842_tqtl_vqtl_results.parquet      # all pairs, including untestable
├── final_22842_tqtl_vqtl_results.csv.gz
├── vqtl_overlap_summary.csv                   # n eligible, tested, FDR hits, fractions
├── untestable_primary_pairs.csv
├── errors.csv
├── vqtl_sensitivity_metadata.json
├── run_fingerprint.json
└── batches/batch_*.json                       # durable resumable checkpoints
```

The **primary** manuscript-ready values are in the row `crossfit_log_variance_additive_PRIMARY` of `vqtl_overlap_summary.csv`. Do not reuse the historical 42,455/1,430 counts for this updated 22,842-pair panel.

## Dependencies

Use the existing project's environment, including `numpy`, `pandas`, `scipy`, `pyarrow`, `bed-reader` or the configured PLINK reader, and other dependencies already needed by `gengps.amp_pd.real_qtl`. This patch adds **no new third-party dependencies**.
