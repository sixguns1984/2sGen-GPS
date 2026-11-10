# Three-phenotype PD temporal association integration

This replaces the four-phenotype integration for `cross_phenotype_summary_lag1`.
Only `updrs1`, `updrs2`, and `updrs3` are read. MoCA is excluded from inputs,
outputs, counts, and downstream candidate construction.

## Run

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
OUT=$ROOT/single_iv_causal/cross_phenotype_summary_lag1

python "$CODE/scripts_amp_pd/summarize_pd_three_phenotypes.py" \
  --root "$ROOT" \
  --out-dir "$OUT"
```

If keeping the script in the existing summary directory instead:

```bash
python "$OUT/summarize_pd_three_phenotypes.py" --root "$ROOT" --out-dir "$OUT"
```

## Main outputs

- `Supplement_PD_IV_gene_ALL_statistics_3phenotypes.csv`: all source statistics, unchanged.
- `Supplement_PD_IV_gene_publication_3phenotypes.csv`: long publication table.
- `PD_three_phenotypes_publication_wide.csv`: wide pair-level table.
- `PD_three_phenotypes_IV_gene_pairs.csv`: exact union of gene + lead_snp pairs that are
  primary downstream candidates in at least one UPDRS phenotype.
- `PD_three_phenotypes_gene_summary.csv` and `PD_three_phenotypes_gene_list.txt`.
- `PD_three_phenotypes_summary.json`: counts, input paths, and provenance.

The IV-pair list is the recommended input for downstream MRI DBM and CSF Olink analyses
when those workflows accept gene + lead_snp candidates. Use the gene list only if a
downstream workflow explicitly operates at gene-only level.

## Important
The script reads the three existing `standard_final/*_FINAL_causal_IV_gene_master_FULL.csv`
files (falling back to the concise master only if FULL is absent). It does not rerun
Stage-II UPDRS analyses and does not change any phenotype-specific candidate-selection
rules. It only integrates the existing final outputs.

## 2026-10-30 v1.1 hotfix

- Supports final masters already containing `phenotype` and `replication_design`.
- Rejects unexpected pre-existing phenotype/replication metadata instead of silently changing source evidence.
- Fixes the gene-level `n_supported_phenotypes` when distinct SNP-IVs for the same gene support distinct UPDRS traits.
- Run `python scripts_amp_pd/test_three_phenotypes.py` to verify both master schemas.

This hotfix does not rerun or alter individual Stage-II models. Existing four-phenotype
outputs remain present until explicitly archived; downstream MRI/CSF commands must use
`PD_three_phenotypes_IV_gene_pairs.csv`, not an old four-phenotype CSV.
