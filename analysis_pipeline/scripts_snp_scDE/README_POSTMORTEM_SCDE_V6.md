# 2sGenGPS postmortem scDE — v6 VCF-verified ALT carrier analysis

This is a **formal, self-contained downstream analysis module** in `scripts_snp_scDE/`; not a diagnostic-only patch. It preserves Stage-I CTAR-MPTGA, Stage-II UPDRS I/II/III ARX, the project root and all existing downstream result directories. No dynamic QTL or MoCA analysis is added.

## 1. Verified data sources

- Original postmortem genotype: `/public/labdata/public_data/AMP_PD_Postmortem_Brain/vcf/merged.100s.vcf`, 100 WGS samples. Use **GT** as the authoritative genotype; do not infer carrier from PLINK A1 or A2.
- PLINK cross-check: `/public/labdata/public_data/AMP_PD_Postmortem_Brain/vcf/merged.100s.new.fixed.{bed,bim,fam}`. FAM cross-check is informative if IDs match; if no FAM match can be made, leave the BED comparison explicitly unverified.
- Stage-I reference BIM: `$ROOT/data/genotype/PP/amp_pd_PP_qc.bim`; replication BIM: `$ROOT/data/genotype/REP/amp_pd_REP_qc.bim`.
- Stage-I significant CTAR: `$ROOT/qtl/PP/submitted_followup_ctar_mptga/final/significant_tqtl.parquet`, replication `$ROOT/qtl/PD/submitted_followup_ctar_mptga/final/all_results.parquet`. Existing IV candidate table `$ROOT/single_iv_causal/iv_validation/single_iv_primary_eligible.csv` is read only for associating the target SNPs with their eligible IV-gene pairs.
- Five cell count H5AD files under `/public2/data/duwenxuan/AMP_PD_anno/try/`: `GPI_subtype_count_try.h5ad`, `DMNX_subtype_count_try.h5ad`, `PFC_subtype_count_try.h5ad`, `PMC_subtype_count_try.h5ad`, `PVC_subtype_count_try.h5ad`.
- Known obs keys: `IID`, `cell_type`, `subtype`. Age/sex/diagnosis are autodetected among common obs aliases and **must be present** in obs or supplied using an external donor-level CSV.
- Required cell types: `Oli`, `Immune`, `Inh`, `OPC`, `Ast`, `Vast`, `Exc`. `Other`/unlisted cells excluded.
- H5AD var index: Ensembl feature ID. Gene symbol is retrieved from `gene_name`, `gene_symbols`, `gene_symbol`, `feature_name`, `genename`, or another explicit symbol column. No internet mappings and no implicit gene-name-to-ENSG coercion. Outputs preserve both `gene_id` and `gene_symbol`. Version suffixes on Ensembl IDs are stripped; duplicate Ensembl IDs stop preparation instead of allowing ambiguous results.

## 2. Corrected VCF grouping from actual raw GT

| rsID | position | REF | ALT | GT 0/0 | GT 0/1 | GT 1/1 | missing | ALT carrier | ALT noncarrier |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| rs2253588 | chr6:30931600 | G | C | 56 | 40 | 0 | 4 | 40 | 56 |
| rs35162690 | chr7:43178314 | TC | T | 20 | 47 | 0 | 33 | 47 | 20 |

GT=0/0 -> carrier 0; GT=0/1 or 1/1 -> carrier 1; missing GT excluded from corresponding SNP. No genotype imputation. The above numbers are **full-VCF totals**, not the final donor count within each region/cell type. The script checks these reference carrier totals and stops if a different genotype file is accidentally supplied.

**Crucial**: VCF ALT, PLINK A1 and Stage-I effect allele are not interchangeable. Stage-I PPMI BIM has rs2253588 A1=C/A2=G and rs35162690 A1=TC/A2=T; postmortem PLINK A1/A2 can be reversed. The reported contrast here is always **ALT carrier versus ALT noncarrier**. Do not interpret the sign as the Stage-I tQTL A1 effect without allele harmonization (especially for carrier versus allele-dose contrasts).

VCF sample `0_PM-HA_S00447-BLM0-PVC-DWGS` maps to H5AD `obs['IID']='PM-HA_S00447-BLM0'` after **exactly** removing the `0_` prefix and `-(GPI|DMNX|PFC|PMC|PVC)-DWGS` suffix. Never rely on row order, substring / partial matches, or PV[C] region string to identify actual brain region of a donor; the region comes from the H5AD filename.

## 3. Methods

**Pseudobulk preparation**: backed H5AD by 3,000-cell chunks, verify raw nonnegative integer counts from `layers['counts']` (if present), else `X`; sum counts by donor × brain region × major `cell_type`. Donors must have >=30 cells in a specific region/cell type. No normalized expression matrices are pseudobulked. Counts, donor covariates, feature mappings and unmatched donors saved. Genotype matching occurs by donor identifier; absent/missing genotypes are not controls.

**Primary donor-level analysis**:

- PyDESeq2 **0.4.12**, on raw donor pseudobulk counts: `~ Carrier + age + sex + Case/Control` (`case-only`: `~ Carrier + age + sex`). Uses explicit contrast `['carrier', '1', '0']`, nonzero-gene filtering, BH FDR within each SNP × region × cell type × stratum. User's existing environment must have pydeseq2==0.4.12.
- OLS HC3 sensitivity analysis, response gene log2-CPM with library-size pseudo-count: identical covariates, HC3 robust standard errors and BH FDR. The HC3 beta is a log2-CPM association coefficient, NOT the DESeq2 count-GLM log2FC estimate.
- Cell-level donor random-intercept MixedLM sensitivity, for the top 30 pseudobulk genes per unit: sample ≤30 cells per donor with deterministic RNG; use log1p(count/total-count × 1e4). The cell-LMM p-values are sensitivity only, not independent donor replication.

Full (all cases+controls) and PD-only results are separate; for full, donor case/control must be known. Age and sex must be valid. Rank-deficient designs and donor groups with fewer than 5 carriers or 5 noncarriers are **skipped**, not silently fitted with fewer covariates. Case-only drops a constant case covariate. A disease label such as AD or another diagnosis is **not** silently treated as PD.

**Caution**: rs35162690 has 33/100 missing calls and no observed ALT homozygotes. Check missingness by diagnosis/age and variant-level QC before biological interpretation. Hardy–Weinberg deviations require scrutiny but do not by themselves prove a genotyping problem.

## 4. Installation (non-destructive)

Recommended overlay onto the **current** `$CODE` rather than replacing all Stage-I/II files (the full source snapshot included with this release is based on the accessible 2026-10-26 code ZIP; later remote server changes may exist):

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
MODULE_ZIP=$ROOT/2sGenGPS_postmortem_scDE_v6_integrated_module.zip

# Back up only the old postmortem module (if present)
if [ -d "$CODE/scripts_snp_scDE" ]; then
  cp -a "$CODE/scripts_snp_scDE" "$CODE/scripts_snp_scDE_before_v6_$(date +%Y%m%d_%H%M%S)"
fi
unzip -oq "$MODULE_ZIP" -d "$CODE"
```

Install runtime dependencies in the appropriate existing Python env (do NOT overwrite Stage-I environment if it already works):

```bash
python -m pip install 'pydeseq2==0.4.12' anndata scipy statsmodels pandas numpy
```

## 5. Run

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
RUN=$CODE/scripts_snp_scDE/run_snp_scDE.sh

# No-data offline tests
python "$CODE/scripts_snp_scDE/test_postmortem_scde_v6.py"
python "$CODE/scripts_snp_scDE/test_mock_h5ad_v6.py"

# Raw VCF GT -> audit of BIM/PLINK and Stage-I IV-gene pair mapping
bash "$RUN" audit

# Five H5AD, donor-level count pseudobulk
bash "$RUN" prepare --chunk-size 3000 --min-cells 30

# Inspect true region/cell-type/snp donor counts before fitting:
head -n 25 "$ROOT/single_iv_causal/postmortem_snp_scDE/carrier_vs_noncarrier_v6_vcf_verified/matched_group_counts.csv"

# Both donor-level methods (all + case-only); case-only does not need separate flag
bash "$RUN" fit --methods both --n-cpus 4 --min-group 5

# Selected-gene cell-LMM sensitivity
bash "$RUN" cell-lmm --top-n 30 --max-cells-per-donor 30
bash "$RUN" summarize
```

If diagnosis/age/sex are not available in H5AD.obs (check `pseudobulk/<region>/input_columns.json`), provide donor covariates with `IID,age,sex,case`:

```bash
bash "$RUN" prepare --donor-metadata /absolute/path/to/postmortem_donor_metadata.csv
bash "$RUN" fit --methods both --n-cpus 4
bash "$RUN" cell-lmm --donor-metadata /absolute/path/to/postmortem_donor_metadata.csv
bash "$RUN" summarize
```

**Note**: do not substitute a guessed metadata path. If no diagnosis metadata is available, the code explicitly fails instead of giving an unadjusted result falsely labeled as covariate-adjusted.

## 6. Outputs

All saved under `$ROOT/single_iv_causal/postmortem_snp_scDE/carrier_vs_noncarrier_v6_vcf_verified/`, separate from prior `carrier_vs_noncarrier_v5` results.

- `vcf_genotypes.csv` (VCF sample, normalized donor, GT and ALT dosage/carrier)
- `vcf_gt_summary.csv` (carrier/noncarrier/missing counts)
- `vcf_plink_concordance.csv` (per-donor VCF vs PLINK genotype agreement when FAM mapped)
- `stage1_postmortem_BIM_audit.csv` and `stage1_stage2_target_snp_IV_genes.csv`
- `pseudobulk/<region>/gene_mapping.csv`, `input_columns.json`, `donor_covariates.csv`, `unmatched_sc_donors.csv`
- `pseudobulk/<region>/<cell_type>/counts.npz`, `donors.csv`
- `matched_group_counts.csv`: n_carrier/n_noncarrier/n_missing per SNP × region × cell_type × all/PD-case-only
- `de/<all|case_only>/<snp>/<region>/<cell_type>/<hc3|deseq2>.csv` (full gene-level fit output)
- `fit_status.csv` (all attempted and skipped jobs, reasons, actual n)
- `cell_LMM_top_genes.csv` (donor random-intercept sensitivity)
- `ALL_genes_DE_results.csv`, `FDR05_significant_DE_genes.csv`, `Top30_DE_genes_by_group.csv`, `DE_group_summary.csv`
- `stage1_target_gene_presence_in_sc.csv` (Stage-I/II eligible SNP-gene pairs matched to each H5AD's exact Ensembl feature IDs)

**No Stage-I or Stage-II results are recomputed, and no prior analysis outputs are overwritten.**

## 7. Scope of validation

Validated with a synthetic 100-sample VCF reproducing the two genotype tables and a synthetic H5AD-like input that exercises donor grouping, pseudobulk aggregation, HC3 regression, case-only/full fits and summaries. **Not** validated against actual remote 10.75-GB VCF or H5AD, and PyDESeq2 0.4.12 / MixedLM against real donor data must be checked on the server. The actual H5AD metadata column names for age, sex, and diagnosis were not in the accessible source material and are not assumed to be confirmed.
