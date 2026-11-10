# 2sGenGPS postmortem scDE v6.2 — confirmed 2026-11-02 donor clinical source

## Source provenance (not inferred)

The donor metadata file and its exact clinical columns were provided and checked in the 2026-11-02 project chat. The 2026-11-03 v6/v6.1 code erroneously searched for clinical columns inside H5AD.obs. This is fixed here by setting the actual clinical table as the default.

```
/public2/data/luojunfeng/2sTGCS_binary/scQTL/AMP_PD_Postmortem_Brain_2024_8_30_dwx.csv
```

Columns: `IID`, `age_at_baseline` (numeric age), `SEX`, `case_control_other_latest` (Case/Control/Other). Historical audit: 71 Case, 25 Control, 4 Other across 100 rows; verify these counts in the current source file before processing. Exclude `Other`; never turn Other into Control, and do not infer covariates from genotype.

H5AD folder: `/public2/data/duwenxuan/AMP_PD_anno/try/` with `GPI`, `DMNX`, `PFC`, `PMC`, `PVC` `_subtype_count_try.h5ad` files. In H5AD, `obs['IID']` and `obs['cell_type']` identify donors and cell types; `var['gene_name']` is used to merge duplicated gene symbols with raw count summation (unannotated genes retain their Ensembl IDs).

Original genotype VCF: `/public/labdata/public_data/AMP_PD_Postmortem_Brain/vcf/merged.100s.vcf`, target rs2253588 (REF G ALT C) and rs35162690 (REF TC ALT T). Carrier is always VCF ALT dosage >0, not the PLINK A1 allele. A sample name `0_PM-HA_S00447-BLM0-PVC-DWGS` maps to clinical/H5AD `PM-HA_S00447-BLM0`.

Seven cell types: Oli, Immune, Inh, OPC, Ast, Vast, Exc. Other diagnoses are excluded, and nonlisted cell types are excluded. Keep all valid expressed genes: PyDESeq2 and HC3 pseudobulk across all genes, cell LMM top selected genes only. Compare both pooled Case+Control (adjust age, sex, diagnosis) and PD Case-only (adjust age, sex).

## Installation of the official module

The module zip has paths relative to the code directory. To install, upload it to a path on the server and run:

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
ZIP=$ROOT/2sGenGPS_postmortem_scDE_v6_2_clinical_fixed_module.zip
cp -a "$CODE/scripts_snp_scDE" "$CODE/scripts_snp_scDE_before_v6_2_$(date +%Y%m%d_%H%M%S)"
unzip -oq "$ZIP" -d "$CODE"
```

Run a light audit without touching the VCF/H5AD or writing analysis files:

```bash
python "$CODE/scripts_snp_scDE/test_postmortem_scde_v6.py"
python "$CODE/scripts_snp_scDE/test_postmortem_clinical_columns_v6_2.py"
python "$CODE/scripts_snp_scDE/test_mock_h5ad_v6.py"
python "$CODE/scripts_snp_scDE/test_prepare_resume_v6_1.py"
```

Verify the clinical table *on the server*:

```bash
python - <<'PY'
import pandas as pd
p='/public2/data/luojunfeng/2sTGCS_binary/scQTL/AMP_PD_Postmortem_Brain_2024_8_30_dwx.csv'
x=pd.read_csv(p)
print('path:',p,'rows:',len(x));print('columns:',x.columns.tolist())
required=['IID','age_at_baseline','SEX','case_control_other_latest']
assert set(required).issubset(x.columns)
print('clinical groups:',x.case_control_other_latest.value_counts(dropna=False).to_dict())
print('duplicate ID:',x.IID.astype(str).duplicated().sum())
print('missing age:',x.age_at_baseline.isna().sum())
print('missing sex:',x.SEX.isna().sum())
PY
```

## Official run, resume same output directory

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
RUN=$CODE/scripts_snp_scDE/run_snp_scDE.sh
OUT=$ROOT/single_iv_causal/postmortem_snp_scDE/carrier_vs_noncarrier_v6_vcf_verified

# Only if vcf_genotypes.csv does not exist. The original 10.75-GB VCF is scanned once.
test -s "$OUT/vcf_genotypes.csv" || bash "$RUN" audit --out "$OUT"

# New v6.2 marker will force re-preparing any prior v6/v6.1 region even if files exist.
bash "$RUN" prepare --out "$OUT" --skip-prepared --min-cells 30 --chunk-size 3000

# Stage of interest: all expressed genes; PyDESeq2 0.4.12 + HC3.
bash "$RUN" fit --out "$OUT" --methods both --n-cpus 4 --min-group 5
bash "$RUN" summarize --out "$OUT"

# Optional: top-n gene cell-level donor random intercept validation
bash "$RUN" cell-lmm --out "$OUT" --top-n 30 --max-cells-per-donor 30
```

Prior v6/v6.1 pseudobulk data cannot be silently reused because gene_name collapse and clinical covariate schema changed. Once a region has a v6.2 marker, `--skip-prepared` safely resumes subsequent interrupted runs. Previous Stage-I/II code and outputs are untouched. For safety, if existing v6 output must be preserved, pass a new `--out` folder and rerun `audit` to initialize that directory.

## Validation restrictions

This release is tested with synthetic data. It does not claim access to the user's cluster; the real clinical counts (historically 71 Case / 25 Control / 4 Other), WGS matching, GT call missingness, full pseudobulk sample numbers and PyDESeq2 execution must be confirmed on-server. All donor covariates are sourced from the actual clinical table, not guessed. All p-values are association tests and are not a replacement for Stage-I/II causal evidence.
