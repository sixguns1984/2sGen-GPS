# Supplementary Figure 6 — CTAR prospective sample size (final, 2026-10-30)

## 1. Scientific answer

The old Figure S6 mixed two different questions:

1. **Retrospective feasibility:** How many PPMI-discovered tQTLs replicate in the currently available PDBP data?
2. **Prospective design:** How many **independent participants**, with a specified number and timing of RNA collections, are necessary to detect a *predefined effect* using CTAR and independently reproduce it?

Those cannot be answered using the same significance-selected effect coefficients. Significant discovery coefficients are upward biased by selection (winner's curse). The old `n≈400 significant pairs` fit-derived PVE panel was conditional and could not justify a universal minimum N. **This version does not estimate future effect sizes from the significant hits.** It samples **1,500 distinct real tQTL pairs by default (>=1,000 passing MAF QC mandatory)** only to capture the actual observed MAF distribution and the effect of the two-channel CTAR genetic coding. A clearly labeled *assumed* PVE grid supplies the biological effect size. Real PPMI/PDBP participant-visit schedules determine the longitudinal information.

The primary result is a conditional **N80/80**, defined as:

> The smallest *equal independent participant count per cohort* where at least 80% of sampled loci each have >=80% predicted *joint* power to pass the **PPMI discovery** and **PDBP replication** tests, for a stated PVE, reference MAF, visit schedule, rho, replication effect retention, and raw P-value thresholds.

There is **no universal minimum sample size** for CTAR: weak effects, rare homozygotes, strong correlation, fewer observations, stringent multiple testing, and attenuation across studies affect the required N. Distinguish `N80/80` from the purely algebraic number of observations needed to fit a regression.

## 2. Statistical fidelity to the production model

The original Stage-I `submitted_followup_ctar_mptga` model is:

```
y_ij = covariates + common cubic(t_ij)
     + I(G_i=1) × [1, t_ij, t_ij², t_ij³]
     + I(G_i=2) × [1, t_ij, t_ij², t_ij³] + epsilon_ij
Corr(epsilon_ij,epsilon_ik) = rho ^ abs(t_ij-t_ik)    (annual time unit)
```

- Overall temporal tQTL: **8-df** joint genetic test.
- Dynamic tQTL: 8-df overall test **and** **6-df** genotype-by-time confirmation.
- PPMI overall discovery: **raw genome-wide BH cutoff** (`--alpha-disc`; it is NOT FDR adjusted p = 0.05).
- PPMI dynamic confirmation: **stagewise** raw `alpha_II = 0.05 × R/G` (`--alpha-inter`), with `G` finite overall tests and `R` BH-screened pairs.
- PDBP replication: **nominal raw p < 0.05** for overall tQTL, and both overall and dynamic interaction nominal p < 0.05 for dynamic replication.
- PPMI/PDBP samples are independent, hence the **joint power** is `Power(PPMI) × Power(PDBP)`.
- Correlation uses exact OU/AR continuous-time innovation precision for the original irregular times. There is no pretense that correlated visits are independent donors and no interpolation of missing visits.

The **main grid** uses the noncentral chi-square approximation to the CTAR tests *at a fixed specified rho*. For the nested 8-/6-df tests, the joint probability is calculated from the dependent chi-square decomposition, **not by multiplying the two P values or their marginal powers**. The relative genetic signal is an *additive data-generating effect* represented within the production's two-channel eight-parameter model; this is a deliberately specified prospective alternative, not an assumption that the statistical test has one df.

**Important model limitation:** The primary grid is an analytical **design approximation**, not exactly the original GPU full likelihood with a profiled 100-point rho grid. Additional clinical, genotype-PC, expression-PC and cell-composition nuisance effects are assumed independent of genotype after adjustment; population stratification and finite-sample nuisance estimation may lower power. The companion `ctar_profile_mc_validate.py` uses the cubic/two-channel OU profile-LRT (with adjustable 100-point rho grid) for numerical sensitivity and should be run for the N80 boundary before using a specific number in the manuscript. Its simulated nuisance covariates are not automatically the actual production covariate matrix. Run final boundary confirmation using the original Stage-I CTAR scanner too, once exact inputs are available.

## 3. PVE and MAF scale — essential to interpretation

PVE scenarios default to **0.25%, 0.5%, 1%, 2%, and 4%**, defined relative to genetic dosage variance at reference **MAF 0.20** and **five equally spaced observations over 4 years** (for the selected trajectory shape). Signal amplitude remains **fixed across PPMI and PDBP** and across visits/rho comparisons. Thus lower MAF entails lower true PVE and weaker signal; we do **not** improperly force every MAF bin to have the same PVE. For reference MAF p0=0.2, a locus with allele frequency p has genotype variance ratio `2p(1-p)/(2p0(1-p0))`.

Main trajectory: `linear`, f(t)=t/4. Other cubic forms are `early=3u-3u²+u³` and `late=3u²-2u³`, with u=t/4. Adjust `--horizon` if your planned observation horizon is not 4 years. The code saves `N_by_trajectory_shape.csv` to quantify shape sensitivity. The MAF/HWE guard requires at least **5 expected minor homozygotes and 5 expected heterozygotes** per cohort; this is a *screening safeguard*, not a guarantee of full rank or a replacement for real genotype counts. Can be varied using `--min-minor-hom`.

## 4. Final panels (2 × 3)

| Panel | Scientific question | Metric |
|---|---|---|
| **a** | At each sample size and assumed PVE, what proportion of 1,500 real-MAF loci has >=80% *joint* discovery+replication power? | Curves of fraction vs **participants per cohort**, with 80% horizontal rule and observed cohort reference lines |
| **b** | What is the **minimum N** needed for ctAR? | **N80/80** vs assumed PVE, with N not attained clearly indicated, plus optional dynamic-tQTL point |
| **c** | How do planned visit count and residual correlation change minimum N? | Heatmap for 2/3/4/5 planned visits and annual rho 0/0.3/0.6/0.9. **Same follow-up horizon**, 15% post-baseline dropout, no imputation |
| **d** | Is N different for rare vs common variants? | Empirical MAF bins with actual number of sampled loci and N80/80. Values beyond the scanned range are **censored (`>max`)**, never shown as successful convergence |
| **e** | What if true effects are weaker in PDBP? | N80/80 at 50%, 75%, 100% replication effect *amplitude* relative to PPMI, across PVE levels |
| **f** | Does detecting dynamic tQTLs demand a different N than detecting temporal tQTLs? | 8-df temporal vs stagewise 8+6-df **dynamic** joint power, using the same PVE. Box gives **actual nominal PDBP replication rate among sampled matched tQTLs**, if available |

`N80/80` is a **per-cohort** recommendation under balanced discovery/replication sizes. Also inspect `modeled_power_at_actual_cohort_sizes.csv` to evaluate your real *unequal* PPMI/PDBP N. `discovery_only_N80_80` and `replication_only_N80_80` are separately provided in the min-N CSV. Bootstrap CIs resample **genes as blocks**, not individual RNA visits; these intervals describe locus-composition uncertainty within the **selected discovered tQTL set**, not a population-general confidence interval.

## 5. Original project input/output — no upstream script changes

Default root and inputs are compatible with the existing v8.9 project:

```
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
$ROOT/qtl/PP/submitted_followup_ctar_mptga/final/significant_tqtl.parquet
$ROOT/qtl/PP/submitted_followup_ctar_mptga/final/fdr_finalize_summary.json
$ROOT/qtl/PD/submitted_followup_ctar_mptga/final/all_results.parquet
$ROOT/data/metadata/amp_pd_samples.parquet
$ROOT/data/genotype/PP/amp_pd_PP_qc.bed/.bim/.fam   (only if locus MAF is missing)
```

Outputs (additive: nothing in original scan directories is overwritten):

```
$ROOT/manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_N80_final/
  Supplementary_Figure_6_CTAR_N80_80.{png,pdf,svg}
  loci_1500_real_or_demo.csv
  power_grid.csv
  min_N80_80_by_PVE.csv
  modeled_power_at_actual_cohort_sizes.csv
  N_by_visits_rho.csv
  N_by_MAF_bin.csv
  N_by_crosscohort_retention.csv
  N_by_trajectory_shape.csv
  dynamic_power_focus_PVE.csv
  analysis_manifest.json
```

### Install dependencies in the existing environment

```bash
python -m pip install -r requirements.txt
```

`pyarrow` is needed for actual source parquets; `bed-reader` is needed only when the discovered tQTL parquet does not contain MAF. If already available in the conda environment, no reinstall is needed.

### Run the final analysis (updated: numeric thresholds resolved from the actual CTAR scan)

After uploading `CTAR_SFig6_N80_final_v1_1_fix.zip` to `$ROOT`, install **without
creating a duplicate nested directory**:

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
INSTALL=$CODE/scripts_figures/tqtl_power
mkdir -p "$INSTALL" "$ROOT/logs"
unzip -oq "$ROOT/CTAR_SFig6_N80_final_v1_1_fix.zip" -d "$INSTALL"
cd "$INSTALL/CTAR_SFig6_N80_final"
python -m pytest -q tests/
python check_stage1_thresholds.py --root "$ROOT"
unset BH_RAW_P ALPHA_INTER
nohup bash run_sfigure6_final.sh > "$ROOT/logs/sfigure6_ctar_N80_final.log" 2>&1 &
tail -f "$ROOT/logs/sfigure6_ctar_N80_final.log"
```

The wrapper ignores old nonnumeric placeholder values in `BH_RAW_P` or
`ALPHA_INTER`; no manual `export` is required. The Python program automatically
tries the Stage-I `fdr_finalize_summary.json` first, then recomputes the exact BH
screening cutoff and `alpha_II = 0.05R/G` from **unfiltered PPMI genome-wide**
`all_results.parquet`. It never silently uses a cutoff from a small selected or
replicated subset. If neither source can establish both cutoffs, the script stops
and prints precise missing file details instead of substituting made-up values.

**Do not set example/placeholder threshold numbers.** For manual overrides, only
supply real scan-derived `--alpha-disc` and `--alpha-inter` values, not FDR q=0.05.
The log and `analysis_manifest.json` record threshold sources and numeric values.

Alternate metadata columns or supplied MAF tables:

```bash
bash run_sfigure6_final.sh \
  --col-cohort cohort --col-participant participant_id --col-time followup_time \
  --loci-file /path/to/real_QTL_pairs_with_MAF.tsv \
  --plink-prefix "$ROOT/data/genotype/PP/amp_pd_PP_qc"
```

The input locus table must have **variant_id**, **gene_id**, **MAF** (case-insensitive aliases accepted), optionally raw overall discovery P. At least **1,000 real SNP-gene pairs with valid MAF** after filtering are mandatory. Sampling is without replacement and fixed seed. Allele frequencies and actual participant templates must never be silently replaced by synthetic defaults in `--demo`-free runs.

### Validate analytic N80 using actual ctAR-style profile likelihood

First run main figure; read `min_N80_80_by_PVE.csv` and choose the N corresponding to a specified PVE (and nearby N values). Then run:

```bash
python ctar_profile_mc_validate.py \
  --root "$ROOT" \
  --n 1500 --pve 0.01 --pattern linear --rho 0.5 \
  --alpha-disc "$BH_RAW_P" --alpha-inter "$ALPHA_INTER" \
  --n-loci 120 --replications 200 --rho-points 100
```

`--n 1500` and `--pve 0.01` here are **examples**, not results or recommended sample sizes. Use the `N80/80` identified by the real input run; repeat just below/at/above it. Save the two profile-LRT audit files and report the observed power difference from the analytic approximation. If the full-profile empirical power is lower than predicted at the proposed boundary, increase N accordingly. Increasing `--replications` narrows Monte Carlo uncertainty; 200 still gives an approximately ±5.5 percentage-point marginal 95% interval for a per-locus power of 0.8, so this validator is a sensitivity check rather than an exact 80/80 certification.

### Smoke test and automated unit tests

```bash
python -m pytest -q tests/test_power.py
python ctar_sfig6_n80.py --demo --n-pairs 1200 \
  --max-n 4000 --step-n 100 --bootstraps 10 --output ./DEMO_only
```

`--demo` generates entirely synthetic MAF/visits, **never** represents PPMI/PDBP. The synthetic demo is only for checking code and design. The package was tested in demo mode and passed its numerical unit tests; it has **not** been tested against the original AMP-PD parquet data in this environment (those data are not available here).

## 6. Editorial wording / what NOT to conclude

Acceptable: "Under an assumed full-trajectory genetic PVE of X%, the observed allele-frequency spectrum, five annual visits, annual residual correlation rho=Y, the actual discovery cutoff, and nominal replication testing, N participants per cohort were required to achieve the N80/80 design target. This threshold varied across effect size, visit number and allele frequency."

**Not acceptable:** "ctAR always requires at least N participants" or "our 1,400-person cohorts have 80% power for most true QTLs" without a justified true effect distribution. The real discovered set is subject to significance selection and cannot stand in for all genuine cis tQTL effects. Observed PDBP replication rate is a separate descriptive quantity, not an estimator of unconditional power in the new cohort.

The main scan uses the observed BH raw cutoff as a *design reference*; it does not re-run BH on an entire genome-wide synthetic scan for every future N, so exact prospective FDR at each N is not guaranteed. This limitation and finite-sample rho profiling should be stated in Methods when reporting results.

## V1.2 real Stage-I schema and WGS input correction (2026-10-30)

**Replaces V1.1**, fixing the observed crash:

```
Cannot determine column from ['gene_id',...]; available ... 'Variant_ID', 'Ensembl_ID', 'Pvalue', ...
```

V1.2 explicitly supports the production CTAR output columns:

- `Variant_ID` → internal `variant_id`
- `Ensembl_ID` → internal `gene_id` (in **both** PPMI discovery and PDBP replication)
- `Pvalue` → internal raw overall discovery/replication P
- `Interaction_Pvalue`, `FDR`, `Interaction_FDR_stagewise`, `beta_g_0...7` remain untouched

The actual Stage-I `fdr_finalize_summary.json` fields are supported:
`hierarchical_screening_n_tests_G`,
`hierarchical_screening_n_rejected_R`, and
`hierarchical_confirmation_alpha`.
If R is recorded and `significant_tqtl.parquet` contains **exactly R** finite
screened overall P values, the maximum P is the verified raw BH cutoff; the
code no longer needlessly sorts the enormous all-tests Parquet for this case.
No arbitrary discovery cutoff is invented.

The actual result files often **lack MAF**. V1.2 scans the original PPMI
`amp_pd_PP_qc.bim` as a stream, indexes only QTL variants needed for the
calculation, resolves exact IDs and unambiguous coordinates with matching
alleles when available, and extracts the true MAF from the corresponding
`amp_pd_PP_qc.bed` columns. This avoids loading the whole WGS BIM into RAM.
Crucially, after SNP mapping it samples eligible SNP-gene pairs and only
accepts frequencies computed from the actual PPMI genotype; it will not
silently assume MAF=0.20. It checks for at least 1,000 real-MAF loci, targets
1,500 pairs, and prints [INPUT] and [MAF] diagnostics to the log.

### Server install from the exact project paths

Upload **`CTAR_SFig6_N80_final_v1_2_real_schema_fix.zip`** to:
`/public2/data/luojunfeng/2sGenGPS_revised_chatgpt/`.

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE="$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized"
mkdir -p "$CODE/scripts_figures/tqtl_power" "$ROOT/logs"
unzip -oq "$ROOT/CTAR_SFig6_N80_final_v1_2_real_schema_fix.zip" -d "$CODE/scripts_figures/tqtl_power"
cd "$CODE/scripts_figures/tqtl_power/CTAR_SFig6_N80_final"
python -m pip install -r requirements.txt   # only if missing dependencies
python -m pytest -q tests/
# Preflight: thresholds, schema, MAF, cohorts and PDBP replication matching.
python audit_real_inputs.py --root "$ROOT" > "$ROOT/logs/sfig6_input_audit.log" 2>&1
cat "$ROOT/logs/sfig6_input_audit.log"
# Real run. Do not define any artificial discovery/interaction thresholds.
unset BH_RAW_P ALPHA_INTER
nohup bash run_sfigure6_final.sh > "$ROOT/logs/sfigure6_ctar_N80_final.log" 2>&1 &
tail -f "$ROOT/logs/sfigure6_ctar_N80_final.log"
```

**No Stage-I scans are repeated, no existing Stage-I/II files are modified.**
The preflight outputs `input_audit.json` under the existing SFig6 output
folder. Final figures remain in the original agreed folder.

Environment limitation of the development container: it lacks `pyarrow` and
`bed-reader`; regression tests exercised the real Stage-I schema using CSV
and a controlled BED decoder stub, plus the full end-to-end DEMO figure path.
Therefore the server's *real* Parquet/BED dataset cannot be declared already
successfully executed. Run the preflight against real files before publishing
results. The sample-size estimates remain fixed-rho analytical approximations;
run profile-likelihood MC for final boundaries as stated above.
