# 2sGenGPS Supplementary Figure 6 — real-data CTAR refit / donor-bootstrap v3

**Scope: only overall temporal cis-tQTL.**  The entire package tests the original
**8-df** genetic trajectory block. It NEVER evaluates a separate dynamic-QTL,
genotype × time interaction or minimum required sample size.

This replaces the earlier observed-LRT -> noncentral chi-square empirical-Bayes
approximation with **participant-level resampling and repeated 8-df CTAR GLS
likelihood-ratio fitting**. It returns the number and fraction of sampled real
PPMI temporal tQTLs and PDBP matched tQTLs whose bootstrap re-detection rate is
at least 80%, in the spirit of the manuscript's original Supplemental Fig. S6.

## Method and interpretation

- Sample **5,000 distinct, BH-significant, overall PPMI tQTL SNP–gene pairs**
  uniformly without replacement. Keep PPMI denominators separate from the
  subset matched to PDBP. Sample from only the confirmed final Stage-I output.
- Use actual PPMI and PDBP RNA sample IDs, participant IDs, follow-up times,
  the matched PPMI/PDBP PLINK genotypes, observed expression and nuisance covariates.
- Alternative: common cubic follow-up trajectory + genotype hard-call 1 × cubic
  + genotype hard-call 2 × cubic + covariates. Null: same without all eight
  genetic coefficients. Test **overall 8-df likelihood ratio**.
- Within each person, residual covariance is `Corr(e(t),e(s))=rho**abs(t-s)`
  with time in **years**. Use OU innovations for arbitrarily irregular intervals
  and missing visits (no imputation).
- Default main loop **freezes the actual fitted Stage-I rho**, then re-estimates
  all fixed-effect coefficients and residual variance in every resample. This is
  a **conditional, fixed-rho approximation to the CTAR likelihood refit**, not the
  production GPU scan with its 100-point rho profile search. `--profile-rho
  --rho-points 100` can be used on a smaller validation subset to explicitly
  profile rho under H0 and H1. The default analytical chi-square 8-df reference
  is the same asymptotic test family, but exact scan identity must be audited.
- Draw participant IDs with replacement, keeping *all* visits and all
  participant-level genotype measurements together, recompute the LRT, compare
  its raw P value against the **observed full-PPMI global BH rejection P cutoff**
  for PPMI, or nominal **P<0.05** for PDBP. The default is **1,000 replicates
  per locus and cohort**, as in the original figure's resampling objective.
- To reduce accidental publication of wrongly matched inputs, the pipeline
  reports and **tests actual refit LRT versus the original Stage-I LRT**.
  It will stop before final figures if this fundamental comparison fails.
- The quantity is a **conditional bootstrap detection / stability fraction**
  in the already-selected tQTL universe, **not an unbiased estimate of true
  prospective power**. Discovery selection can inflate empirical effects
  (winner's curse). The figure legend uses this precise language.

## Installation: exact project layout

Upload `2sGenGPS_CTAR_SFig6_refit_v3.zip` to:

`/public2/data/luojunfeng/2sGenGPS_revised_chatgpt/`

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
mkdir -p "$CODE/scripts_figures/tqtl_power" "$ROOT/logs"
unzip -oq "$ROOT/2sGenGPS_CTAR_SFig6_refit_v3.zip" -d "$CODE/scripts_figures/tqtl_power"
cd "$CODE/scripts_figures/tqtl_power/2sGenGPS_CTAR_SFig6_refit_v3"
python -m pip install -r requirements.txt
python -m pytest -q tests/
```

All outputs are written to this *new* directory, not to the original Stage-I files:

`$ROOT/manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_refit_v3/`

## Data inputs and exact field matching

Defaults inherited from your `v8.9` project:

```
$ROOT/qtl/PP/submitted_followup_ctar_mptga/final/significant_tqtl.parquet
$ROOT/qtl/PD/submitted_followup_ctar_mptga/final/all_results.parquet
$ROOT/data/metadata/amp_pd_samples.parquet
$ROOT/data/expression/amp_pd_expression_z.parquet
$ROOT/data/genotype/PP/amp_pd_PP_qc.bed/.bim/.fam
$ROOT/data/genotype/REP/amp_pd_REP_qc.bed/.bim/.fam
```

Reads **Ensembl_ID**, **Variant_ID**, **Pvalue**, **FDR**, **LRT**, **rho**,
`n_covariates_used` from the original Stage-I parquet when present. Does not
mistakenly require `gene_id` / `variant_id`. PDBP uses its own observed `rho` and
Pvalue. The PLINK 2 `--export A` columns are read using actual variant ID and
allele suffix; we never fabricate MAF, reverse complements, SNP IDs or genotypes.
Both discovery and replication use the two-channel *hard-call* genotype 0/1/2
model (not an additive one-parameter substitute).

**Important requirement on covariates:** the current-v8.9 Stage-I uses baseline
age, sex, diagnosis, RIN, 8 genotype PCs, FULL cohort expression PCs and six
LM22 CLR covariates. This pipeline discovers matching numeric metadata columns;
if the metadata parquet lacks external nuisance-PC/LM22 covariates, provide their
sample-aligned parquet/CSV using repeated `--cov-file` options. This package
will NOT silently invent PCs or claim equivalence to the production model.
Run the preflight audit and compare `n_covariates_used` from the original scan.

Additional sample metadata column aliases can be overridden with
`--sample-col SAMPLE_ID --participant-col PATNO --time-col followup_time
--cohort-col cohort`. `followup_time` MUST be in years. If cohort coding or
sample IDs do not align with genotype IID, the program fails with a QC message
instead of guessing a join. RNA input parquet can be long
(`sample_id, Ensembl_ID, expression`) or wide (`sample_id` + gene columns).
Check the detected expression mode in `input_model_audit.json`.

## Step 1 — prepare 5,000 locus list + export PLINK dosages

```bash
export ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
export CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
cd "$CODE/scripts_figures/tqtl_power/2sGenGPS_CTAR_SFig6_refit_v3"

bash run_sfigure6.sh prepare
```

This calls PLINK 2 (must be on PATH). If your actual PDBP prefix differs from
`data/genotype/REP/amp_pd_REP_qc`, specify the real existing files:

```bash
bash run_sfigure6.sh prepare \
  --pd-genotype "$ROOT/data/genotype/PD/<actual_PDBP_PLINK_prefix>"
```

This is a **placeholder path for an override**, not a claim that such a file
exists. Locate your real prefix by `find "$ROOT/data/genotype" -name '*.bed'`.

Outputs: `selected_loci.csv`, `selection_audit.json`,
`genotypes_PP.raw`, `genotypes_PD.raw`.

### Optional — cache the selected real-expression matrix in one pass

For 5,000 loci, repeated reads of the original multi-gene expression parquet
can dominate runtime. **After `prepare`**, create a compact cache of only the
selected genes (the same values, without re-normalization):

```bash
bash run_sfigure6.sh cache-expression
```

Both preflight and full runs automatically use this subset parquet when it
exists. The exporter supports long, wide and gene-row parquet layouts; it
stops if selected gene IDs cannot be matched rather than silently dropping
unmatched genes. It never changes the original expression file.

## Step 2 — obligatory 30-locus / 100-bootstrap preflight

```bash
bash run_sfigure6.sh preflight
cat "$ROOT/manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_refit_v3/_preflight/fit_audit.json"
```

**The audit must pass before the 5,000-locus run.** If you have external
nuisance files, add these to BOTH preflight and full steps, for example:

```bash
bash run_sfigure6.sh preflight \
  --cov-file /path/to/stage1_sample_covariates.parquet
```

When `Stage-I n_covariates_used` disagrees with those available, restore the
true Stage-I covariate matrix (or inspect the Stage-I preprocessing script).
Do not use `--allow-covariate-mismatch` for the final manuscript analysis.
The preflight uses a SEPARATE output directory, so its 100-bootstrap estimates
never contaminate the full 1,000-bootstrap results.

Audit limit `median |LRT_refit - LRT_original| <= 2` is a fail-fast *screen*,
not proof of exact numerical equivalence: production may independently profile
rho under null and alternative; fixed-rho can differ. For close agreement,
run the smaller validation with `--profile-rho --rho-points 100` and compare.
If agreement fails, align **mean model, covariates, sample IDs, genotype coding,
time scaling and rho profiling** before reporting power.

## Step 3 — full 5,000 × 1,000 (checkpoint/resumable)

```bash
nohup bash run_sfigure6.sh full \
  > "$ROOT/logs/sfig6_CTAR_refit_v3.log" 2>&1 &
tail -f "$ROOT/logs/sfig6_CTAR_refit_v3.log"
```

The script outputs one cohort-locus row at a time; `--resume` skips already
completed results. Failed locus/cohort evaluations are logged in
`locus_failures.csv`, **not quietly discarded**. After a fix, rerun with `--resume`.

Options: `BOOTSTRAPS=200` for a preliminary pilot; **use 1,000 for the final**.
`SUBSAMPLE_PAIRS=400 SUBSAMPLE_REPS=100` defaults for actual participant
subsampling on an *explicitly labeled* subset of loci. The main 5,000 loci are
evaluated in full PP and matched PD independently, not artificially reduced
to the PDBP-matched denominator.

For a final rho-profile sensitivity subset, use a DIFFERENT `--outdir` and
provide the existing selected_loci.csv and genotype .raw paths. Example:

```bash
OUT=$ROOT/manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_refit_v3
python 02_run_power.py --root "$ROOT" \
  --outdir "$OUT/_rho_profile_30" --selected-file "$OUT/selected_loci.csv" \
  --pp-raw "$OUT/genotypes_PP.raw" --pd-raw "$OUT/genotypes_PD.raw" \
  --alpha-disc "$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["observed_max_raw_p_of_all_hits"])' "$OUT/selection_audit.json")" \
  --max-pairs 30 --bootstrap-reps 200 --sensitivity-reps 0 \
  --profile-rho --rho-points 100
```

This is computationally much more expensive than frozen-rho bootstraps, and
should not be represented as the production scanner unless all nuisance axes
and identical likelihood conventions are verified.

## Step 4 — QC and figure

```bash
bash run_sfigure6.sh plot
```

Outputs:

- `locus_bootstrap_results.csv`: one tested cohort–SNP–gene pair per row,
  with observed and resampled overall LRT, MAF, genotype counts, actual donors,
  valid bootstrap draws and empirical conditional detection probability.
- `locus_failures.csv`: exclusions and their exact reasons (important denominator QC).
- `fit_audit.json`: Stage-I LRT reproduction check.
- `donor_sample_size_sensitivity.csv`: **real donor** subset refit results for
  selected loci at 25, 50, 75, 90 and 100% cohort size.
- `matched_pair_power.csv`: PPMI/PDBP matched conditional detection fractions.
- `power_summary.json`: how many loci have estimated bootstrap detection ≥80%.
- `Supplementary_Figure_6_CTAR_refit.{png,pdf,svg}`:
  (a) two actual resampling LRT distributions around 80/90%; (b) PPMI histogram;
  (c) PDBP histogram; (d) participant-subset stability; (e) observed independent
  PDBP replication and sensitivity; (f) summary fractions.

All 3 figure formats have text/vector support; the output is for the *observed
selected tQTL universe*, not for all true cis regulatory effects.

## Performance

The OU innovation approach produces small per-subject sufficient-statistic
matrices (`X'X`, `X'y`, `y'y`) and uses bootstrap multiplicity weights on
**participants**, avoiding a large RNA row-by-row linear regression each
resample. It is much faster than 1000 full design re-materializations per
locus. Computational performance depends on number of nuisance PCs and rare
second genotype groups. Main/frozen-rho and slow/profile-rho results should be
reported as separate sensitivity analyses.

## Research integrity / limitations

1. **Observed bootstrap detection ≠ prospective power.** Selected PPMI effects
   are subject to winner's curse. If the user wishes to generalize to unseen
   small effects, a prespecified alternative-effect simulation is required,
   which is a different scientific question and is not plotted here.
2. A bootstrap PPMI raw P cutoff fixed to the original genome-wide BH rejection
   boundary is a *conditional reference*. This script does NOT redo all 77M
   cis tests and recompute BH for every resample.
3. The distribution and number of *eligible* SNP-gene pairs must be recorded;
   sparse genotype classes, lack of cross-cohort SNP matching, failed nuisance
   alignment or singular regressions affect the eligible denominator.
4. Full-cohort observed PDBP replication P<0.05 is a separate actual outcome;
   it must not be described as 80% power.
5. A 90% percentile interval from participant resampling measures bootstrap
   uncertainty of detection stability, not genomic causal effect size.
6. The current standalone refitter has NOT been tested against original raw
   PPMI/PDBP Stage-I server files (they are inaccessible to this environment).
   This is why the hard LRT audit is included rather than claiming numerical
   identity without evidence.

## Original manuscript methodological correspondence

The original article describes 5,000 temporal cis-eQTLs, 1,000 resampled
subject-ID bootstraps and LRT rejection fractions. Our implementation improves
reproducibility by fixing the exact gene/SNP file schema, preserving independent
PPMI/PDBP denominators, correctly resampling visits in donor blocks, and
logging exact model mismatches before interpreting results.

Original publication: *Nonlinear dynamic genetic regulation identifies peripheral
drivers of neurodegenerative disease progression* (2026), power Methods /
Supplemental Figure S6. Winner's curse caution: Huang et al., *Nucleic Acids
Research* 2018, “Power, false discovery rate and Winner's Curse in eQTL studies.”
