# Supplementary Fig. 6 — practical CTAR sample-size analysis (v1.3)

## Why this exists

Previous v1.2's **N80/80** demanded `80% of sampled significant tQTLs each achieve 80% joint discovery+replication power`. This is a **strong-cohort-coverage** design objective, **NOT the minimum N required to do a longitudinal eQTL study**. The PPMI/PDBP locus distribution used in that model is discovery-selected and biased towards strongly detectable loci and particular MAFs; its result must never be claimed as a universal sample requirement.

v1.3 therefore reports separately:

* **Primary**: `N80` for **one prespecified variant-gene temporal tQTL** (8-df overall CTAR LRT) at a fixed prospective PVE and MAF, with 80% statistical power.
* **Scenario A**: nominal test alpha=.05 (adequate only for one prespecified test, **not a genome-wide discovery claim**).
* **Scenario B**: the real PPMI full-scan BH raw-P cutoff as a **fixed reference threshold** for stricter screening. The BH threshold itself **changes with study N and the ensemble of other tests**, so this is NOT the expected FDR=q=.05 at every N. For rigorous prospective genome-wide FDR power, the full scan and multiple-testing procedure must be simulated jointly at each N.
* Independent replication: nominal alpha=.05 in PDBP, holding the PPMI discovery N fixed at the actual value (not doubling both Ns by default); separate joint discovery+replication power.
* Dynamic tQTL: overall 8-df detection **AND** 6-df interaction confirmation at `0.05*R/G`; sensitivity panel, not the default definition of any temporal tQTL.

## Limits (must report in legend and Methods)

Calculations use exact OU/continuous-time AR innovation covariance at **fixed** rho and the noncentral chi-square asymptotic likelihood-ratio approximation; they do **not** run the production CTAR 100-point profile-rho GPU likelihood at each N. The simulated genetic alternative is one dosage-like trajectory embedded in the full two-channel 8-df test (not all possible two-channel genetic trajectories). PVE is normalized to a five-visit 0–4-year reference schedule at MAF 0.2. Genotype coding / minor homozygote sparsity can invalidate the asymptotic approximation at low MAF. This version **does not** set power artificially to zero when expected minor homozygotes <5; it flags that part of the MAF curve as uncertain instead. For publishable N values, validate selected settings with full 8-df CTAR profile LRT Monte Carlo and actual observed genotype frequencies.

The user-provided, real CTAR Stage-I format is `Variant_ID`, `Ensembl_ID`, `Pvalue`, `Interaction_Pvalue`; it is handled by the copied base helper in this folder. Real thresholds are reconstructed from `$ROOT/qtl/PP/submitted_followup_ctar_mptga/final/fdr_finalize_summary.json` or verified full overall P values, never guessed. Real PPMI/PDBP visits are loaded from `$ROOT/data/metadata/amp_pd_samples.parquet`. If the prior SFig6 locus MAF CSV exists, it is reused to skip expensive PLINK MAF extraction. Otherwise this version computes from `$ROOT/data/genotype/PP/amp_pd_PP_qc.{bed,bim,fam}`.

## Install without overwriting previous SFig6

Upload `CTAR_SFig6_practical_v1_3.zip` into the project `$ROOT` directory:

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
mkdir -p "$CODE/scripts_figures/tqtl_power" "$ROOT/logs"
unzip -oq "$ROOT/CTAR_SFig6_practical_v1_3.zip" -d "$CODE/scripts_figures/tqtl_power"
cd "$CODE/scripts_figures/tqtl_power/CTAR_SFig6_practical_v1_3"
# Your existing ctAR Python environment must contain matplotlib numpy pandas scipy pyarrow bed-reader
python -m pytest -q tests/
nohup bash run_practical.sh > "$ROOT/logs/sfig6_practical_v1_3.log" 2>&1 &
tail -f "$ROOT/logs/sfig6_practical_v1_3.log"
```

For local **simulation only** (not empirical manuscript numbers):

```bash
bash run_practical.sh --demo --output /tmp/sfig6_demo --max-n 3000 --step-n 100
```

To change effect sizes, e.g. PVE 1,2,4,8% (use **fractions** in flags):

```bash
bash run_practical.sh --pves .01 .02 .04 .08 --focus-pve .02 --max-n 5000 --step-n 50
```

Outputs under `$ROOT/manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_practical_v1_3/`, without modifying old output files:

- `Supplementary_Figure_6_CTAR_practical.{png,pdf,svg}`
- `N80_by_PVE_and_significance_policy.csv`
- `N_by_PVE_power_curves.csv`
- `N80_by_visits_and_rho.csv`
- `N80_by_MAF.csv` (including flags for sparse genotype categories)
- `replication_N_with_fixed_actual_PPMI.csv`
- `analysis_manifest.json` with true cohort N and all thresholds.

## Suggested legend gist

**Supplementary Figure 6 | Prospective sample-size considerations for CTAR-MPTGA.** (a) Power to detect a single prespecified temporal cis-eQTL across independent discovery-cohort sizes, for five assumed genetic PVE levels. Solid lines represent nominal P<0.05; dashed lines represent the observed full-scan PPMI BH raw-P cutoff used as a fixed reference significance threshold. (b) Corresponding minimum sample size for 80% detection power. (c) Sensitivity to number of repeated observations and residual annual correlation. (d) Sensitivity to SNP MAF; low expected minor-homozygote counts are highlighted and asymptotic estimates may be unreliable. (e) Conditional replication power as PDBP cohort size increases while PPMI size is fixed at its observed value, versus joint discovery-and-replication power. (f) Comparison of temporal tQTL discovery, dynamic tQTL confirmation, and joint discovery-and-replication. Estimates use real cohort visit structures (where available), scenario-defined PVE rather than discovery-selected effect estimates, and fixed-rho OU asymptotic power; they are not direct full CTAR GPU profile-likelihood simulation estimates. Nominal power does not assert genome-wide FDR control.
