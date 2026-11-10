# Supplemental Figure 6 — empirical power of existing temporal tQTLs

**No dynamic QTL analysis.** This project reports overall 8-df CTAR-MPTGA temporal tQTLs only.

## Target
Among existing PPMI significant temporal tQTLs, draw a reproducible sample of 5,000 SNP–gene pairs; use matched PDBP overall CTAR statistics. Report the observed fraction with *model-estimated* ≥80% discovery power, ≥80% replication power, and ≥80% joint power, as well as size-reduction sensitivity. PDBP unmatched loci are excluded from paired percentages and counted separately. No prospective N80 sample-size planning is performed.

## Input
Defaults:
- `$ROOT/qtl/PP/submitted_followup_ctar_mptga/final/significant_tqtl.parquet`
- `$ROOT/qtl/PD/submitted_followup_ctar_mptga/final/all_results.parquet`

Uses real `Variant_ID`, `Ensembl_ID`, `Pvalue`, `LRT`, `FDR`, `n_participants`, `n_obs` fields when present. **Does not require gene-expression or PLINK files** for this approximate analysis. Input files must be from the same finalized CTAR scan and contain the overall genetic trajectory (8-df) LRT, NOT time-interaction statistics.

## Method
The power of an overall 8-df likelihood-ratio test is approximated by the survival function of a noncentral χ² distribution. LRTs of PPMI-selected hits are biased upward; it is incorrect to calculate power simply as `ncx2.sf(threshold,8,max(LRT-8,0))` and report it as an unbiased result. This script fits a regularized discrete empirical-Bayes prior on the noncentrality parameter from all *unselected-on-PDBP* PDBP overall LRTs of the PPMI-discovered loci. It combines independent PDBP evidence with the PPMI selection-truncated likelihood to obtain a posterior noncentrality distribution for each matched locus. It calculates posterior-predictive power in PPMI and PDBP at the **actual** raw PPMI discovery cutoff and nominal PDBP P < 0.05.

For transfer of noncentrality between cohorts, the primary analysis assumes comparably informative visit designs and uses the locus-specific independent-participant-count ratio (`--info-scale participants`). Sensitivity: `--info-scale observations`; differing follow-up schedules and OU residual covariance may violate both assumptions. These are approximations, not full CTAR profile likelihood re-fits; **do not describe as a ctAR 1,000-replicate donor bootstrap**. Further sensitivity should simulate RNA from the fitted 8-parameter ctAR model with empirical visit times and re-fit ctAR on an independent subset when available.

PPMI selection posterior term is `f_NCX2(T_PP; 8, λ_PP)/P(T_PP ≥ c_PP | λ_PP)`; PDBP term is untruncated. Group prior is fitted on all PDBP-tested selected PPMI loci. Default sample N=5,000. Cluster bootstrap 95% CI samples **genes**, not independent SNPs, partially acknowledging cis-LD and repeated variants. The joint power is the posterior average of the product of the two conditional cohort powers; this is an approximation conditional on the selected signals and cohorts, not observed replication success. The `power_joint` percentage is specifically among matched pairs; separate `both_components_80` counts loci meeting both per-cohort 80% criteria.

**Interpretation note:** Realized PDBP replication rate is not identical to the expected/estimated power of a single prespecified variant; report both without equating them. Because selected PPMI hits and the PDBP likelihood both enter the posterior, an internal comparison of predicted PDBP power with observed replication is descriptive, not independent calibration.

## Installation and run
```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
unzip -oq "$ROOT/CTAR_SFig6_empirical_power_v2.zip" -d "$CODE/scripts_figures/tqtl_power/"
cd "$CODE/scripts_figures/tqtl_power/CTAR_SFig6_empirical_power_v2"
python -m pip install -r requirements.txt
python -m pytest -q tests
nohup bash run_empirical_power.sh > "$ROOT/logs/sfig6_ctar_empirical_power.log" 2>&1 &
tail -f "$ROOT/logs/sfig6_ctar_empirical_power.log"
```
Run a sensitivity on observed-count scaling:
```bash
bash run_empirical_power.sh --info-scale observations --output "$ROOT/manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_empirical_power_v2_obs"
```

Outputs `summary.json`, `locus_power_PP_PD.csv`, `prior_lambda_grid.csv`, `power_by_sample_fraction.csv`, and 300-dpi PNG/PDF/SVG in `$ROOT/manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_empirical_power_v2`.

## Constraints
If 5,000 sampled PPMI tQTLs contain fewer than 1,000 PDBP-tested pairs, the script aborts; do not silently treat unmatched as nonsignificant. If n_participants or n_obs are unavailable, it errors rather than guessing. The full output should record the exact numerator, denominator, assumptions and CI. The scan-wide actual BH boundary should be validated against `fdr_finalize_summary.json` if available; the script uses the maximum raw P among the **complete PPMI significant-tQTL file** as a conservative empirical discovery cutoff.
