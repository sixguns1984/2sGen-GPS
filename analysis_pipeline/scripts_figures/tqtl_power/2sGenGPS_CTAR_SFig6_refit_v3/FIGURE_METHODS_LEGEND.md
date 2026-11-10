# Draft manuscript language (validate exact LRT first)

**Method.** We uniformly selected 5,000 overall temporal cis-tQTL SNP–gene pairs
meeting the genome-wide PPMI Benjamini–Hochberg FDR < 0.05 criterion. For each
pair and cohort, we resampled participants with replacement and retained all of
their observed longitudinal samples, genotypes and covariates. We re-estimated
the ctAR cubic genetic-trajectory mean model and a corresponding no-genotype
null model under continuous-time autoregressive/OU residual covariance. Our
primary implementation fixed the original cohort-specific estimated residual
correlation for computational efficiency and used the overall eight-degree-of-
freedom likelihood ratio statistic. We repeated each cohort-locus test 1,000
times. The proportion of re-fitted tests crossing the original PPMI BH-derived
raw-P cutoff or PDBP nominal P < 0.05 threshold was termed the **conditional
bootstrap detection fraction**. We report the number of evaluated pairs with
bootstrap detection fractions ≥80%, separately in PPMI and independently
matched PDBP pairs. Participant-subsample analyses were repeated on real
without-replacement subsets of PDBP. These values characterize detection
stability among discovered pairs; they should not be interpreted as a
prospective power estimate for all true temporal cis-tQTLs.

**Legend.** Supplementary Fig. 6. Donor-bootstrap detection stability of overall
temporal cis-tQTLs. (a) Distributions of overall eight-degree-of-freedom
ctAR likelihood-ratio statistics for two representative observed temporal
cis-tQTLs with approximately 80% and 90% bootstrap detection fractions;
the dashed line is the actual PPMI genome-wide discovery threshold.
(b,c) Distribution of detection fractions across assessed PPMI discovery
and PDBP matched loci, respectively; dashed lines show 80%.
(d) Fraction of the explicitly evaluated sensitivity-locus subset achieving
≥80% bootstrap re-detection after reducing the number of independent donors.
(e) Full-cohort observed independent PDBP nominal replication fraction and
PDBP real-donor subset re-detection, which are distinct conditional quantities.
(f) Summary of fractions achieving ≥80% conditional bootstrap detection in
PPMI, PDBP and both cohorts. The main tests consider the overall temporal
genetic trajectory only; no separate genotype-by-time interaction test is shown.
