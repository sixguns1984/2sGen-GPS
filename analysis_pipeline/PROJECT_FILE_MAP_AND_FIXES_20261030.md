# 2sGenGPS source audit, Stage-I/II interfaces and SFig6/Figure2 repair

**Scope:** audited the user-supplied 2026-10-30 ZIP (665 archive members, including AppleDouble, cache files, legacy patches). This deliverable retains Stage-I/Stage-II statistical code and their output directories; only Figure2/Power source and diagnostics were edited. Do not interpret historical unused files as production entrypoints.

## Reproducible directory map

```
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
PLINK=/usr/bin/plink1.9

Stage I CTAR: $CODE/gengps/amp_pd/real_qtl.py
              $CODE/gengps/methods/ctar_mptga.py
              $CODE/scripts_amp_pd/03g_run_revised_followup_discovery.py
              $CODE/scripts_amp_pd/04g_run_revised_followup_replication.py

Data: $ROOT/data/metadata/amp_pd_samples.parquet
      $ROOT/data/expression/amp_pd_expression_z.parquet  # columns are HGNC symbols
      $ROOT/data/annotation/genes_used.parquet          # Ensembl ID -> symbols
      $ROOT/data/annotation/gene_locations.parquet      # optional fallback
      $ROOT/data/genotype/PP/amp_pd_PP_qc.{bed,bim,fam,pca.eigenvec}
      $ROOT/data/genotype/REP/amp_pd_REP_qc.{bed,bim,fam,pca.eigenvec}
      $ROOT/data/cell_deconvolution/cibersortx_lm22/LM22_CLR_covariates.parquet

Stage I results: $ROOT/qtl/PP/submitted_followup_ctar_mptga/final/
                 $ROOT/qtl/PD/submitted_followup_ctar_mptga/final/
                 significant_tqtl.parquet, all_results.parquet, fdr_finalize_summary.json

Stage II main production steps: $CODE/scripts_stage2/00_trajectory_direction.py ...
                                $CODE/scripts_stage2/10_assemble_final_master.py
Stage II shared IVs: $ROOT/single_iv_causal/iv_validation/
Stage II per phenotype: $ROOT/single_iv_causal/$ANALYSIS_NAME/
                        single_iv_genetic_component/*.parquet
                        temporal_causal_agecubic_full_followup_fast/
                        standard_final/
Figure2 source: $CODE/scripts_figures/pd_figure2/
Power source:   $CODE/scripts_figures/tqtl_power/2sGenGPS_CTAR_SFig6_refit_v3/
```

## Core Stage-I contract (do not guess from output columns)

- QTL rows: `Variant_ID`, `Ensembl_ID`, `Pvalue`, `LRT`, `rho`, `n_covariates_used`, temporal FDR.
- Expression rows: `sample_id` and **gene-symbol columns**. Resolve via `real_qtl.resolve_expression_gene_columns`.
- Metadata: `cohort=PP/PD`, `participant_id`, `followup_time`, `sample_id`; merge genotype PCs from `.pca.eigenvec` and LM22 CLR from canonical Parquet. `method_covariates` chooses the FULL expression PC sets in each cohort and discards non-informative columns.
- Duplicate participant-time visits: retain highest RIN, sorted deterministically.
- Genotype: Stage-I `genotype_channels(binary)` encodes 0:00, 1:01, 2:10, missing -1:11. Stage-I BED genotype source is preferred over independently exported allele dosage.
- Overall temporal QTL test: 8 df, raw cubic follow-up time, continuous-time OU residual, full and null profile rho separately using 100 grid points. **No dynamic-QTL analysis in SFig6.**

## Fixed Power bugs

1. Replaced heuristic Metadata and Expression loaders with `production_adapter.py` using real Stage-I loader functions. This avoids Ensembl-symbol mismatches and missing PCA/LM22 covariates.
2. Reads selected SNPs from the exact Stage-I BED/PLINK 1.9 source and retains production allele coding and missing genotype -1.
3. Uses Stage-I two-channel binary genetic terms, not dummy 1/2 terms when genotype=-1.
4. PDBP has its OWN covariate count; a PPMI `n_covariates_used` value must never be checked against PDBP.
5. Fixed-rho donor bootstrap remains the high-throughput **conditional redetection approximation**. For first `AUDIT_PAIRS` loci/cohort, repeat the exact 100-rho profile LRT and compare with stored Stage-I LRT. If mismatch, do not publish power fractions. The 100-rho model cross-check on synthetic data reproduced original scanner LRT to <1e-6.
6. New `audit-inputs` command checks valid per-locus matched sample, genotype, expression and covariates before full bootstrap. Empty/zero result aborts with reason summary instead of misleading `[FINISH]` and EmptyDataError.
7. Preserves original Supplementary Fig 6 goal: 5000 observed temporal tQTLs, 80% conditional donor-bootstrap redetection fraction for PP and matched PD (distinct denominators). *Not* prospective minimum sample size.

## Fixed Figure 2 bugs

1. Full main figure now uses a 2-by-2 grid. Panel a no longer spans the height of three cases; first row a+b, second row c+d.
2. Select examples by an **exact** `phenotype + gene + lead_snp` tuple, validated against Stage-II frozen `single_iv_genetic_component` and real outcome observations. No gene-only selector that might render a different SNP.
3. Don't require one example from each of UPDRS3, UPDRS2 and MoCA if the real high-MAF A-tier dataset doesn't contain such a triplet. Prefer topic/MAF/evidence, then phenotype diversity, and display the selected evidence tier. MAF cutoff searches .15, .10, .05 sequentially and records chosen MAF.
4. Report SNP-specific PPMI PLINK1.9 `.frq` MAF and A1/A2. Do not falsely claim A1 is the outcome-effect allele or the allele counted by Stage-II dosage without an explicit genotype-orientation audit.
5. Outcome and tGReX are joined on cohort+participant+time and shown only within observed outcome follow-up. Bin centers outside observed support are not plotted.

## Retained historical code

There are many previous revisions under `scripts_amp_pd/`, `scripts_figures/tqtl_power/`, `patch_backups/`, and other experimental entrypoints. These were **not deleted**, because the uploaded ZIP alone does not establish which older audit or sensitivity products the paper still cites. Production Stage-I/Stage-II entrypoints were left unchanged. Prefer the commands in the installation guide and not the earlier experimental Power launchers.

## Validation completed / limitations

- Unit tests for revised Figure2 frequency ingestion, Stage-I binary mapping, numerical CTAR 100-rho LRT equivalence and older Bootstrap module passed.
- Synthetic outcome+frozen-component Figure2 end-to-end rendering completed (PDF, SVG, PNG). These are not real PD data.
- This environment lacks `pyarrow` and has no `/public2/data/...` user data, so **real PPMI/PDBP Parquet/BED end-to-end fitting remains unverified**. The new read-only `audit-inputs` is the required next gate on the server.
- The fixed-rho Bootstrap is a conditional observed-data resampling/re-detection estimate, not an unbiased prospective power calculation. Don't label as an unconditional population-level power estimate.
