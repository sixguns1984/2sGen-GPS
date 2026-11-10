# Manuscript analysis pipeline (stepwise, source-derived)

This folder holds the relevant analysis scripts copied from the user's source archive dated November 2026. File names and CLI conventions are preserved. **It is not a new one-command pipeline**: workflows depend on controlled AMP-PD, ADNI and ROSMAP data, PLINK, cohort-specific covariates, and GPU-capable PyTorch where applicable. The scripts here have not been reexecuted against the restricted real datasets in this release environment.

## Server project roots

```bash
export ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
export CODE=$(cd .. && pwd)
export PLINK=/usr/bin/plink1.9
```

The manuscript analysis scripts use `sys.path` imports from a sibling `gengps/` package. Run from the **repository root** after `pip install -e .`, rather than inside this folder, or set `PYTHONPATH=$(pwd)`.

## 1. Stage-I source-data preparation

- `scripts_amp_pd/00a_export_stage1_cibersortx_lm22_inputs.py`
- `scripts_amp_pd/00b_run_stage1_cibersortx_lm22.sh`
- `scripts_amp_pd/00c_import_stage1_cibersortx_lm22.py`
- `scripts_amp_pd/01e_run_expression_pc_BE.py`
- `scripts_amp_pd/00_prepare_stage1_manuscript_inputs.py` (historical uniform-visit benchmark only; **not** the current full CTAR observational universe)
- `scripts_amp_pd/00d_audit_stage1_ctar_inputs.py`

Revised CTAR source-data preparation is configured through the source `AmpPdPaths` class. Review cohort input caches, SNP identity, RNA sample ordering, CIBERSORTx and expression PCs before running. Controlled data are **not included**.

## 2. Stage-I CTAR discovery and PDBP replication

From repository root:

```bash
export ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
export CODE=$(pwd)
# The source launchers use GPU IDs 0,1,2; edit for your machine.
bash analysis_pipeline/run_stage1_ctar_genomewide.sh
bash analysis_pipeline/run_stage1_ctar_replication.sh
```

The original drivers expect the working directory to be the source code repository root and call `scripts_amp_pd/...`. In this release they live under `analysis_pipeline/`. **Use the release wrapper** `analysis_pipeline/run_here.sh` to invoke commands in the correct location, e.g.:

```bash
bash analysis_pipeline/run_here.sh run_stage1_ctar_genomewide.sh
bash analysis_pipeline/run_here.sh run_stage1_ctar_replication.sh
```

The main pipeline uses the exact CTAR engine `gengps/methods/ctar_mptga.py` and out-of-core FDR code in `gengps/amp_pd/fdr.py`. Genome-wide q-values must be computed across **all** tested cis pairs, not separately within gene/chunk. Stage-I final results are expected under:

```text
$ROOT/qtl/PP/submitted_followup_ctar_mptga/final/significant_temporal.parquet
$ROOT/qtl/PP/submitted_followup_ctar_mptga/final/significant_tqtl.parquet
$ROOT/qtl/PD/submitted_followup_ctar_mptga/final/all_results.parquet
```

## 3. Independent SNP–gene IV preparation

```bash
bash analysis_pipeline/run_here.sh run_stage2_prepare_temporal_ivs.sh
```

Key steps:

- `scripts_stage2/00_trajectory_direction.py`: match PPMI/PDBP genotype-effect trajectories on common support
- `scripts_stage2/01_prepare_temporal_iv_candidates.py`: discovery, nominal PDBP replication, trajectory concordance and **within-gene LD pruning**
- `scripts_stage2/02_compute_iv_strength.py`: descriptive IV strength audit (first-stage F is **not** a causal-tier gate)

Final candidates expected at:

```text
$ROOT/single_iv_causal/iv_validation/single_iv_primary_eligible.csv
```

**LD warning:** The supplied launcher's default `LD_R2` is `0.1`; historical conversation references also mention `0.01`. Verify the actual final command/log before claiming either cutoff generated the published 22,842 pairs.

## 4. Participant-level donor bootstrap detectability

The source figure/power scripts are in `scripts_figures/tqtl_power/`. Use them for manuscript-calibrated redetection across the fixed retained candidate list, PPMI and PDBP, with the intended 80% threshold. Do not replace the production calculation with the package's quick example bootstrap.

## 5. Stage-II longitudinal PD (UPDRS I, II, III)

Primary stages (inspect source scripts' `--help` and existing ROOT input paths):

```bash
python analysis_pipeline/scripts_stage2/04_prepare_phenotype_pcs.py --help
python analysis_pipeline/scripts_stage2/06_run_temporal_causal_fast.py --help
python analysis_pipeline/scripts_stage2/07_finalize_replicated_candidates.py --help
python analysis_pipeline/scripts_stage2/08_candidate_robustness.py --help
python analysis_pipeline/scripts_stage2/09_cross_gene_ld_audit.py --help
python analysis_pipeline/scripts_stage2/10_assemble_final_master.py --help
```

Source production execution (run after candidate predictions and phenotype covariate files exist):

```bash
python analysis_pipeline/scripts_stage2/06_run_temporal_causal_fast.py \
  --root "$ROOT" --replication-cohorts PD,SU \
  --replication-pooled-groups PD+SU --max-lag 4 \
  --min-participants 50 --use-genotype-pcs \
  --adjust-current-age-cubic --no-require-stage1-followup-support \
  --ar-mode off --detail-mode minimal
```

**Careful:** This command alone does not materialize tGReX predictions. It assumes prior Stage-I IV identification, genotype-PC preparation and IV prediction outputs, and phenotype-specific input files. Use sequential preparation drivers preserved in `scripts_stage2/`.

The source inference is OLS estimation plus **participant-clustered Wald/F** for the tested lagged tGReX block. It selects among replicated lag models using **SC** (historically written as BIC in internal scripts). Cross-cohort discovery FDR and nominal replication are **not** identical to requiring BH-FDR in all cohorts. The current PD/AD evidence labels are descriptive and do not use first-stage F.

**Important audit:** SC computed over different lag-specific sample sets may require comparability analysis; this refactor preserves historical manuscript code and does **not** silently change lag-selection results.

## 6. AD-related cognition (MAP discovery; ADNI replication)

- `scripts_stage2/16_prepare_ad_mmse.py`
- `scripts_stage2/17_run_ad_mmse_temporal_fast.py`
- `scripts_stage2/18_materialize_ad_mmse_all_followup.py`
- `scripts_stage2/19_prepare_ad_mmse_iv_no_reqc.py`
- `scripts_stage2/20_finalize_ad_mmse.py`
- `scripts_stage2/21_regrade_ad_mmse_existing_final.py`
- `scripts_stage2/run_ad_mmse_stage2.sh` / `run_ad_mmse_finalize.sh` for the established stepwise paths

## 7. Independent genetic evidence (optional but required for published tiers)

- `scripts_stage2/08_candidate_robustness.py`: reduced-form, multi-IV fixed-effect analysis, heterogeneity and LOO
- `scripts_stage2/09_cross_gene_ld_audit.py`: cross-gene LD
- `scripts_stage2/10_assemble_final_master.py`: final PD annotations and A/B/C tiers
- AD-specific grade finalization: `scripts_stage2/20_finalize_ad_mmse.py`, `21_regrade_ad_mmse_existing_final.py`

The general `2sgengps evidence` CLI calls the same source annotation logic **only after full robustness results exist**.

## 8. MRI DBM and postmortem brain analyses

MRI (exploratory, lag 1):

```bash
bash analysis_pipeline/run_here.sh scripts_stage2/run_mri_dbm_lag1_repredict.sh
bash analysis_pipeline/run_here.sh scripts_stage2/run_mri_dbm_evidence_full.sh
```

Postmortem SNP-associated expression by donor pseudobulk:

```bash
bash analysis_pipeline/run_here.sh scripts_snp_scDE/run_edger.sh
```

Use `scripts_snp_scDE/edger_postmortem.py` and `edger_qlf_fit.R` for donor-level edgeR quasi-likelihood analyses. The package also includes earlier exploratory single-cell methods; the reported final results must use actual donor-pseudobulk results, **not** earlier HC3/DESeq2 or single-cell LMM tests.

## 9. Figures and Supplementary Tables

- `scripts_figures/pd_figure2/01_make_pd_figure2.py`
- `scripts_figures/ad_figure4/01_make_ad_figure4.py`
- `scripts_figures/mri_dbm/` and `scripts_figures/postmortem_fig3f/`
- `scripts_figures/tqtl_power/` and `scripts_figures/tqtl_benchmark/`
- `scripts_amp_pd/summarize_pd_three_phenotypes.py`

Scientific plots intended for Adobe Illustrator use Cairo before pyplot import. Some historical auxiliaries may use Agg; use the final primary figure scripts for editable vector export.

## 10. vQTL sensitivity on final 22,842 CTAR pairs

The optional updated script is `scripts_amp_pd/06d_stage1_final_tqtl_vqtl_sensitivity.py` with a corresponding launcher `run_stage1_final_22842_vqtl.sh`. It reads the final `single_iv_primary_eligible.csv` rather than old `significant_dynamic.parquet`.

## Explicit exclusions from revised manuscript

Original submission's dynamic/additive subclassification, MoCA progression, CSF-mediated C5orf63–LRP1 mechanism and experimental patch/backup folders were **not selected as manuscript-result workflows**. Old exploratory scripts still present in the source archive were excluded from this clean tree; compare with the original archive if independent historical investigation is needed.

## Reproducibility cautions

1. This is a **code-only public release**. No restricted genotype, RNA-seq, proteomic, MRI or clinical records are included.
2. Absolute original server roots remain in some default arguments; override `--root` and data paths on another server.
3. Genome-wide full AMP-PD run requires substantial HPC resources and GPU/PLINK/R dependencies.
