# 2sGen-GPS

**Temporal cis-eQTL mapping and genetically informed longitudinal phenotype association.**

This repository contains (1) an installable Python package for CTAR-MPTGA Stage-I mapping and cumulative ARX Stage-II testing, and (2) the stepwise manuscript analysis pipeline under [`analysis_pipeline/`](analysis_pipeline/README.md).

## Why continuous time?

The previous MPTGA setup used a complete, equally scheduled five-visit design. **CTAR-MPTGA** models within-donor residual correlation as `Corr(e(t),e(s)) = exp(-phi * abs(t-s))` and supports unequal follow-up intervals and missing scheduled visits without time-grid imputation. The mean model includes a shared cubic follow-up trend, covariates, and genotype-specific cubic trajectories. The overall likelihood-ratio test uses **8 degrees of freedom** with the manuscript's two-channel binary hard-call genotype coding.

The revised manuscript analyzed **7,650 RNA-seq observations from 2,778 participants**, versus **3,195 observations from 639 participants** in the original manuscript; after discovery, replication, trajectory concordance and LD pruning, **22,842 SNP–gene associations (7,570 genes)** were retained. These are manuscript results, **not outputs of the small synthetic example**.

## Install

Requires **Python >=3.10**. PyTorch is used by the original CTAR engine. For GPU usage, install the CUDA-compatible PyTorch build appropriate for your server first.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
# Or from an unpacked release directory:
pip install .
2sgengps --help
```

When this repository is publicly pushed, install directly from GitHub:

```bash
pip install 'git+https://github.com/sixguns1984/2sGen-GPS.git'
```

**Important:** The package is not published to PyPI by generating this release. `pip install 2sGen-GPS` works only after a separate, authorized PyPI upload. Git installation or `pip install .` works without PyPI.

Large-scale AMP-PD pipelines require the extended dependencies and controlled cohort data:

```bash
pip install -e '.[pipeline]'
```

## Quick start: Stage I CTAR-MPTGA

Input files are small CSV tables in [`examples/`](examples/README.md):

- `stage1_covariates.csv`: `sample_id,participant_id,followup_time` and optional numeric covariates.
- `stage1_expression.csv`: `sample_id` plus expression gene columns (same sample identifiers).
- `stage1_genotypes.csv`: `participant_id` plus hard-call genotype columns (`0/1/2`, optionally `-1` for missing as in original core).
- `stage1_cis_map.csv`: `geneid,snpid` pairs.

```bash
2sgengps ctar \
  --covariates examples/stage1_covariates.csv \
  --expression examples/stage1_expression.csv \
  --genotypes examples/stage1_genotypes.csv \
  --cis-map examples/stage1_cis_map.csv \
  --covariates-cols sex \
  --rho-points 100 \
  --output results/stage1_ctar.csv
```

For all tests in the input cis-map, the CLI **recomputes BH-FDR globally over the evaluated overall CTAR LRT P values**. The `q_bh_global` and `significant_overall` columns are the intended primary tQTL screen; genotype-by-time interaction columns are diagnostics and not used for manuscript candidate selection. The small-data CLI keeps all pairs in memory; for genome-wide QTL scans and out-of-core global FDR use the `analysis_pipeline` Stage-I drivers.

Optional participant-level bootstrap redetection:

```bash
2sgengps ctar --covariates examples/stage1_covariates.csv \
  --expression examples/stage1_expression.csv \
  --genotypes examples/stage1_genotypes.csv \
  --cis-map examples/stage1_cis_map.csv --covariates-cols sex \
  --rho-points 10 --bootstrap-reps 20 \
  --output results/stage1_bootstrap_demo.csv
```

Bootstrap resamples participants and re-fits the CTAR model. **The pip CLI's bootstrap threshold is nominal `P < --alpha`**, suitable for a basic redetection demonstration. The manuscript's 80% power screening uses its dedicated donor-bootstrap calibration scripts, cohorts and thresholds from `analysis_pipeline`; do not substitute a demonstration fraction for the manuscript-level power statistic.

## Quick start: Stage II cumulative ARX

Input must contain `participant_id,time,<outcome>,grex_single_iv` and optionally `gene,lead_snp,iv_dosage,age,sex,AAO,duration` plus genotype PCs. The `grex_single_iv` column must be **computed externally from a fitted Stage-I expression model**; the CLI does not infer it from a raw expression matrix. The publication prediction/harmonization workflow is in `analysis_pipeline/`.

```bash
2sgengps stage2 \
  --input examples/stage2_visits.csv \
  --outcome UPDRS_III --exposure grex_single_iv \
  --max-lag 4 --min-participants 20 --age-cubic \
  --output results/stage2_arx.csv
```

Fits all lag orders using the original study's OLS coefficient estimation and participant-clustered Wald/F joint test. BH-FDR is applied to all valid lag tests in this provided file. Individual `coef_x_lag*`, `coef_sum`, `SC` (Schwarz criterion; mathematically BIC) and statistical support fields are written. Cross-cohort replication and lag selection must be performed separately with the manuscript pipeline to match the published design.

### Optional genetic evidence

For optional **reduced-form genetic robustness** within the Stage-II dataset, include `iv_dosage` and run:

```bash
2sgengps stage2 --input examples/stage2_visits.csv --outcome UPDRS_III \
  --age-cubic --evidence --output results/stage2_with_evidence.csv
```

The source implementation evaluates an additional genotype-derived reduced-form block. The `stage2_reduced_form_evidence.csv` file contains its results. For **full manuscript A/B/C evidence tiers**, first compute cross-cohort reduced-form, multi-IV Q/I²/LOO and cross-gene LD outputs using `analysis_pipeline/scripts_stage2/08_candidate_robustness.py`, `09_cross_gene_ld_audit.py` and the original finalization scripts. Then:

```bash
2sgengps evidence --input /path/to/completed_robustness_master.csv \
  --output results/graded_master.csv
```

This applies the exact post-selection `add_evidence_annotations()` logic from the source manuscript pipeline, including BH correction. It intentionally refuses to assign tiers when required robustness evidence columns are absent. **A/B/C are descriptive support categories, not formal causal certification or additional discovery selection gates.** First-stage F is not a tiering criterion.

## Source-level use

```python
from gengps.methods.ctar_mptga import ContinuousTimeMPTGA
from gengps.amp_pd.single_iv_causal import fit_all_lags
from gengps.amp_pd.evidence import add_evidence_annotations
```

## Reproduce the manuscript

See [`analysis_pipeline/README.md`](analysis_pipeline/README.md) for source data requirements, Stage-I discovery/replication, genome-wide FDR, trajectory harmonization, LD pruning, tGReX prediction, Stage-II PD/MMSE, evidence tiers, bootstrap, imaging, postmortem and figure generation.

This GitHub release contains **code and synthetic examples only**. Individual-level cohort data are subject to original data-use restrictions and are not redistributed.

## License

No open-source license is assigned in this draft. The repository owner should choose and add a license before releasing the code for unrestricted reuse.

## Version and reproducibility

Version `2.0.0` is the reorganized release candidate based on the supplied November 2026 source archive. Manuscript stage-specific scripts retain historical filenames to preserve their parameters and filesystem conventions. Original statistics have **not** been recomputed here with the restricted full AMP-PD datasets. For a frozen manuscript release, create a GitHub tag and archive DOI after checking the final server-side paths and datasets.

## Citation

Luo J et al. *Mapping temporal genetic regulation reveals longitudinal associations with neurodegenerative disease progression.* Manuscript under revision; publication identifiers to be added after acceptance.
