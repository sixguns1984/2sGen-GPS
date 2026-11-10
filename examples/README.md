# Synthetic quick-start dataset
All donor identifiers, genotypes, expression values and outcomes in this folder are **synthetic**.
These files exercise the public pip CLI with two cis pairs and one Stage-II gene/SNP.
They cannot be used to reproduce AMP-PD statistics or the 22,842 final cis-eQTL pairs.

Run from repository root after `pip install -e .`:

```bash
2sgengps ctar --covariates examples/stage1_covariates.csv --expression examples/stage1_expression.csv --genotypes examples/stage1_genotypes.csv --cis-map examples/stage1_cis_map.csv --covariates-cols sex --output results/demo_ctar.csv --rho-points 10
2sgengps stage2 --input examples/stage2_visits.csv --outcome UPDRS_III --age-cubic --min-participants 20 --output results/demo_stage2.csv --evidence
```
Optional donor-bootstrap quick test: add `--bootstrap-reps 5` to the CTAR command.
