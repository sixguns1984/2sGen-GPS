# CTAR SFig6 v4 — real-input attribute fix + faster, reproducible bootstrap

This patch **overlays** the exact project folder `scripts_figures/tqtl_power/2sGenGPS_CTAR_SFig6_refit_v3/`. It does not modify Stage-I, Stage-II, or their results. Do NOT re-run Stage-I, PLINK extraction or `prepare` if they already succeeded.

## Confirmed direct failure

`02_run_power.py` was accessing `ProductionExpression.mode`, but the Stage-I-derived `ProductionExpression` class did not define it. Fixed by defining `mode='wide_symbol_mapped'` and adding a defensive provenance fallback. The user's input alignment audit had passed 24/24, hence this was the immediate failure, not missing expression or genotype inputs.

## Numerical and performance approach

- Same 8-df **overall** CTAR temporal-tQTL LRT, no dynamic-QTL analysis.
- Same donor/participant resampling with replacement for main bootstraps; visit rows remain grouped.
- Cache per-donor fixed-rho whitened GLS cross-products once per locus.
- Use batched BLAS3 contractions of donor multiplicities with cross-products and Cholesky nuisance/full GLS refitting, with pseudoinverse fallback for singular matrices.
- Check the first six bootstraps against the original pseudoinverse implementation using identical RNG draws. Auto-fallback for numerically unstable designs. This is **not** noncentral-chi-square power extrapolation.
- Reuse per-cohort Stage-I nuisance covariate arrays across loci.
- Run multiple separate worker processes; each writes a separate CSV and logs. Same `POWER_JOBS` must be used on resume. Original rho-profiled 100-grid LRT audit remains independent of the fixed-rho bootstrap.

In a local benchmark matching PPMI dimensions (1405 donors, 51 covariates, 3 irregular visits), 200 replicates took ~0.706 s (legacy) vs ~0.111 s (fast): 6.35-fold kernel speedup, max LRT difference 2.2e-9. **This is not an end-to-end 5000-locus runtime benchmark**, and no software can guarantee a globally fastest algorithm from this measurement.

## Server installation

Upload `2sGenGPS_CTAR_SFig6_v4_speedfix.zip` to `$ROOT`:

```bash
ROOT=/public2/data/luojunfeng/2sGenGPS_revised_chatgpt
CODE=$ROOT/2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized
POWER=$CODE/scripts_figures/tqtl_power/2sGenGPS_CTAR_SFig6_refit_v3
cp -a "$POWER" "${POWER}_backup_before_v4"
unzip -oq "$ROOT/2sGenGPS_CTAR_SFig6_v4_speedfix.zip" -d "$CODE"
cd "$CODE"
python -m pytest -q "$POWER/tests"
```

## Commands

```bash
# Existing prepare and cache-expression already succeeded; do not repeat.
# This removes only old _preflight outputs and reruns the 30-pair test.
PREFLIGHT_PAIRS=30 PREFLIGHT_REPS=100 PREFLIGHT_AUDIT_PAIRS=10 \
  bash "$POWER/run_sfigure6.sh" preflight

cat "$ROOT/manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_refit_v3/_preflight/fit_audit.json"

# Pass must be true. 4 processes is a conservative default; profile RAM/CPU
# before increasing POWER_JOBS. 1 BLAS thread per worker prevents oversubscription.
POWER_JOBS=4 BOOTSTRAPS=1000 POWER_BATCH_SIZE=32 \
  nohup bash "$POWER/run_sfigure6.sh" full \
  > "$ROOT/logs/sfig6_ctar_refit_v4_fast.log" 2>&1 &

tail -f "$ROOT/logs/sfig6_ctar_refit_v4_fast.log"

# After all workers finish successfully and merge:
bash "$POWER/run_sfigure6.sh" plot
```

Intermediate checkpoints:

```
$ROOT/manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_refit_v3/_shards/job_00/locus_bootstrap_results.csv
$ROOT/manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_refit_v3/_shards/job_00/worker.log
...
```

`full` now means safe parallel execution; `full-serial` retains the prior behavior. If interrupted, run the same `full` command with the SAME `POWER_JOBS`; completed locus/cohort pairs are skipped. To re-merge valid worker outputs without rerunning, use:

```bash
POWER_JOBS=4 bash "$POWER/run_sfigure6.sh" merge
```

## Important caveats

- This reports **conditional donor-bootstrap re-detection frequency among the observed PPMI-significant overall temporal tQTLs**, not unbiased prospective statistical power.
- A real Stage-I LRT audit is mandatory before `full`; failed audit does not mean zero biological signal.
- PP denominator is all sampled 5000 discoveries successfully modeled; PDBP denominator consists of matched evaluable pairs. Unmatched/failed pairs are reported separately, not silently renamed as zero-power.
- The fixed-rho bootstrap is a conditional approximation to Stage-I's separately profiled H0/H1 rho; exact full profile refit is retained as an optional slower mode.
- This patch does not claim a GPU implementation or globally optimal speed; if GPU/server CPU constraints differ, tune `POWER_JOBS` and `POWER_BATCH_SIZE` using the included `08_benchmark_bootstrap.py`.
