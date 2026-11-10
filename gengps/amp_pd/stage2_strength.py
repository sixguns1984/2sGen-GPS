from __future__ import annotations

"""Minimal backend for Stage-II single-IV first-stage strength diagnostics.

This module intentionally contains only the functionality required by the current
single-temporal-IV Stage-II pipeline.  It replaces the historical dependency on the
retired multi-tQTL GReX training script.  The statistical quantities and output
semantics are unchanged: each LD-independent temporal IV is evaluated separately
using its frozen Stage-I MPTGA genetic component in PPMI and PDBP.
"""

from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from .config import AmpPdPaths
from .genotype import make_genotype_source
from .real_qtl import bfile_for_cohort, load_cohort_metadata, load_expression_columns
from .grex import (
    add_current_age,
    donor_balanced_weights,
    evaluate_frozen_grex,
    legacy_single_iv_from_qtl_row,
    prediction_metrics,
)

_WORKER = None


def _pc_cols(meta: pd.DataFrame, prefix: str) -> list[str]:
    out = []
    for c in meta.columns:
        s = str(c)
        if s.startswith(prefix):
            try:
                out.append((int(s[len(prefix):]), s))
            except Exception:
                pass
    return [x[1] for x in sorted(out)]


def _prediction_covariates(meta: pd.DataFrame, cohort: str) -> list[str]:
    cols = [c for c in ["sex_binary", "diagnosis_binary", "RIN"] if c in meta.columns]
    prefix = "gPC_PP_" if cohort == "PP" else "gPC_REP_"
    cols += _pc_cols(meta, prefix)
    return list(dict.fromkeys(cols))


def _prepare_meta(meta: pd.DataFrame, cohort: str) -> pd.DataFrame:
    z = meta.copy()
    if "followup_time" not in z.columns:
        raise KeyError("metadata lacks followup_time")
    return add_current_age(z)


def _clean_cached_meta(meta: pd.DataFrame, covars: list[str]) -> pd.DataFrame:
    ok = meta.participant_id.notna().to_numpy()
    ok &= np.isfinite(pd.to_numeric(meta.current_age, errors="coerce").to_numpy(float))
    ok &= np.isfinite(pd.to_numeric(meta.followup_time, errors="coerce").to_numpy(float))
    for c in covars:
        ok &= np.isfinite(pd.to_numeric(meta[c], errors="coerce").to_numpy(float))
    return meta.loc[ok].copy()


def _participant_mean_metrics(pred_df: pd.DataFrame) -> dict:
    if pred_df.empty:
        return {}
    d = pred_df.groupby("participant_id", as_index=False).agg(
        observed_residual=("observed_residual", "mean"),
        grex_residualized=("grex_residualized", "mean"),
    )
    if len(d) < 3:
        return {}
    return prediction_metrics(d.observed_residual, d.grex_residualized)


def _within_participant_metrics(pred_df: pd.DataFrame, prediction_col: str) -> dict:
    if pred_df.empty or prediction_col not in pred_df.columns:
        return {}
    d = pred_df[["participant_id", "observed_residual", prediction_col]].copy()
    d["participant_id"] = d.participant_id.astype(str)
    n = d.groupby("participant_id")["participant_id"].transform("size")
    d = d[n >= 2].copy()
    if d.participant_id.nunique() < 3 or len(d) < 6:
        return {}
    d["y_within"] = d.observed_residual - d.groupby("participant_id").observed_residual.transform("mean")
    d["p_within"] = d[prediction_col] - d.groupby("participant_id")[prediction_col].transform("mean")
    w = donor_balanced_weights(d.participant_id.to_numpy(str))
    return prediction_metrics(d.y_within.to_numpy(float), d.p_within.to_numpy(float), w)


def _metric_common(gene: str, n_samples: int, n_participants: int, cohort: str, level: str) -> dict:
    return {
        "gene": gene,
        "model": "legacy_single_iv",
        "n_snps": 1,
        "n_discovery_dynamic_snps": 1,
        "n_interaction_snps": 1,
        "n_samples": int(n_samples),
        "n_participants": int(n_participants),
        "cohort": cohort,
        "level": level,
    }


def _legacy_eval_rows(model, gene: str, meta: pd.DataFrame, y, covars: list[str], geno: pd.DataFrame, cohort: str, out_pred_file: Path) -> list[dict]:
    # Descriptive FULL genetic-block partial R2 (G*t0 ... G*t3), NOT a weak-IV cutoff.
    # The nuisance model uses the same expression-support age/covariate inputs;
    # this is a weighted auxiliary regression, NOT Stage1 ctAR's ARX likelihood.
    from .grex import donor_balanced_weights
    dose = pd.to_numeric(geno.loc[:, model.snp].reindex(meta.participant_id.astype(str)), errors="coerce").to_numpy(float)
    tm = pd.to_numeric(meta.followup_time, errors="coerce").to_numpy(float)
    yy = np.asarray(y, float)
    good = np.isfinite(dose) & np.isfinite(tm) & np.isfinite(yy)
    for cc in covars:good &= np.isfinite(pd.to_numeric(meta[cc],errors="coerce").to_numpy(float))
    age = pd.to_numeric(meta.current_age, errors="coerce").to_numpy(float)
    good &= np.isfinite(age)
    block_r2 = np.nan
    block_rank = 0
    if good.sum() > len(covars)+12 and np.unique(dose[good]).size > 1:
        ix = np.where(good)[0];t=tm[ix];t=(t-t.mean())/max(t.std(),1e-12)
        az=age[ix];az=(az-az.mean())/max(az.std(),1e-12)
        base=[np.ones(len(ix)),az,az*az,az*az*az]
        for cc in covars:base.append(pd.to_numeric(meta.iloc[ix][cc],errors="coerce").to_numpy(float))
        C=np.column_stack(base);G=np.column_stack([dose[ix]*t**j for j in range(4)])
        w=donor_balanced_weights(meta.iloc[ix].participant_id.astype(str).to_numpy());sw=np.sqrt(w)
        A=C*sw[:,None];B=np.column_stack([C,G])*sw[:,None];v=yy[ix]*sw
        r0=np.linalg.norm(v-A@np.linalg.lstsq(A,v,rcond=None)[0])**2
        r1=np.linalg.norm(v-B@np.linalg.lstsq(B,v,rcond=None)[0])**2
        block_rank=int(np.linalg.matrix_rank(B)-np.linalg.matrix_rank(A))
        if block_rank==4 and r0>1e-12:block_r2=float(np.clip((r0-r1)/r0,0,1))
    grex = model.predict_grex(geno, meta)
    em, yres, gres, _ = evaluate_frozen_grex(meta, y, grex, covars)
    support = model.support_mask(meta)
    em.update(_metric_common(gene, len(meta), meta.participant_id.nunique(), cohort, "sample"))
    em["full_genetic_block_partial_R2_descriptive"]=block_r2
    em["full_genetic_block_rank"]=block_rank
    em.update({
        "legacy_lead_snp": model.snp,
        "legacy_source_pvalue": model.source_pvalue,
        "legacy_time_support_fraction": float(np.mean(support)) if len(support) else np.nan,
        "n_candidate_features": int(len(model.beta_g)),
        "median_nonzero_features": int(np.sum(np.abs(np.asarray(model.beta_g, float)) > 1e-12)),
        "median_nonzero_main_features": np.nan,
        "median_nonzero_interaction_features": np.nan,
    })
    rows = [em]
    pdat = pd.DataFrame({
        "sample_id": meta.sample_id.astype(str).to_numpy(),
        "participant_id": meta.participant_id.astype(str).to_numpy(),
        "gene": gene,
        "model": "legacy_single_iv",
        "followup_time": meta.followup_time.to_numpy(float),
        "current_age": meta.current_age.to_numpy(float),
        "observed_expression": y,
        "observed_residual": yres,
        "grex_frozen": grex,
        "grex_residualized": gres,
        "within_stage1_time_support": support,
    })
    out_pred_file.parent.mkdir(parents=True, exist_ok=True)
    pdat.to_parquet(out_pred_file, index=False, compression="zstd")
    if cohort == "PDBP":
        wm = _within_participant_metrics(pdat, "grex_frozen")
        if wm:
            wm.update(_metric_common(gene, wm.get("n", 0), meta.participant_id.nunique(), cohort, "within_participant"))
            wm.update({
                "legacy_lead_snp": model.snp,
                "legacy_source_pvalue": model.source_pvalue,
                "legacy_time_support_fraction": float(np.mean(support)) if len(support) else np.nan,
                "grex_score_beta": np.nan,
                "grex_score_p": np.nan,
                "grex_score_F": np.nan,
            })
            rows.append(wm)
    dm = _participant_mean_metrics(pdat)
    if dm:
        dm.update(_metric_common(gene, len(meta), meta.participant_id.nunique(), cohort, "participant_mean"))
        dm.update({
            "legacy_lead_snp": model.snp,
            "legacy_source_pvalue": model.source_pvalue,
            "legacy_time_support_fraction": float(np.mean(support)) if len(support) else np.nan,
            "grex_score_beta": np.nan,
            "grex_score_p": np.nan,
            "grex_score_F": np.nan,
        })
        rows.append(dm)
    return rows


def init_worker(cfg: dict) -> None:
    global _WORKER
    paths = AmpPdPaths(root=cfg["root"])
    z = dict(cfg)
    z["paths"] = paths
    z["gsrc_pp"] = make_genotype_source(str(bfile_for_cohort(paths, "PP")), paths.plink19, prefer_bed_reader=True)
    z["gsrc_rep"] = make_genotype_source(str(bfile_for_cohort(paths, "PD")), paths.plink19, prefer_bed_reader=True)
    cache = {}
    for cohort in ["PP", "PD"]:
        m = _prepare_meta(load_cohort_metadata(paths, cohort), cohort)
        cov = _prediction_covariates(m, cohort)
        cache[cohort] = (_clean_cached_meta(m, cov), cov)
    z["meta_cache"] = cache
    _WORKER = z


def _prepare_batch_context(tasks: list[dict]) -> dict:
    c = _WORKER
    paths = c["paths"]
    expr_genes = list(dict.fromkeys(str(t["expr_gene"]) for t in tasks))
    union_snps = list(dict.fromkeys(str(t["snps"][0]) for t in tasks))
    expr_all = load_expression_columns(paths, expr_genes)
    meta_pp, _ = c["meta_cache"]["PP"]
    meta_pd, _ = c["meta_cache"]["PD"]
    pids_pp = pd.Index(pd.unique(meta_pp.participant_id.astype(str)), dtype=str)
    pids_pd = pd.Index(pd.unique(meta_pd.participant_id.astype(str)), dtype=str)
    geno_pp = c["gsrc_pp"].read(pids_pp, union_snps, mean_impute=True, dtype=np.float64)
    geno_pd = c["gsrc_rep"].read(pids_pd, union_snps, mean_impute=True, dtype=np.float64)
    out = {}
    for cohort, geno_all in [("PP", geno_pp), ("PD", geno_pd)]:
        meta, covars = c["meta_cache"][cohort]
        expr = expr_all.reindex(meta.sample_id.astype(str)).copy()
        expr.index = meta.sample_id.astype(str).values
        pids = pd.Index(pd.unique(meta.participant_id.astype(str)), dtype=str)
        out[cohort] = (meta, expr, covars, geno_all.reindex(pids))
    return out


def _gene_inputs(task: dict, cohort: str, ctx: dict):
    expr_gene = str(task["expr_gene"])
    snp = str(task["snps"][0])
    meta0, expr0, covars, geno0 = ctx[cohort]
    yraw = pd.to_numeric(expr0[expr_gene], errors="coerce")
    ok = np.isfinite(yraw.to_numpy(float))
    meta = meta0.loc[ok].copy()
    y = yraw.loc[ok].to_numpy(float)
    if len(meta) < 10:
        raise ValueError(f"{task['gene']}: insufficient usable {cohort} observations ({len(meta)})")
    return meta, y, covars, geno0.loc[:, [snp]]


def _error_result(task: dict, exc: Exception) -> dict:
    return {
        "gene": task.get("gene", "unknown"),
        "error": repr(exc),
        "traceback": traceback.format_exc(),
        "metrics_pp": [],
        "metrics_ext": [],
        "legacy_train": [],
    }


def process_pair_batch(tasks: list[dict]) -> list[dict]:
    """Evaluate a batch of independent temporal IV-gene pairs with shared I/O."""
    from threadpoolctl import threadpool_limits
    out = []
    try:
        with threadpool_limits(limits=int(_WORKER.get("blas_threads", 1))):
            ctx = _prepare_batch_context(tasks)
            for task in tasks:
                try:
                    t0 = time.perf_counter()
                    pseudo_gene = str(task["gene"])
                    legacy_row = task["legacy_row"]
                    model = legacy_single_iv_from_qtl_row(legacy_row, qtl_method=_WORKER["qtl_label"])
                    od = Path(_WORKER["out_dir"])
                    md = od / "models" / "legacy_single_iv"
                    md.mkdir(parents=True, exist_ok=True)
                    model.to_json(md / f"{pseudo_gene}.json")
                    meta_pp, y_pp, cov_pp, geno_pp = _gene_inputs(task, "PP", ctx)
                    meta_pd, y_pd, cov_pd, geno_pd = _gene_inputs(task, "PD", ctx)
                    pp_rows = _legacy_eval_rows(
                        model, pseudo_gene, meta_pp, y_pp, cov_pp, geno_pp, "PPMI",
                        od / "legacy_training_predictions" / f"{pseudo_gene}__legacy_single_iv.parquet",
                    )
                    for z in pp_rows:
                        z["level"] = f"{z['level']}_training_apparent"
                        z["qtl_label"] = _WORKER["qtl_label"]
                    pd_rows = _legacy_eval_rows(
                        model, pseudo_gene, meta_pd, y_pd, cov_pd, geno_pd, "PDBP",
                        od / "pdbp_predictions" / f"{pseudo_gene}__legacy_single_iv.parquet",
                    )
                    for z in pd_rows:
                        z["qtl_label"] = _WORKER["qtl_label"]
                    out.append({
                        "gene": pseudo_gene,
                        "metrics_pp": [],
                        "metrics_ext": pd_rows,
                        "legacy_train": pp_rows,
                        "gene_runtime_seconds": float(time.perf_counter() - t0),
                    })
                except Exception as e:
                    out.append(_error_result(task, e))
    except Exception as e:
        tb = traceback.format_exc()
        for task in tasks:
            out.append({
                "gene": task.get("gene", "unknown"),
                "error": repr(e),
                "traceback": tb,
                "metrics_pp": [],
                "metrics_ext": [],
                "legacy_train": [],
            })
    return out
