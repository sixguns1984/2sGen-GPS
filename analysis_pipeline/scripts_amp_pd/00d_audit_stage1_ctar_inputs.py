#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gengps.amp_pd.config import AmpPdPaths, CELL_COMPOSITION_COVARIATES
from gengps.amp_pd.real_qtl import _pc_columns, bfile_for_cohort, _augment_fixed_genotype_pcs, _merge_current_cell_covariates


def summarize_cohort(paths: AmpPdPaths, cohort: str):
    mp = paths.data_dir / "metadata" / "amp_pd_samples.parquet"
    ep = paths.data_dir / "expression" / "amp_pd_expression_z.parquet"
    cp = paths.data_dir / "cell_deconvolution" / "cibersortx_lm22" / "LM22_CLR_covariates.parquet"
    for x in [mp, ep, cp]:
        if not x.exists():
            raise FileNotFoundError(x)

    meta = pd.read_parquet(mp)
    # The all-sample CIBERSORTx parquet is the canonical source.  Do not
    # require amp_pd_samples.parquet to have been rewritten after import.
    meta = _merge_current_cell_covariates(paths, meta, required=True)
    meta = meta[meta["cohort"].astype(str).eq(cohort)].copy()
    meta["followup_time"] = pd.to_numeric(meta["followup_time"], errors="coerce")
    n_start = len(meta)
    meta = meta[meta["followup_time"].notna()].copy()

    # Check that raw-derived expression contains all candidate samples.
    expr_schema = pq.ParquetFile(ep).schema.names
    if "sample_id" not in expr_schema:
        raise RuntimeError(f"{ep} lacks sample_id")
    expr_ids = set(pd.read_parquet(ep, columns=["sample_id"])["sample_id"].astype(str))
    meta = meta[meta["sample_id"].astype(str).isin(expr_ids)].copy()

    # Revised covariate set.
    gpref = "gPC_PP_" if cohort == "PP" else "gPC_REP_"
    eig = Path(str(bfile_for_cohort(paths, cohort)) + ".pca.eigenvec")
    if not eig.exists():
        raise FileNotFoundError(eig)
    eig_npcs = pd.read_csv(eig, sep=r"\s+", header=None, nrows=1).shape[1] - 2
    if eig_npcs < 8:
        raise RuntimeError(f"{eig} has only {eig_npcs} PCs; revised Stage-I requires first 8")
    if cohort == "PP":
        eprefs = ["ePC_BE_PP_FULL_"]
    else:
        eprefs = ["ePC_BE_PD_FULL_"]
    gpcs = _pc_columns(meta, gpref)
    epcs = []
    epref_used = None
    for pref in eprefs:
        epcs = _pc_columns(meta, pref)
        if epcs:
            epref_used = pref
            break
    if not gpcs:
        raise RuntimeError(f"No genotype PCs found with prefix {gpref}")
    if not epcs:
        raise RuntimeError(f"No full-sample expression PCs found; tried {eprefs}")

    # Match the revised scanner exactly: the first eight PCs are read directly
    # from the current raw-derived PLINK PCA file, rather than relying on how many
    # Tracy-Widom-selected PCs happened to be merged into frozen metadata.
    meta = _augment_fixed_genotype_pcs(paths, cohort, meta, n_pcs=8)
    fixed_gpcs = [f"ctar_gPC{i}" for i in range(1, 9)]
    wanted = ["age_at_baseline", "sex_binary", "diagnosis_binary", "RIN"] + fixed_gpcs + epcs + list(CELL_COMPOSITION_COVARIATES)
    missing_cols = [c for c in wanted if c not in meta.columns]
    if missing_cols:
        raise RuntimeError(f"Missing CTAR covariate columns for {cohort}: {missing_cols}")

    num = meta[wanted].apply(pd.to_numeric, errors="coerce")
    complete_cov = np.isfinite(num.to_numpy(float)).all(axis=1)
    meta2 = meta.loc[complete_cov].copy()

    # Resolve exact duplicate participant-time observations identically to the scanner.
    meta2["__rin"] = pd.to_numeric(meta2["RIN"], errors="coerce").fillna(-np.inf)
    meta2["__sid"] = meta2["sample_id"].astype(str)
    meta2 = meta2.sort_values(["participant_id", "followup_time", "__rin", "__sid"], ascending=[True, True, False, True])
    dup = meta2.duplicated(["participant_id", "followup_time"], keep="first")
    n_dup = int(dup.sum())
    meta2 = meta2.loc[~dup].copy()

    vc = meta2.groupby("participant_id").size()
    old_grid = {"PP": {0.0, 0.5, 1.0, 2.0, 3.0}, "PD": {0.0, 0.5, 1.0, 1.5, 2.0}}[cohort]
    complete_old = 0
    for _, g in meta2.groupby("participant_id"):
        if old_grid.issubset(set(map(float, g["followup_time"]))):
            complete_old += 1

    out = {
        "cohort": cohort,
        "n_rows_cohort_metadata": int(n_start),
        "n_rows_numeric_followup_and_expression": int(len(meta)),
        "n_rows_complete_revised_covariates": int(complete_cov.sum()),
        "n_duplicate_participant_time_rows_removed": n_dup,
        "n_observations_ctar": int(len(meta2)),
        "n_participants_ctar": int(meta2["participant_id"].astype(str).nunique()),
        "n_participants_ge2_visits": int((vc >= 2).sum()),
        "n_participants_ge3_visits": int((vc >= 3).sum()),
        "n_participants_ge4_visits": int((vc >= 4).sum()),
        "n_participants_old_complete_grid": int(complete_old),
        "visits_min": int(vc.min()) if len(vc) else 0,
        "visits_median": float(vc.median()) if len(vc) else None,
        "visits_max": int(vc.max()) if len(vc) else 0,
        "followup_min": float(meta2["followup_time"].min()) if len(meta2) else None,
        "followup_max": float(meta2["followup_time"].max()) if len(meta2) else None,
        "n_genotype_pcs_in_frozen_metadata": len(gpcs),
        "genotype_pc_prefix_in_frozen_metadata": gpref,
        "n_genotype_pcs_used_revised_stage1": 8,
        "genotype_pc_source": str(eig),
        "n_expression_pcs": len(epcs),
        "expression_pc_prefix": epref_used,
        "cell_covariates": list(CELL_COMPOSITION_COVARIATES),
        "cell_covariates_complete": bool(meta2[CELL_COMPOSITION_COVARIATES].notna().all().all()) if len(meta2) else False,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt")
    a = ap.parse_args()
    paths = AmpPdPaths(root=a.root)
    report = {c: summarize_cohort(paths, c) for c in ["PP", "PD"]}
    outdir = paths.data_dir / "stage1_ctar"
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / "input_audit.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(dest)


if __name__ == "__main__":
    main()
