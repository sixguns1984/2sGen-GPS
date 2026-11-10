#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gengps.amp_pd.config import AmpPdPaths, CELL_COMPOSITION_COVARIATES
from gengps.amp_pd.deconvolution import (
    read_cibersortx_lm22,
    aggregate_lm22,
    clr_transform_broad,
    standardize_within_cohort,
)

PP_COV_DEFAULT = "/public2/data/luojunfeng/2sTGCS_binary/exposure/ppmi/teqtl.cov.csv"
PD_COV_DEFAULT = "/public2/data/luojunfeng/2sTGCS_binary/exposure/pdbp/teqtl.cov.csv"


def _ids(path, cohort, expected):
    x = pd.read_csv(path, index_col=0)
    ids = x["sample_id"].astype(str).tolist()
    if len(ids) != expected or len(set(ids)) != expected:
        raise RuntimeError(f"{cohort}: expected exactly {expected} unique Stage-I sample IDs, got rows={len(ids)} unique={len(set(ids))}")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt")
    ap.add_argument("--ppmi", required=True)
    ap.add_argument("--pdbp", required=True)
    ap.add_argument("--pp-cov", default=PP_COV_DEFAULT)
    ap.add_argument("--pd-cov", default=PD_COV_DEFAULT)
    ap.add_argument("--pseudocount", type=float, default=1e-5)
    a = ap.parse_args()

    paths = AmpPdPaths(root=a.root)
    expected = {"PP": _ids(a.pp_cov, "PP", 1465), "PD": _ids(a.pd_cov, "PD", 1730)}
    results = {"PP": a.ppmi, "PD": a.pdbp}

    fine_parts, broad_parts, diag_parts = [], [], []
    qc = []
    meta_rows = []
    for cohort in ["PP", "PD"]:
        frac, diag = read_cibersortx_lm22(results[cohort])
        got = set(frac.index.astype(str))
        exp = expected[cohort]
        missing = [s for s in exp if s not in got]
        extra = sorted(got - set(exp))
        if missing:
            raise RuntimeError(
                f"{cohort}: Stage-I CIBERSORTx result is missing {len(missing)} exact manuscript samples; first={missing[:10]}. "
                "Do not drop or impute them; rerun the Stage-I exact mixture."
            )
        frac = frac.loc[exp].copy()
        diag = diag.loc[exp].copy()
        broad = aggregate_lm22(frac)
        frac["cohort"] = cohort
        broad["cohort"] = cohort
        diag["cohort"] = cohort
        fine_parts.append(frac)
        broad_parts.append(broad)
        diag_parts.append(diag)
        meta_rows.extend({"sample_id": s, "cohort": cohort} for s in exp)
        qc.append({
            "cohort": cohort,
            "expected_samples": len(exp),
            "result_samples": len(got),
            "matched_exact_samples": len(exp),
            "missing_exact_samples": 0,
            "extra_result_samples": len(extra),
        })

    fine = pd.concat(fine_parts, axis=0)
    broad = pd.concat(broad_parts, axis=0)
    diag = pd.concat(diag_parts, axis=0)
    raw_broad = broad.drop(columns=["cohort"])
    clr = clr_transform_broad(raw_broad, pseudocount=a.pseudocount)
    meta = pd.DataFrame(meta_rows)
    transformed = standardize_within_cohort(clr, meta)

    outdir = paths.data_dir / "stage1_manuscript" / "cell_deconvolution" / "cibersortx_lm22"
    outdir.mkdir(parents=True, exist_ok=True)
    fine.to_parquet(outdir / "LM22_fine_fractions.parquet", compression="zstd")
    broad.to_parquet(outdir / "LM22_broad_fractions.parquet", compression="zstd")
    diag.to_parquet(outdir / "LM22_diagnostics.parquet", compression="zstd")
    transformed.to_parquet(outdir / "LM22_CLR_covariates.parquet", compression="zstd")
    cov = transformed.reset_index()[["sample_id"] + list(CELL_COMPOSITION_COVARIATES)]
    if cov[CELL_COMPOSITION_COVARIATES].isna().any().any():
        raise RuntimeError("Non-finite Stage-I cell-composition covariates after exact-cohort standardization")
    cov.to_parquet(outdir / "stage1_cell_covariates.parquet", index=False, compression="zstd")
    qcdf = pd.DataFrame(qc)
    qcdf.to_csv(outdir / "LM22_stage1_import_QC.csv", index=False)

    report = {
        "PP": qc[0], "PD": qc[1],
        "pseudocount": float(a.pseudocount),
        "primary_covariates": list(CELL_COMPOSITION_COVARIATES),
        "cell_covariate_file": str(outdir / "stage1_cell_covariates.parquet"),
        "standardization": "CLR over seven broad compartments, omit Other coordinate, z-score within exact manuscript cohort",
    }
    (outdir / "stage1_cell_covariates_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
