#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gengps.amp_pd.config import AmpPdPaths
from gengps.amp_pd.preprocess import load_gene_annotation, _norm_sample_id

PP_COV_DEFAULT = "/public2/data/luojunfeng/2sTGCS_binary/exposure/ppmi/teqtl.cov.csv"
PD_COV_DEFAULT = "/public2/data/luojunfeng/2sTGCS_binary/exposure/pdbp/teqtl.cov.csv"


def _read_signature(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    try:
        z = pd.read_csv(p, sep="\t")
        if z.shape[1] <= 2:
            z = pd.read_csv(p, sep=None, engine="python")
    except Exception:
        z = pd.read_csv(p, sep=None, engine="python")
    genes = z.iloc[:, 0].astype(str).tolist()
    # preserve LM22 order but drop repeated names if any
    return list(dict.fromkeys(genes))


def _read_exact_sample_ids(path: str, cohort: str, expected_n: int):
    x = pd.read_csv(path, index_col=0, usecols=lambda c: c in {"", "sample_id"} or c.startswith("Unnamed"))
    # pandas usecols + index_col can be brittle across unnamed index headers; fallback robustly
    if "sample_id" not in x.columns:
        x = pd.read_csv(path, index_col=0)
    ids = x["sample_id"].astype(str).tolist()
    if len(ids) != expected_n or len(set(ids)) != expected_n:
        raise RuntimeError(f"{cohort}: expected exactly {expected_n} unique manuscript sample IDs, got rows={len(ids)} unique={len(set(ids))}")
    return ids


def _resolve_source_columns(matrix_path: str, wanted_ids: set[str]):
    hdr = pd.read_csv(matrix_path, sep="\t", nrows=0)
    if len(hdr.columns) < 2:
        raise RuntimeError(f"Unexpected raw TPM header in {matrix_path}")
    first = hdr.columns[0]
    if str(first) != "Geneid":
        raise RuntimeError(f"Expected first raw TPM column Geneid, got {first!r}")
    mapping = {}
    for c in hdr.columns[1:]:
        n = _norm_sample_id(str(c))
        if n in wanted_ids:
            if n in mapping:
                raise RuntimeError(f"Duplicate normalized raw TPM sample ID {n!r}")
            mapping[n] = str(c)
    missing = sorted(wanted_ids - set(mapping))
    if missing:
        raise RuntimeError(
            f"Raw AMP-PD TPM matrix is missing {len(missing)} exact manuscript samples; first={missing[:10]}"
        )
    return mapping


def _annotation_maps(annotation_csv: str, sig_genes: list[str]):
    ann = load_gene_annotation(annotation_csv, min_width=0, autosome_only=False).copy()
    ann["gene_id"] = ann["gene_id"].astype(str)
    ann["gene_name"] = ann["gene_name"].astype(str)
    ann = ann[ann["gene_name"].isin(set(sig_genes))].copy()
    # match the current deconvolution convention: first annotation per symbol
    ann = ann.drop_duplicates("gene_name", keep="first")
    exact = dict(zip(ann["gene_id"], ann["gene_name"]))
    base = {}
    for gid, sym in zip(ann["gene_id"], ann["gene_name"]):
        base.setdefault(str(gid).split(".")[0], sym)
    return exact, base


def _read_signature_expression(matrix_path: str, source_columns: dict[str, str], exact_map, base_map,
                               chunksize: int = 2000):
    original_cols = list(source_columns.values())
    usecols = ["Geneid"] + original_cols
    blocks = []
    for chunk in pd.read_csv(matrix_path, sep="\t", usecols=usecols, chunksize=chunksize):
        gid = chunk["Geneid"].astype(str)
        sym = gid.map(exact_map)
        miss = sym.isna()
        if miss.any():
            sym.loc[miss] = gid.loc[miss].str.split(".").str[0].map(base_map)
        keep = sym.notna()
        if not keep.any():
            continue
        y = chunk.loc[keep, original_cols].apply(pd.to_numeric, errors="coerce")
        y.index = sym.loc[keep].astype(str).values
        blocks.append(y)
    if not blocks:
        raise RuntimeError("No LM22 signature genes were found in the raw AMP-PD TPM matrix")
    out = pd.concat(blocks, axis=0)
    if out.index.duplicated().any():
        # gene-level matrix should already be unique; sum only as a safe deterministic fallback
        out = out.groupby(level=0, sort=False).sum(min_count=1)
    # rename source columns to exact canonical sample IDs
    reverse = {v: k for k, v in source_columns.items()}
    out = out.rename(columns=reverse)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt")
    ap.add_argument("--lm22", required=True)
    ap.add_argument("--pp-cov", default=PP_COV_DEFAULT)
    ap.add_argument("--pd-cov", default=PD_COV_DEFAULT)
    ap.add_argument("--deconv-mean-tpm", type=float, default=0.01)
    ap.add_argument("--chunksize", type=int, default=2000)
    a = ap.parse_args()

    paths = AmpPdPaths(root=a.root)
    pp_ids = _read_exact_sample_ids(a.pp_cov, "PP", 1465)
    pd_ids = _read_exact_sample_ids(a.pd_cov, "PD", 1730)
    all_ids = set(pp_ids) | set(pd_ids)
    if len(all_ids) != 3195:
        raise RuntimeError("PPMI/PDBP manuscript sample IDs overlap unexpectedly")

    sig_order = _read_signature(a.lm22)
    exact_map, base_map = _annotation_maps(paths.gene_annotation_csv, sig_order)
    src_cols = _resolve_source_columns(paths.matrix_genes_tsv, all_ids)
    expr = _read_signature_expression(
        paths.matrix_genes_tsv, src_cols, exact_map, base_map, chunksize=a.chunksize
    )

    # Apply the same CIBERSORTx-only mean TPM threshold to the exact Stage-I sample universe.
    mean_tpm = expr[pp_ids + pd_ids].mean(axis=1, skipna=True)
    expr = expr.loc[mean_tpm >= float(a.deconv_mean_tpm)].copy()
    sig_present = [g for g in sig_order if g in expr.index]
    expr = expr.loc[sig_present]
    expr = expr.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    outdir = paths.data_dir / "stage1_manuscript" / "cell_deconvolution" / "cibersortx_lm22"
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for cohort, ids, label in [("PP", pp_ids, "PPMI"), ("PD", pd_ids, "PDBP")]:
        out = outdir / f"{cohort}_{label}_LM22_mixture_TPM.tsv"
        z = expr.loc[:, ids].copy()
        z.index.name = "GeneSymbol"
        z.to_csv(out, sep="\t", float_format="%.8g")
        outputs[cohort] = str(out)

    report = {
        "source_raw_tpm": paths.matrix_genes_tsv,
        "pp_cov": a.pp_cov,
        "pd_cov": a.pd_cov,
        "n_pp_samples": len(pp_ids),
        "n_pd_samples": len(pd_ids),
        "deconv_mean_tpm": float(a.deconv_mean_tpm),
        "n_lm22_signature_genes": len(sig_order),
        "n_lm22_genes_after_mapping_and_threshold": len(expr),
        "lm22_overlap_fraction": len(expr) / len(sig_order) if sig_order else None,
        "outputs": outputs,
    }
    (outdir / "stage1_mixture_files.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["lm22_overlap_fraction"] is not None and report["lm22_overlap_fraction"] < 0.95:
        print("WARNING: <95% of LM22 signature genes are available. Inspect stage1_mixture_files.json.", flush=True)


if __name__ == "__main__":
    main()
