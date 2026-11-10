from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .config import AmpPdPaths, CELL_COMPOSITION_COVARIATES, COHORT_CODE_TO_NAME
from .preprocess import load_gene_annotation, read_matrix_genes_tpm, filter_tpm_to_gene_symbols

# Exact standard LM22 cell labels.
LM22_GROUPS = {
    "B": [
        "B cells naive",
        "B cells memory",
        "Plasma cells",
    ],
    "CD4T": [
        "T cells CD4 naive",
        "T cells CD4 memory resting",
        "T cells CD4 memory activated",
        "T cells follicular helper",
        "T cells regulatory (Tregs)",
    ],
    "CD8T": [
        "T cells CD8",
    ],
    "NK": [
        "NK cells resting",
        "NK cells activated",
    ],
    "Monocytes": [
        "Monocytes",
    ],
    "Neutrophils": [
        "Neutrophils",
    ],
    "Other": [
        "T cells gamma delta",
        "Macrophages M0",
        "Macrophages M1",
        "Macrophages M2",
        "Dendritic cells resting",
        "Dendritic cells activated",
        "Mast cells resting",
        "Mast cells activated",
        "Eosinophils",
    ],
}

LM22_CELL_TYPES = [x for xs in LM22_GROUPS.values() for x in xs]
CIBERSORTX_DIAGNOSTIC_CANDIDATES = {
    "P-value", "P value", "P.Value", "p-value",
    "Correlation", "RMSE", "Absolute score",
}

def export_cibersortx_inputs(
    paths: AmpPdPaths,
    split_by_cohort: bool = True,
    lm22_path=None,
    deconv_mean_tpm: float | None = None,
    signature_only: bool = False,
):
    """Export QC-retained raw TPM matrices for CIBERSORTx LM22.

    Important separation from the tQTL pipeline
    ------------------------------------------
    This function never rewrites the frozen QTL expression matrix, genotype PCs,
    expression PCs, or metadata.  If ``deconv_mean_tpm`` is supplied, it rebuilds
    the CIBERSORTx-only raw TPM matrix directly from the original AMP-PD TPM source
    and then subsets it to the already-finalized samples in amp_pd_samples.parquet.

    The recommended LM22 setting after the threshold-sensitivity audit is
    ``deconv_mean_tpm=0.01``.  In the current data this retains 529/547 LM22 genes
    (96.7%); lowering further to 0 adds only one additional LM22 gene.

    If ``signature_only=True``, output mixture files are restricted to genes shared
    with LM22. This is computationally efficient and does not change the information
    available to an LM22 deconvolution because only genes shared by the signature
    matrix and mixture can contribute to the fit.
    """
    mp = paths.data_dir/"metadata"/"amp_pd_samples.parquet"
    if not mp.exists():
        raise FileNotFoundError(f"{mp} not found. Run preprocessing first.")

    meta = pd.read_parquet(mp)
    if "sample_id" not in meta.columns or "rna_column_id" not in meta.columns:
        raise ValueError("Frozen metadata requires sample_id and rna_column_id.")

    # Build/read CIBERSORTx-only raw TPM without modifying any frozen QTL objects.
    if deconv_mean_tpm is None:
        fp = paths.data_dir/"expression"/"amp_pd_expression_tpm_raw_qc.parquet"
        if not fp.exists():
            raise FileNotFoundError(
                f"{fp} not found. Either run preprocessing first or pass "
                "--deconv-mean-tpm to rebuild CIBERSORTx input from the raw TPM source."
            )
        raw = pd.read_parquet(fp)
        threshold_used = None
        source_mode = "existing_preprocessed_deconv_parquet"
    else:
        ann_all = load_gene_annotation(paths.gene_annotation_csv, autosome_only=False)
        rna_geneid = read_matrix_genes_tpm(paths.matrix_genes_tsv)
        tpm_deconv, _ = filter_tpm_to_gene_symbols(
            rna_geneid,
            ann_all,
            mean_tpm_threshold=float(deconv_mean_tpm),
            autosome_only=False,
        )

        rid_to_sid = dict(
            zip(meta["rna_column_id"].astype(str), meta["sample_id"].astype(str))
        )
        source_cols = [c for c in tpm_deconv.columns.astype(str) if c in rid_to_sid]
        if not source_cols:
            raise RuntimeError("No frozen RNA samples matched the raw TPM matrix.")

        tmp = tpm_deconv.loc[:, source_cols].copy()
        tmp = tmp.rename(columns=rid_to_sid)
        wanted = [s for s in meta["sample_id"].astype(str) if s in tmp.columns]
        tmp = tmp.loc[:, wanted]

        raw = tmp.T.copy()
        raw.index.name = "sample_id"
        raw = raw.reset_index()
        threshold_used = float(deconv_mean_tpm)
        source_mode = "rebuilt_from_original_raw_TPM_for_CIBERSORTx_only"

    if "sample_id" not in raw.columns:
        raise ValueError("Raw TPM matrix must contain sample_id.")
    raw["sample_id"] = raw["sample_id"].astype(str)
    if raw["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id in raw TPM matrix.")
    raw = raw.set_index("sample_id")

    outdir = paths.data_dir/"cell_deconvolution"/"cibersortx_lm22"
    outdir.mkdir(parents=True, exist_ok=True)

    sig_genes = None
    lm22_report = None
    if lm22_path:
        lp = Path(lm22_path)
        if not lp.exists():
            raise FileNotFoundError(f"LM22 signature matrix not found: {lm22_path}")
        try:
            sig = pd.read_csv(lp, sep="\t")
            if sig.shape[1] <= 2:
                sig = pd.read_csv(lp, sep=None, engine="python")
        except Exception:
            sig = pd.read_csv(lp, sep=None, engine="python")

        sig_genes = set(sig.iloc[:, 0].astype(str))
        mix_genes = set(raw.columns.astype(str))
        overlap = sorted(sig_genes & mix_genes)
        missing = sorted(sig_genes - mix_genes)
        lm22_report = {
            "lm22_file": str(lp),
            "deconv_mean_tpm": threshold_used,
            "source_mode": source_mode,
            "signature_only_output": bool(signature_only),
            "n_signature_genes": len(sig_genes),
            "n_expression_genes_after_threshold": len(mix_genes),
            "n_overlap": len(overlap),
            "overlap_fraction": (len(overlap) / len(sig_genes)) if sig_genes else None,
            "n_missing": len(missing),
            "missing_genes": missing,
        }

        if signature_only:
            # Preserve LM22's own row order where possible.
            sig_order = [g for g in sig.iloc[:, 0].astype(str).tolist() if g in mix_genes]
            raw = raw.loc[:, sig_order].copy()

    elif signature_only:
        raise ValueError("--signature-only requires --lm22.")

    outputs = {}
    cohorts = ["PP", "PD", "BF"] if split_by_cohort else ["ALL"]
    for cohort in cohorts:
        if cohort == "ALL":
            sids = meta["sample_id"].astype(str).tolist()
            label = "ALL"
        else:
            sids = meta.loc[
                meta["cohort"].astype(str).eq(cohort), "sample_id"
            ].astype(str).tolist()
            label = COHORT_CODE_TO_NAME.get(cohort, cohort)

        sids = [s for s in sids if s in raw.index]
        if not sids:
            continue

        x = raw.loc[sids].T.copy()
        x.index = x.index.astype(str)
        x.index.name = "GeneSymbol"
        x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if x.index.duplicated().any():
            x = x.groupby(level=0, sort=False).sum()

        out = outdir/f"{cohort}_{label}_LM22_mixture_TPM.tsv"
        x.to_csv(out, sep="\t", float_format="%.8g")
        outputs[cohort] = str(out)

    (outdir/"mixture_files.json").write_text(
        json.dumps(outputs, indent=2), encoding="utf-8"
    )
    if lm22_report is not None:
        (outdir/"LM22_signature_overlap.json").write_text(
            json.dumps(lm22_report, indent=2), encoding="utf-8"
        )
        if (
            lm22_report["overlap_fraction"] is not None
            and lm22_report["overlap_fraction"] < 0.95
        ):
            print(
                "WARNING: <95% of LM22 signature genes are present. "
                "Inspect LM22_signature_overlap.json before running CIBERSORTx.",
                flush=True,
            )

    return {"mixtures": outputs, "lm22_overlap": lm22_report}

def _read_delimited(path: str) -> pd.DataFrame:
    p = Path(path)
    # CIBERSORTx output is normally tab-delimited, but tolerate CSV output.
    try:
        z = pd.read_csv(p, sep="\t")
        if z.shape[1] <= 2:
            z = pd.read_csv(p)
    except Exception:
        z = pd.read_csv(p, sep=None, engine="python")
    return z

def read_cibersortx_lm22(path: str):
    """Read CIBERSORTx Fractions output and validate all 22 LM22 columns.

    Returns:
      fractions: sample x 22 LM22 relative fractions
      diagnostics: sample x available diagnostic columns
    """
    z = _read_delimited(path)
    if z.empty:
        raise ValueError(f"Empty CIBERSORTx result: {path}")

    first = z.columns[0]
    if str(first) not in LM22_CELL_TYPES:
        z = z.rename(columns={first: "sample_id"})
        z["sample_id"] = z["sample_id"].astype(str)
        z = z.set_index("sample_id")
    else:
        raise ValueError(
            "CIBERSORTx result appears transposed. Expected samples in rows and "
            "LM22 cell types in columns."
        )

    missing = [c for c in LM22_CELL_TYPES if c not in z.columns]
    if missing:
        raise ValueError(
            f"CIBERSORTx output is not a complete LM22 Fractions result. "
            f"Missing {len(missing)} columns, e.g. {missing[:6]}"
        )

    frac = z[LM22_CELL_TYPES].apply(pd.to_numeric, errors="coerce")
    diag_cols = [c for c in z.columns if c not in LM22_CELL_TYPES]
    diag = z[diag_cols].copy() if diag_cols else pd.DataFrame(index=z.index)
    for c in diag.columns:
        converted = pd.to_numeric(diag[c], errors="coerce")
        # Keep numeric diagnostics when conversion is meaningful; otherwise
        # preserve the original text column.
        if converted.notna().sum() > 0:
            diag[c] = converted
    frac.index = frac.index.astype(str)
    diag.index = diag.index.astype(str)
    frac.index.name = diag.index.name = "sample_id"
    return frac, diag

def aggregate_lm22(fractions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate LM22 into seven broad compartments covering all 22 cell types."""
    broad = pd.DataFrame(index=fractions.index)
    for group, cols in LM22_GROUPS.items():
        broad[group] = fractions[cols].sum(axis=1, min_count=1)

    # CIBERSORTx relative fractions should sum to 1, but close them explicitly
    # to avoid numerical drift and to make the compositional transform exact.
    rs = broad.sum(axis=1)
    bad = (~np.isfinite(rs)) | (rs <= 0)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} samples have invalid LM22 total fractions.")
    broad = broad.div(rs, axis=0)
    broad.index.name = "sample_id"
    return broad

def clr_transform_broad(broad: pd.DataFrame, pseudocount: float = 1e-5) -> pd.DataFrame:
    """Centered log-ratio transform of the seven-part composition.

    The CLR is computed using all seven broad compartments. "Other" remains
    part of the geometric mean but is dropped from the QTL design matrix to
    avoid exact linear dependence among the CLR coordinates.
    """
    cols = ["B", "CD4T", "CD8T", "NK", "Monocytes", "Neutrophils", "Other"]
    x = broad[cols].astype(float).clip(lower=0.0) + float(pseudocount)
    logx = np.log(x)
    clr = logx.sub(logx.mean(axis=1), axis=0)
    clr.columns = [f"cibx_clr_{c}" for c in cols]
    return clr

def standardize_within_cohort(clr: pd.DataFrame, sample_meta: pd.DataFrame) -> pd.DataFrame:
    """Z-standardize CLR covariates within PPMI/PDBP/BioFIND separately."""
    cohort_map = sample_meta.set_index("sample_id")["cohort"].astype(str)
    out = clr.copy()
    out["cohort"] = out.index.map(cohort_map)
    base_cols = [c for c in clr.columns if c != "cibx_clr_Other"]
    for c in base_cols:
        out[c+"_z"] = np.nan
        for cohort, idx in out.groupby("cohort").groups.items():
            v = pd.to_numeric(out.loc[idx, c], errors="coerce")
            sd = v.std(ddof=1)
            if not np.isfinite(sd) or sd <= 0:
                sd = 1.0
            out.loc[idx, c+"_z"] = (v - v.mean()) / sd
    return out

def import_cibersortx_lm22_results(
    paths: AmpPdPaths,
    ppmi_result: str,
    pdbp_result: str,
    biofind_result: str,
    pseudocount: float = 1e-5,
):
    """Import three independently-run CIBERSORTx-LM22 Fractions results.

    PPMI, PDBP and BioFIND are deconvolved separately using identical LM22
    settings. Broad-cell CLR covariates are merged into amp_pd_samples.parquet.
    """
    meta_path = paths.data_dir/"metadata"/"amp_pd_samples.parquet"
    meta = pd.read_parquet(meta_path)
    expected = {
        "PP": set(meta.loc[meta.cohort.astype(str).eq("PP"), "sample_id"].astype(str)),
        "PD": set(meta.loc[meta.cohort.astype(str).eq("PD"), "sample_id"].astype(str)),
        "BF": set(meta.loc[meta.cohort.astype(str).eq("BF"), "sample_id"].astype(str)),
    }
    result_paths = {"PP": ppmi_result, "PD": pdbp_result, "BF": biofind_result}

    fine_all, broad_all, diag_all = [], [], []
    qc_rows = []
    for cohort, path in result_paths.items():
        frac, diag = read_cibersortx_lm22(path)
        got = set(frac.index.astype(str))
        missing = expected[cohort] - got
        extra = got - expected[cohort]
        # Extra samples are not used; missing samples are reported and will be
        # excluded only from the cell-adjusted sensitivity analysis.
        keep = sorted(got & expected[cohort])
        frac = frac.loc[keep].copy()
        diag = diag.loc[keep].copy()

        broad = aggregate_lm22(frac)
        broad["cohort"] = cohort
        frac["cohort"] = cohort
        diag["cohort"] = cohort
        fine_all.append(frac)
        broad_all.append(broad)
        diag_all.append(diag)
        qc_rows.append({
            "cohort": cohort,
            "expected_samples": len(expected[cohort]),
            "cibersortx_samples": len(got),
            "matched_samples": len(keep),
            "missing_expected_samples": len(missing),
            "extra_result_samples": len(extra),
        })

    fine = pd.concat(fine_all, axis=0)
    broad = pd.concat(broad_all, axis=0)
    diag = pd.concat(diag_all, axis=0)

    clr = clr_transform_broad(broad.drop(columns=["cohort"]), pseudocount=pseudocount)
    transformed = standardize_within_cohort(clr, meta)

    outdir = paths.data_dir/"cell_deconvolution"/"cibersortx_lm22"
    outdir.mkdir(parents=True, exist_ok=True)
    fine.to_parquet(outdir/"LM22_fine_fractions.parquet", compression="zstd")
    broad.to_parquet(outdir/"LM22_broad_fractions.parquet", compression="zstd")
    diag.to_parquet(outdir/"LM22_diagnostics.parquet", compression="zstd")
    transformed.to_parquet(outdir/"LM22_CLR_covariates.parquet", compression="zstd")
    pd.DataFrame(qc_rows).to_csv(outdir/"LM22_import_QC.csv", index=False)

    # Deterministic re-merge: remove prior CIBERSORTx covariates/diagnostics.
    dropcols = [c for c in meta.columns if c.startswith("cibx_")]
    meta = meta.drop(columns=dropcols, errors="ignore")
    merge_cols = ["sample_id"] + CELL_COMPOSITION_COVARIATES
    m = transformed.reset_index()[merge_cols]
    meta = meta.merge(m, on="sample_id", how="left", validate="one_to_one")
    meta.to_parquet(meta_path, index=False, compression="zstd")

    report = pd.DataFrame(qc_rows)
    report["samples_with_all_qtl_cell_covariates"] = [
        int(meta.loc[meta.cohort.astype(str).eq(c), CELL_COMPOSITION_COVARIATES]
            .notna().all(axis=1).sum())
        for c in report["cohort"]
    ]
    report.to_csv(outdir/"LM22_merge_report.csv", index=False)
    return report
