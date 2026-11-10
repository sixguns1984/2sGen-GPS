from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd

from .config import (
    AmpPdPaths, SUBMITTED_BASE_COVARIATES, REVISED_BASE_COVARIATES,
    FOLLOWUP_METHODS, REVISED_FOLLOWUP_METHODS, ALL_FOLLOWUP_METHODS, AGE_METHODS, LEGACY_TIMES_BY_COHORT, CELL_COMPOSITION_COVARIATES,
    implementation_tag,
)
from .stage1_data import (
    get_stage1_genotype_source, load_stage1_observations, stage1_covariates,
    stage1_cache_paths, require_stage1_cache,
)
from .genotype import make_genotype_source
from ..benchmark.registry import run_method


def genotype_group_for_cohort(cohort: str) -> str:
    return "PP" if str(cohort) == "PP" else "REP"


def bfile_for_cohort(paths: AmpPdPaths, cohort: str):
    g = genotype_group_for_cohort(cohort)
    return paths.data_dir / "genotype" / g / f"amp_pd_{g}_qc"


def _augment_fixed_genotype_pcs(paths: AmpPdPaths, cohort: str, meta: pd.DataFrame, n_pcs: int = 8):
    """Add the first n PLINK genotype PCs from the raw-derived QC genotype set.

    The historical submitted model adjusted for eight genotype PCs.  The current
    v8.9 preprocessing may merge only a Tracy-Widom-selected leading subset into
    amp_pd_samples.parquet.  For the revised CTAR benchmark we read the already
    computed PLINK PCA file directly and expose the first eight PCs so the
    covariate category/count remains comparable without using the historical
    article-prepared covariate matrix.
    """
    bfile = bfile_for_cohort(paths, cohort)
    eig = Path(str(bfile) + ".pca.eigenvec")
    if not eig.exists():
        raise FileNotFoundError(
            f"{eig} not found. Re-run current-v8.9 genotype PCA before CTAR Stage-I."
        )
    z = pd.read_csv(eig, sep=r"\s+", header=None, dtype=str)
    navail = z.shape[1] - 2
    if navail < int(n_pcs):
        raise RuntimeError(f"{eig} contains only {navail} PCs; need {n_pcs}")
    z.columns = ["FID", "IID"] + [f"PC{i}" for i in range(1, navail + 1)]
    wanted = set(meta["participant_id"].astype(str))
    nf = z["FID"].astype(str).isin(wanted).sum()
    ni = z["IID"].astype(str).isin(wanted).sum()
    idcol = "FID" if nf >= ni else "IID"
    z["participant_id"] = z[idcol].astype(str)
    keep = ["participant_id"] + [f"PC{i}" for i in range(1, int(n_pcs) + 1)]
    z = z[keep].drop_duplicates("participant_id")
    z = z.rename(columns={f"PC{i}": f"ctar_gPC{i}" for i in range(1, int(n_pcs) + 1)})
    out = meta.merge(z, on="participant_id", how="left", validate="many_to_one", suffixes=("", "__ctar"))
    # Preserve the existing sample-index contract used to align expression.
    if "sample_id" not in out.columns:
        raise RuntimeError("sample_id lost while adding fixed genotype PCs")
    out.index = out["sample_id"].astype(str).values
    return out


def _pc_columns(meta: pd.DataFrame, prefix: str):
    cols = [c for c in meta.columns if re.fullmatch(re.escape(prefix) + r"\d+", str(c))]
    def key(c):
        m = re.search(r"(\d+)$", c)
        return int(m.group(1)) if m else 0
    return sorted(cols, key=key)


def _merge_current_cell_covariates(paths: AmpPdPaths, meta: pd.DataFrame, required: bool = True):
    """Attach the all-sample LM22 CLR covariates by explicit sample_id join.

    The CIBERSORTx importer writes ``LM22_CLR_covariates.parquet`` as the
    canonical all-sample cell-composition table.  Older/current metadata files
    may have been created before those columns were merged, so revised CTAR
    Stage-I must not assume they are already embedded in
    ``amp_pd_samples.parquet``.  This helper always joins the canonical parquet
    explicitly and preserves the sample-index contract used by expression.
    """
    cp = paths.data_dir / "cell_deconvolution" / "cibersortx_lm22" / "LM22_CLR_covariates.parquet"
    if not cp.exists():
        if required:
            raise FileNotFoundError(
                f"Required all-sample CIBERSORTx covariates not found: {cp}"
            )
        return meta

    cell = pd.read_parquet(cp)
    if "sample_id" not in cell.columns:
        # The canonical file stores sample_id as the parquet index.
        cell = cell.reset_index()
    if "sample_id" not in cell.columns:
        raise RuntimeError(f"{cp} does not expose sample_id as a column or index")

    missing = [c for c in CELL_COMPOSITION_COVARIATES if c not in cell.columns]
    if missing:
        raise RuntimeError(
            f"{cp} lacks required LM22 CLR covariates: {missing}"
        )

    cell = cell[["sample_id"] + list(CELL_COMPOSITION_COVARIATES)].copy()
    cell["sample_id"] = cell["sample_id"].astype(str)
    if cell["sample_id"].duplicated().any():
        dup = cell.loc[cell["sample_id"].duplicated(False), "sample_id"].head(10).tolist()
        raise RuntimeError(f"Duplicate sample_id in {cp}; examples={dup}")

    out = meta.copy()
    if "sample_id" not in out.columns:
        raise RuntimeError("metadata lacks sample_id while merging CIBERSORTx covariates")
    out["sample_id"] = out["sample_id"].astype(str)

    # The standalone parquet is the canonical source.  Drop stale/partial
    # embedded copies before the deterministic one-to-one merge.
    out = out.drop(columns=list(CELL_COMPOSITION_COVARIATES), errors="ignore")
    out = out.merge(cell, on="sample_id", how="left", validate="one_to_one")
    out.index = out["sample_id"].astype(str).values
    return out


def method_covariates(method: str, cohort: str, meta: pd.DataFrame, cell_adjust: bool = False):
    """Select model covariates from the raw-derived/current-v8.9 metadata.

    Revised irregular-follow-up methods use the same human-disease adjustment
    categories as the submitted model, but on the full longitudinal sample set:
    baseline age, sex, diagnosis, RIN, genotype PCs, participant-balanced
    expression PCs, and the six LM22 broad-cell CLR covariates.

    The historical unadjusted comparators are intentionally preserved under
    their original names. ``lin2018_regression_cov`` is the covariate-adjusted
    regression comparator used for the revised benchmark.
    """
    if method == "submitted_followup_mptga":
        base = list(SUBMITTED_BASE_COVARIATES)
        g_prefix = "gPC_PP_" if cohort == "PP" else "gPC_REP_"
        e_prefix = "ePC_BE_PP_LEGACY_" if cohort == "PP" else "ePC_BE_PD_LEGACY_"
        base += _pc_columns(meta, g_prefix)
        base += _pc_columns(meta, e_prefix)
        if cell_adjust:
            base += [c for c in CELL_COMPOSITION_COVARIATES if c in meta.columns]
    elif method in {"submitted_followup_ctar_mptga", "lin2018_regression_cov"}:
        base = list(SUBMITTED_BASE_COVARIATES)
        fixed_gpcs = [f"ctar_gPC{i}" for i in range(1, 9) if f"ctar_gPC{i}" in meta.columns]
        if len(fixed_gpcs) == 8:
            base += fixed_gpcs
        else:
            g_prefix = "gPC_PP_" if cohort == "PP" else "gPC_REP_"
            base += _pc_columns(meta, g_prefix)

        # Full-sample expression PCs are the correct analogue for an
        # all-available-visits analysis. PDBP uses a PDBP-only FULL PCA/BE set;
        # BioFIND expression is never used to define PDBP nuisance axes.
        if cohort == "PP":
            e_candidates = ["ePC_BE_PP_FULL_"]
        elif cohort == "PD":
            e_candidates = ["ePC_BE_PD_FULL_"]
        else:
            e_candidates = ["ePC_BE_REP_FULL_"]
        ecols = []
        for pref in e_candidates:
            ecols = _pc_columns(meta, pref)
            if ecols:
                break
        if not ecols:
            raise KeyError(
                f"No full-sample expression PCs found for {cohort}. "
                f"Expected one of {e_candidates}; rerun scripts_amp_pd/01e_run_expression_pc_BE.py."
            )
        base += ecols

        missing_cell = [c for c in CELL_COMPOSITION_COVARIATES if c not in meta.columns]
        if missing_cell:
            raise KeyError(
                "Revised follow-up methods require the six LM22 CLR cell-composition covariates. "
                f"Missing={missing_cell}. Import the already-computed CIBERSORTx results first."
            )
        base += list(CELL_COMPOSITION_COVARIATES)
    elif method in AGE_METHODS:
        base = list(REVISED_BASE_COVARIATES)
        g_prefix = "gPC_PP_" if cohort == "PP" else "gPC_REP_"
        e_prefix = "ePC_BE_PP_FULL_" if cohort == "PP" else "ePC_BE_REP_FULL_"
        base += _pc_columns(meta, g_prefix)
        base += _pc_columns(meta, e_prefix)
        if cohort == "REP" and "cohort_BF" in meta.columns:
            base += ["cohort_BF"]
        if cell_adjust:
            base += [c for c in CELL_COMPOSITION_COVARIATES if c in meta.columns]
    else:
        base = []

    # Keep finite, nonconstant columns. For the revised methods, cell covariates
    # are required rather than silently omitted.
    out = []
    for c in base:
        if c not in meta.columns:
            continue
        v = pd.to_numeric(meta[c], errors="coerce")
        if v.notna().sum() == 0:
            continue
        if v.nunique(dropna=True) <= 1:
            continue
        out.append(c)
    return list(dict.fromkeys(out))


def method_allowed_in_cohort(method: str, cohort: str):
    if cohort in {"BF", "REP"} and method in ALL_FOLLOWUP_METHODS:
        return False
    return True


def _strip_ensembl_version(x: str) -> str:
    x = str(x)
    return x.split(".", 1)[0] if x.startswith("ENSG") else x


def resolve_expression_gene_columns(paths: AmpPdPaths, genes, allow_unresolved: bool = False):
    """Resolve requested gene identifiers to columns in amp_pd_expression_z.parquet.

    The current-v8.9 expression matrix is stored with unique HGNC gene-symbol
    columns, whereas Stage-I manuscript tQTL results retain Ensembl gene IDs.
    Stage II therefore needs an explicit Ensembl-ID -> gene-symbol bridge.

    Returns a dict mapping each requested identifier to the physical parquet
    column containing its expression values. Direct symbol matches are kept.
    Both exact and version-stripped Ensembl identifiers are supported.
    """
    import pyarrow.parquet as pq

    genes = [str(g) for g in genes]
    expr_path = paths.data_dir / "expression" / "amp_pd_expression_z.parquet"
    schema = set(map(str, pq.ParquetFile(expr_path).schema.names))

    resolved = {g: g for g in genes if g in schema}
    missing = [g for g in genes if g not in resolved]
    if not missing:
        return resolved

    ann_path = paths.data_dir / "annotation" / "genes_used.parquet"
    if not ann_path.exists():
        ann_path = paths.data_dir / "annotation" / "gene_locations.parquet"
    if not ann_path.exists():
        raise FileNotFoundError(
            f"Cannot map Ensembl IDs to expression columns: missing genes_used.parquet "
            f"and gene_locations.parquet under {paths.data_dir / 'annotation'}"
        )

    ann = pd.read_parquet(ann_path, columns=["gene_id", "gene_name"]).copy()
    ann["gene_id"] = ann["gene_id"].astype(str)
    ann["gene_name"] = ann["gene_name"].astype(str)
    ann = ann[ann["gene_name"].isin(schema)].drop_duplicates(["gene_id", "gene_name"])

    exact = {}
    base = {}
    ambiguous_base = set()
    for gid, gname in ann[["gene_id", "gene_name"]].itertuples(index=False):
        exact.setdefault(str(gid), str(gname))
        b = _strip_ensembl_version(gid)
        if b in base and base[b] != str(gname):
            ambiguous_base.add(b)
        else:
            base.setdefault(b, str(gname))
    for b in ambiguous_base:
        base.pop(b, None)

    unresolved = []
    for g in missing:
        src = exact.get(g)
        if src is None:
            src = base.get(_strip_ensembl_version(g))
        if src is not None and src in schema:
            resolved[g] = src
        else:
            unresolved.append(g)

    if unresolved and not allow_unresolved:
        ex = unresolved[:10]
        raise KeyError(
            f"Could not resolve {len(unresolved)} requested gene identifier(s) to "
            f"amp_pd_expression_z.parquet columns; examples={ex}. "
            f"Stage-I uses Ensembl IDs while current-v8.9 expression uses gene symbols."
        )
    return resolved


def load_cohort_metadata(paths: AmpPdPaths, cohort: str):
    """Load current-v8.9 metadata once for a cohort.

    Stage-II processes many genes on the same observation universe.  Keeping
    metadata loading separate from expression-column loading allows worker
    processes to cache this relatively small table instead of re-reading and
    re-joining CIBERSORTx covariates for every gene.
    """
    meta = pd.read_parquet(paths.data_dir / "metadata" / "amp_pd_samples.parquet")
    meta = _merge_current_cell_covariates(paths, meta, required=False)
    if str(cohort) == "REP":
        meta = meta[meta.cohort.astype(str).isin(["PD", "BF"])].copy()
    else:
        meta = meta[meta.cohort.astype(str) == str(cohort)].copy()
    meta = meta.sort_values(
        ["cohort", "participant_id", "followup_time", "sample_id"], kind="mergesort"
    ).reset_index(drop=True)
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta.index = meta["sample_id"].astype(str).values
    return meta


def load_expression_columns(paths: AmpPdPaths, genes):
    """Read requested expression columns once, indexed by sample_id.

    This low-level helper is useful for Stage-II gene batches shared across
    PPMI/PDBP/BioFIND.  It avoids reading the same parquet columns separately
    for each cohort.  Requested identifiers may be either physical expression
    columns or Stage-I Ensembl IDs; output columns keep the requested names.
    """
    genes = [str(g) for g in genes]
    if not genes:
        return pd.DataFrame()
    gene_map = resolve_expression_gene_columns(paths, genes)
    physical = list(dict.fromkeys(gene_map[g] for g in genes))
    expr = pd.read_parquet(
        paths.data_dir / "expression" / "amp_pd_expression_z.parquet",
        columns=["sample_id"] + physical,
    )
    expr["sample_id"] = expr["sample_id"].astype(str)
    expr = expr.set_index("sample_id")
    out = pd.DataFrame(index=expr.index)
    for g in genes:
        out[g] = pd.to_numeric(expr[gene_map[g]], errors="coerce")
    return out


def load_expression_for_meta(paths: AmpPdPaths, meta: pd.DataFrame, genes):
    """Load requested expression columns and align to cached metadata."""
    expr = load_expression_columns(paths, genes)
    if expr.empty and len(genes) == 0:
        return pd.DataFrame(index=meta.index.copy())
    expr = expr.reindex(meta["sample_id"].astype(str))
    expr.index = meta["sample_id"].astype(str).values
    return expr


def load_cohort_observations(paths: AmpPdPaths, cohort: str, genes):
    """Compatibility wrapper: load cohort metadata plus selected expression genes."""
    meta = load_cohort_metadata(paths, cohort)
    expr = load_expression_for_meta(paths, meta, genes)
    meta.index = expr.index
    return meta, expr


def _drop_missing_covariates(meta, expr, covars):
    needed = ["participant_id"] + list(covars)
    ok = np.ones(len(meta), bool)
    for c in needed:
        if c not in meta.columns:
            raise KeyError(f"Required covariate column missing: {c}")
        ok &= meta[c].notna().to_numpy()
    meta = meta.loc[ok].copy()
    expr = expr.loc[meta.index].copy()
    return meta, expr


def _prepare_irregular_followup_rows(meta, expr, id_col="participant_id", time_col="followup_time"):
    """Keep all usable irregular visits while resolving exact duplicate timepoints.

    Continuous-time AR(1) is singular if the same participant contributes two
    observations at exactly the same follow-up time.  When technical/repeat RNA
    samples exist for one participant-time, keep the row with the highest RIN
    (then lexicographically smallest sample_id for deterministic tie breaking).
    This is not complete-case filtering: participants retain every distinct
    observed follow-up time and may have any visit count >=1.
    """
    meta = meta.copy()
    meta[time_col] = pd.to_numeric(meta[time_col], errors="coerce")
    ok = meta[time_col].notna().to_numpy()
    meta = meta.loc[ok].copy()
    expr = expr.loc[meta.index].copy()
    if meta.empty:
        return meta, expr, 0

    rin = pd.to_numeric(meta.get("RIN", np.nan), errors="coerce")
    meta["__ctar_rin_sort"] = rin.fillna(-np.inf)
    meta["__ctar_sample_sort"] = meta.get("sample_id", meta.index.astype(str)).astype(str)
    meta = meta.sort_values(
        [id_col, time_col, "__ctar_rin_sort", "__ctar_sample_sort"],
        ascending=[True, True, False, True], kind="mergesort"
    )
    dup = meta.duplicated([id_col, time_col], keep="first")
    ndup = int(dup.sum())
    meta = meta.loc[~dup].copy()
    meta = meta.sort_values([id_col, time_col, "__ctar_sample_sort"], kind="mergesort")
    expr = expr.loc[meta.index].copy()
    meta = meta.drop(columns=["__ctar_rin_sort", "__ctar_sample_sort"], errors="ignore")
    return meta, expr, ndup


def run_gene_chunk(paths: AmpPdPaths, cohort: str, method: str, genes,
                   out_file: str, device="cpu", batch_size=2048, rho_points=40,
                   legacy_times=None, spline_df=5, variance_df=5,
                   genotype_mode="dosage", legacy_genotype_mode="dummies",
                   dtype="float64", cell_adjust=False, cis_path_override=None,
                   data_mode="current_v89", stage1_cell_mode="primary",
                   ctar_reference_interval=1.0, ctar_rho_max=0.99):
    if not method_allowed_in_cohort(method, cohort):
        return {"skipped": True, "reason": f"{method} is not applicable to {cohort}"}

    if method in REVISED_FOLLOWUP_METHODS and data_mode != "current_v89":
        raise ValueError(
            f"{method} is defined on the raw-derived/current-v8.9 longitudinal inputs, "
            "not the historical stage1_manuscript matrices. Use --data-mode current_v89."
        )

    genes = list(map(str, genes))
    if data_mode == "stage1_manuscript":
        if cohort not in {"PP", "PD"}:
            raise ValueError("Stage-I manuscript data mode supports only PP and PD")
        cache = require_stage1_cache(paths, cohort, require_cis=(cohort == "PP"))
        default_cis = stage1_cache_paths(paths, "PP")["cis"]
        cis_path = Path(cis_path_override) if cis_path_override else default_cis
    elif data_mode == "current_v89":
        cis_path = Path(cis_path_override) if cis_path_override else paths.data_dir / "cis" / "cis_1000kb_PP.parquet"
    else:
        raise ValueError(f"Unknown data_mode={data_mode!r}")

    cis = pd.read_parquet(cis_path, filters=[("geneid", "in", genes)])
    if cis.empty:
        pd.DataFrame().to_parquet(out_file, index=False)
        return {"n_pairs": 0}

    genes = sorted(set(cis.geneid.astype(str)))
    if data_mode == "stage1_manuscript":
        meta, expr = load_stage1_observations(paths, cohort, genes)
        if stage1_cell_mode not in {"primary", "nocell_audit"}:
            raise ValueError(f"Unknown stage1_cell_mode={stage1_cell_mode!r}")
        if stage1_cell_mode == "nocell_audit" and method != "submitted_followup_mptga":
            raise ValueError("Stage-I no-cell audit is defined only for submitted_followup_mptga")
        covars = stage1_covariates(
            method, cohort, meta, cell_adjust=cell_adjust,
            include_cell_composition=(stage1_cell_mode == "primary"),
        )
        # The manuscript CSVs are already QC'ed/complete. Missing required covariates
        # are a hard error rather than a reason to alter the analysis sample.
        for c in covars:
            if meta[c].isna().any():
                raise RuntimeError(f"Unexpected missing Stage-I manuscript covariate {c} in {cohort}")
    else:
        meta, expr = load_cohort_observations(paths, cohort, genes)
        if method in REVISED_FOLLOWUP_METHODS:
            meta = _augment_fixed_genotype_pcs(paths, cohort, meta, n_pcs=8)
            expr = expr.reindex(meta.index)
            meta, expr, _ = _prepare_irregular_followup_rows(meta, expr)
        covars = method_covariates(method, cohort, meta, cell_adjust=cell_adjust)
        meta, expr = _drop_missing_covariates(meta, expr, covars)
        if meta.empty:
            pd.DataFrame().to_parquet(out_file, index=False)
            return {"n_pairs": 0, "reason": "no observations after covariate filtering"}

    snps = sorted(set(cis.snpid.astype(str)))
    pids = pd.unique(meta.participant_id.astype(str))
    # Stage-I uses only legacy hard-call binary coding.  Keep the genotype cache
    # compact as int8 until the scanner forms the two floating-point channels;
    # this cuts genotype memory traffic by 8x versus eager float64 materialization.
    geno_dtype = np.int8 if data_mode == "stage1_manuscript" else (np.float64 if str(dtype) == "float64" else np.float32)
    if data_mode == "stage1_manuscript":
        gsrc = get_stage1_genotype_source(paths, cohort)
        # Exact legacy hard calls are preserved. Missing calls are represented as
        # -1 and encoded by the legacy binary mapping; they are never mean-imputed.
        geno = gsrc.read(pids, snps, mean_impute=False, dtype=geno_dtype)
        bfile = "stage1_PP_memmap" if cohort == "PP" else "stage1_PD_rawWGS_PLINK_subset"
    else:
        bfile = bfile_for_cohort(paths, cohort)
        gsrc = make_genotype_source(str(bfile), paths.plink19, prefer_bed_reader=True)
        if legacy_genotype_mode == "binary" and method in ALL_FOLLOWUP_METHODS:
            geno = gsrc.read(pids, snps, mean_impute=False, dtype=geno_dtype)
            # Preserve the historical hard-call missing category for binary coding.
            geno = geno.fillna(-1.0)
        else:
            geno = gsrc.read(pids, snps, mean_impute=True, dtype=geno_dtype)

    cfg = {
        "covariates": covars,
        "id_col": "participant_id",
        "visit_col": "visit",
        "submitted_visit_col": "followup_time",
        "followup_col": "followup_time",
        "age_col": "age",
        "required_visits": list(LEGACY_TIMES_BY_COHORT.get(cohort, []) if legacy_times is None else legacy_times),
        "batch_size": int(batch_size),
        "rho_points": int(rho_points),
        "spline_df": int(spline_df),
        "variance_df": int(variance_df),
        "genotype_mode": genotype_mode,
        "legacy_genotype_mode": legacy_genotype_mode,
        "dtype": dtype,
        "device": device,
        "rho_grid_mode": "legacy_exact" if data_mode == "stage1_manuscript" else "linspace_098",
        "ctar_reference_interval": float(ctar_reference_interval),
        "ctar_rho_max": float(ctar_rho_max),
    }
    res = run_method(method, meta, expr, geno, cis[["snpid", "geneid"]], cfg)
    if len(res):
        res.insert(0, "cohort", cohort)
        res.insert(1, "analysis_genotype_group", genotype_group_for_cohort(cohort))
        res.insert(2, "compute_dtype", dtype)
        res.insert(3, "device_used", device)
        res.insert(4, "requested_batch_size", int(batch_size))
        res.insert(5, "n_covariates_used", len(covars))
        res.insert(6, "data_mode", data_mode)
        tag = implementation_tag(method, data_mode, stage1_cell_mode=stage1_cell_mode)
        if "implementation_tag" in res.columns:
            res["implementation_tag"] = tag if tag is not None else res["implementation_tag"]
        else:
            res.insert(7, "implementation_tag", tag)
        if "stage1_cell_mode" in res.columns:
            res["stage1_cell_mode"] = stage1_cell_mode
        else:
            res.insert(8, "stage1_cell_mode", stage1_cell_mode if data_mode == "stage1_manuscript" else "not_applicable")
        for c in ["FDR", "Interaction_FDR"]:
            if c in res.columns:
                res = res.drop(columns=c)
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out_file, index=False, compression="zstd")
    return {
        "n_pairs": int(len(res)),
        "n_observations": int(len(meta)),
        "n_participants": int(meta.participant_id.nunique()),
        "n_covariates": int(len(covars)),
        "bfile": str(bfile),
        "data_mode": data_mode,
        "stage1_cell_mode": stage1_cell_mode if data_mode == "stage1_manuscript" else "not_applicable",
        "covariates": list(covars),
    }


def make_gene_chunks(paths: AmpPdPaths, genes_per_chunk=50, cis_override=None, data_mode="current_v89"):
    stage1_default = stage1_cache_paths(paths, "PP")["cis"]
    use_stage1_gene_manifest = (
        data_mode == "stage1_manuscript"
        and (cis_override is None or Path(cis_override).resolve() == stage1_default.resolve())
    )
    if use_stage1_gene_manifest:
        p = require_stage1_cache(paths, "PP", require_cis=True)
        genes = [x.strip() for x in p["cis_genes"].read_text().splitlines() if x.strip()]
    else:
        cis_path = Path(cis_override) if cis_override else (
            stage1_default if data_mode == "stage1_manuscript"
            else paths.data_dir / "cis" / "cis_1000kb_PP.parquet"
        )
        # DISTINCT in DuckDB avoids materializing tens of millions of repeated gene IDs.
        import duckdb
        con = duckdb.connect()
        try:
            qpath = str(cis_path).replace("'", "''")
            genes = [str(r[0]) for r in con.execute(
                f"SELECT DISTINCT CAST(geneid AS VARCHAR) FROM read_parquet('{qpath}')"
            ).fetchall()]
        finally:
            con.close()
    return [genes[i:i + int(genes_per_chunk)] for i in range(0, len(genes), int(genes_per_chunk))]
