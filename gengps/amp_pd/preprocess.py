from __future__ import annotations
import json, subprocess, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .config import (
    AmpPdPaths, COHORT_CODE_TO_NAME, VISIT_TO_YEARS, GENOTYPE_QC_GROUPS,
    DEFAULT_QTL_MEAN_TPM, DEFAULT_DECONV_MEAN_TPM, DEFAULT_MAX_PCS,
    DEFAULT_MANUAL_GPCS,
)
from .eigensoft import parse_map, select_genotype_pc_count


AUTOSOMES = {str(i) for i in range(1, 23)}


def _norm_sample_id(x: str) -> str:
    return str(x).replace(".", "-")


def parse_amp_pd_sample_id(sample_id: str):
    s = _norm_sample_id(sample_id)
    parts = s.split("-")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse AMP-PD sample id: {sample_id}")
    cohort = parts[0]
    participant_id = "-".join(parts[:2])
    visit = parts[2]
    canonical_sample_id = "-".join(parts[:3])
    return cohort, participant_id, canonical_sample_id, visit


def load_gene_annotation(path: str, min_width: int = 200, autosome_only: bool = False) -> pd.DataFrame:
    g = pd.read_csv(path, index_col=0)
    needed = {"seqnames", "start", "end", "width", "type", "gene_id", "gene_name"}
    miss = needed - set(g.columns)
    if miss:
        raise ValueError(f"Gene annotation missing columns: {sorted(miss)}")
    g = g.loc[
        (g["type"].astype(str) == "gene")
        & (pd.to_numeric(g["width"], errors="coerce") >= int(min_width))
    ].copy()
    g["gene_id"] = g["gene_id"].astype(str)
    g["gene_name"] = g["gene_name"].astype(str)
    g["chr"] = g["seqnames"].astype(str).str.replace(r"^chr", "", regex=True)
    if autosome_only:
        g = g[g["chr"].astype(str).isin(AUTOSOMES)].copy()
    g["start"] = pd.to_numeric(g["start"], errors="coerce").astype("Int64")
    g["end"] = pd.to_numeric(g["end"], errors="coerce").astype("Int64")
    g = g.dropna(subset=["gene_name", "start", "end"])
    if "gene_type" not in g.columns:
        g["gene_type"] = np.nan
    return g[["chr", "seqnames", "start", "end", "width", "strand",
              "gene_id", "gene_name", "gene_type"]].reset_index(drop=True)


def read_matrix_genes_tpm(matrix_path: str) -> pd.DataFrame:
    rna = pd.read_csv(matrix_path, sep="\t")
    if "Geneid" not in rna.columns:
        raise ValueError("matrix.genes.tsv must contain a Geneid column.")
    rna.columns = [rna.columns[0]] + [_norm_sample_id(c) for c in rna.columns[1:]]
    rna["Geneid"] = rna["Geneid"].astype(str)
    rna = rna.set_index("Geneid")
    return rna.apply(pd.to_numeric, errors="coerce")


def filter_tpm_to_gene_symbols(
    rna_geneid_by_sample: pd.DataFrame,
    gene_annotation: pd.DataFrame,
    mean_tpm_threshold: float,
    autosome_only: bool = False,
):
    """Filter TPM matrix and convert Ensembl gene IDs to unique HGNC symbols."""
    ann = gene_annotation.copy()
    if autosome_only:
        ann = ann[ann["chr"].astype(str).isin(AUTOSOMES)].copy()
    ann["gene_id"] = ann["gene_id"].astype(str)
    ann = ann.drop_duplicates("gene_id", keep="first").set_index("gene_id", drop=False)

    means = rna_geneid_by_sample.mean(axis=1, skipna=True)
    keep = means >= float(mean_tpm_threshold)
    rna = rna_geneid_by_sample.loc[keep].copy()

    common = ann.index.intersection(rna.index, sort=False)
    ann2 = ann.loc[common].copy()
    ann2 = ann2.loc[~ann2["gene_name"].duplicated(keep="first")].reset_index(drop=True)
    rna = rna.loc[ann2["gene_id"].astype(str)].copy()
    rna.index = ann2["gene_name"].astype(str).values
    return rna, ann2


def zscore_genes_after_log2(tpm_gene_by_sample: pd.DataFrame):
    x = np.log2(tpm_gene_by_sample.to_numpy(dtype=np.float64) + 1.0)
    mean = np.nanmean(x, axis=1, keepdims=True)
    sd = np.nanstd(x, axis=1, ddof=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd == 0)] = 1.0
    z = (x - mean) / sd
    return pd.DataFrame(z, index=tpm_gene_by_sample.index, columns=tpm_gene_by_sample.columns)


def expression_sample_qc(expr_gene_by_sample: pd.DataFrame, z_cut: float = 5.0):
    X = expr_gene_by_sample.to_numpy(dtype=np.float64)
    med = np.nanmedian(X, axis=0)
    q75 = np.nanpercentile(X, 75, axis=0)
    q25 = np.nanpercentile(X, 25, axis=0)
    iqr = q75 - q25

    def z(v):
        s = np.nanstd(v, ddof=1)
        return (v - np.nanmean(v)) / (s if np.isfinite(s) and s > 0 else 1.0)

    medz, iqrz = z(med), z(iqr)
    pca = PCA(n_components=2, svd_solver="randomized", random_state=1)
    scores = pca.fit_transform(np.nan_to_num(X.T, nan=0.0))
    pc1z, pc2z = z(scores[:, 0]), z(scores[:, 1])
    out_pc = (np.abs(pc1z) > z_cut) | (np.abs(pc2z) > z_cut)
    out_med = np.abs(medz) > z_cut
    out_iqr = np.abs(iqrz) > z_cut
    out_any = out_pc | out_med | out_iqr
    details = pd.DataFrame({
        "sample_id": expr_gene_by_sample.columns.astype(str),
        "outlier_pc": out_pc,
        "outlier_median": out_med,
        "outlier_iqr": out_iqr,
        "median_z": medz,
        "iqr_z": iqrz,
        "pc1_z": pc1z,
        "pc2_z": pc2z,
        "outlier_any": out_any,
    })
    kept = details.loc[~details.outlier_any, "sample_id"].tolist()
    return expr_gene_by_sample.loc[:, kept].copy(), details


def build_sample_metadata(sample_ids, paths: AmpPdPaths):
    rows = []
    for s in sample_ids:
        c, p, ss, v = parse_amp_pd_sample_id(s)
        rows.append({
            "cohort": c,
            "cohort_name": COHORT_CODE_TO_NAME.get(c, c),
            "participant_id": p,
            "sample_id": ss,
            "rna_column_id": _norm_sample_id(s),
            "visit": v,
        })
    meta = pd.DataFrame(rows)

    demo = pd.read_csv(paths.demographics_csv)
    demo["participant_id"] = demo["participant_id"].astype(str)
    keep = [c for c in ["participant_id", "age_at_baseline", "sex", "education_level_years"] if c in demo.columns]
    demo = demo[keep].drop_duplicates("participant_id", keep="first")
    meta = meta.merge(demo, on="participant_id", how="left")

    mh = pd.read_csv(paths.medical_history_csv)
    mh["participant_id"] = mh["participant_id"].astype(str)
    for c in ["age_at_diagnosis", "visit_month"]:
        if c in mh:
            mh[c] = pd.to_numeric(mh[c], errors="coerce")
    sortcol = "age_at_diagnosis" if "age_at_diagnosis" in mh else "participant_id"
    mh = mh.sort_values(sortcol, na_position="last").drop_duplicates("participant_id", keep="first")
    keep = [c for c in ["participant_id", "diagnosis", "age_at_diagnosis"] if c in mh.columns]
    meta = meta.merge(mh[keep], on="participant_id", how="left")
    if "age_at_diagnosis" in meta:
        meta = meta.rename(columns={"age_at_diagnosis": "AAO"})

    rin = pd.read_csv(paths.rin_csv, index_col=0)
    rin.index = [_norm_sample_id(x) for x in rin.index.astype(str)]
    rin_col = "RIN_Value" if "RIN_Value" in rin.columns else next(
        (c for c in rin.columns if "RIN" in c.upper()), None
    )
    if rin_col is None:
        raise ValueError("Could not identify RIN column.")
    rmap = pd.to_numeric(rin[rin_col], errors="coerce")
    meta["RIN"] = meta["sample_id"].map(rmap)

    meta["followup_time"] = meta["visit"].map(VISIT_TO_YEARS)
    bf = meta["cohort"].eq("BF") & meta["followup_time"].isna()
    meta.loc[bf, "followup_time"] = 0.0
    meta["age_at_baseline"] = pd.to_numeric(meta.get("age_at_baseline"), errors="coerce")
    meta["age"] = meta["age_at_baseline"] + meta["followup_time"]
    meta["sex_binary"] = meta.get("sex", pd.Series(index=meta.index, dtype=object)).map({"Male": 1, "Female": 0})
    no_pd = "No PD Nor Other Neurological Disorder"
    if "diagnosis" in meta:
        meta["diagnosis_binary"] = np.where(
            meta["diagnosis"].isna(),
            np.nan,
            (meta["diagnosis"].astype(str) != no_pd).astype(float),
        )
    else:
        meta["diagnosis_binary"] = np.nan
    meta["cohort_BF"] = (meta["cohort"].astype(str) == "BF").astype(float)
    return meta


def source_fam_rows(bfile_prefix: str, participant_ids):
    fam = pd.read_csv(str(bfile_prefix) + ".fam", sep=r"\s+", header=None, dtype=str)
    fam.columns = ["FID", "IID", "PAT", "MAT", "SEX", "PHENO"]
    ids = set(map(str, participant_ids))
    keep = fam["FID"].astype(str).isin(ids) | fam["IID"].astype(str).isin(ids)
    return fam.loc[keep].copy()


def run_plink_qc_group(
    paths: AmpPdPaths,
    group_label: str,
    participant_ids,
    mind=0.05,
    maf=0.05,
    geno=0.05,
    hwe=1e-6,
    pca_n=20,
    threads=16,
):
    outdir = paths.data_dir / "genotype" / group_label
    outdir.mkdir(parents=True, exist_ok=True)
    fam = source_fam_rows(paths.wgs_bfile, participant_ids)
    keepfile = outdir / f"{group_label}_participants.keep.txt"
    fam[["FID", "IID"]].to_csv(keepfile, sep="\t", header=False, index=False)
    prefix = outdir / f"amp_pd_{group_label}_qc"
    cmd = [
        paths.plink19,
        "--bfile", paths.wgs_bfile,
        "--keep", str(keepfile),
        "--mind", str(mind),
        "--maf", str(maf),
        "--geno", str(geno),
        "--hwe", str(hwe),
        "--make-bed",
        "--threads", str(int(threads)),
        "--out", str(prefix),
    ]
    subprocess.run(cmd, check=True)
    subprocess.run([
        paths.plink19,
        "--bfile", str(prefix),
        "--pca", str(int(pca_n)),
        "--threads", str(int(threads)),
        "--out", str(prefix) + ".pca",
    ], check=True)
    return prefix


def bfile_for_group(paths: AmpPdPaths, group_label: str):
    return paths.data_dir / "genotype" / group_label / f"amp_pd_{group_label}_qc"


def cohort_to_genotype_group(cohort: str) -> str:
    return "PP" if str(cohort) == "PP" else "REP"


def read_plink_eigenvec(qc_bfile: str, participant_ids, pc_indices, prefix: str):
    eig = pd.read_csv(str(qc_bfile) + ".pca.eigenvec", sep=r"\s+", header=None)
    n_available = eig.shape[1] - 2
    eig.columns = ["FID", "IID"] + [f"{prefix}{i}" for i in range(1, n_available + 1)]
    participants = set(map(str, participant_ids))
    eig["participant_id"] = np.where(
        eig["FID"].astype(str).isin(participants),
        eig["FID"].astype(str),
        eig["IID"].astype(str),
    )
    pc_indices = [int(i) for i in pc_indices]
    missing = [i for i in pc_indices if i < 1 or i > n_available]
    if missing:
        raise ValueError(f"Requested PLINK PCs are unavailable: {missing}; n_available={n_available}")
    cols = ["participant_id"] + [f"{prefix}{i}" for i in pc_indices]
    return eig[cols].drop_duplicates("participant_id")


def add_genotype_pcs_by_group(
    meta: pd.DataFrame,
    paths: AmpPdPaths,
    pc_mode: str,
    manual_gpcs: dict,
    max_pcs: int,
    tw_alpha: float,
    twstats_exe: str = "twstats",
    twtable: str = "twtable",
):
    meta = meta.copy()
    pc_meta = {}
    for group_label, cohorts in GENOTYPE_QC_GROUPS.items():
        ids = pd.unique(meta.loc[meta["cohort"].astype(str).isin(cohorts), "participant_id"].astype(str))
        qc_bfile = bfile_for_group(paths, group_label)
        n, sel = select_genotype_pc_count(
            str(qc_bfile),
            group_label,
            pc_mode=pc_mode,
            manual_counts=manual_gpcs,
            twstats=twstats_exe,
            twtable=twtable,
            alpha=tw_alpha,
            max_pcs=max_pcs,
        )
        pc_meta[group_label] = sel
        pc_indices = sel.get("selected_pc_indices", list(range(1, int(n) + 1)))
        pcs = read_plink_eigenvec(str(qc_bfile), ids, pc_indices, prefix=f"gPC_{group_label}_")
        meta = meta.merge(pcs, on="participant_id", how="left")
    return meta, pc_meta


def filter_meta_to_group_qc_genotypes(meta: pd.DataFrame, paths: AmpPdPaths):
    """Keep observations whose participants passed their group's genotype QC."""
    meta = meta.copy()
    keep = np.zeros(len(meta), dtype=bool)
    for group_label, cohorts in GENOTYPE_QC_GROUPS.items():
        bfile = bfile_for_group(paths, group_label)
        fam = pd.read_csv(str(bfile) + ".fam", sep=r"\s+", header=None, dtype=str)
        fam.columns = ["FID", "IID", "PAT", "MAT", "SEX", "PHENO"]
        valid = set(fam["FID"].astype(str)) | set(fam["IID"].astype(str))
        idx = meta["cohort"].astype(str).isin(cohorts) & meta["participant_id"].astype(str).isin(valid)
        keep |= idx.to_numpy()
    return meta.loc[keep].copy()


def save_preprocessed_expression_and_metadata(
    paths: AmpPdPaths,
    run_plink=True,
    age_min=30,
    age_max=90,
    qtl_mean_tpm: float = DEFAULT_QTL_MEAN_TPM,
    deconv_mean_tpm: float = DEFAULT_DECONV_MEAN_TPM,
    autosome_only: bool = True,
    pc_mode: str = "eigensoft",
    manual_gpcs=None,
    max_pcs: int = DEFAULT_MAX_PCS,
    tw_alpha: float = 0.05,
    twstats_exe: str = "twstats",
    twtable: str = "twtable",
    plink_threads: int = 16,
    force_plink: bool = False,
):
    for d in [
        paths.root_path, paths.data_dir, paths.qtl_dir, paths.logs_dir,
        paths.data_dir / "expression", paths.data_dir / "metadata",
        paths.data_dir / "annotation", paths.data_dir / "genotype",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    ann_all = load_gene_annotation(paths.gene_annotation_csv, autosome_only=False)
    rna_geneid = read_matrix_genes_tpm(paths.matrix_genes_tsv)

    # QTL universe: stricter TPM threshold, autosomal genes by default.
    tpm_qtl, ann_qtl = filter_tpm_to_gene_symbols(
        rna_geneid, ann_all, mean_tpm_threshold=qtl_mean_tpm, autosome_only=autosome_only
    )
    z = zscore_genes_after_log2(tpm_qtl)
    zqc, qcd = expression_sample_qc(z)
    qcd.to_csv(paths.data_dir / "metadata" / "expression_qc_details.csv", index=False)

    meta = build_sample_metadata(zqc.columns, paths)
    unmapped = meta[(meta.cohort.isin(["PP", "PD"])) & meta.followup_time.isna()].copy()
    unmapped[["cohort", "sample_id", "participant_id", "visit"]].drop_duplicates().to_csv(
        paths.data_dir / "metadata" / "unmapped_longitudinal_visits.csv", index=False
    )

    meta = meta[meta.age.notna()].copy()
    meta = meta[(meta.age >= age_min) & (meta.age <= age_max)].copy()
    kept = [s for s in meta.rna_column_id.astype(str) if s in zqc.columns]
    meta = meta[meta.rna_column_id.isin(kept)].copy()
    zqc = zqc.loc[:, meta.rna_column_id.astype(str)].copy()
    zqc.columns = meta.sample_id.astype(str).values

    # Run two separate genotype QC/PCA groups: PPMI discovery and PDBP+BioFIND replication.
    if run_plink:
        for group_label, cohorts in GENOTYPE_QC_GROUPS.items():
            bfile = bfile_for_group(paths, group_label)
            if force_plink or not Path(str(bfile) + ".fam").exists():
                ids = pd.unique(meta.loc[meta.cohort.astype(str).isin(cohorts), "participant_id"].astype(str))
                if len(ids) == 0:
                    raise RuntimeError(f"No participants for genotype QC group {group_label}.")
                run_plink_qc_group(
                    paths, group_label, ids,
                    pca_n=max_pcs,
                    threads=plink_threads,
                )
    else:
        # Reuse the already-QC'ed BED/BIM/FAM, but regenerate exactly 20 PCA
        # components. This guarantees that a prior v8.6 --pca 50 run does not
        # silently alter the prespecified historical PCA20/twstats procedure.
        for group_label in GENOTYPE_QC_GROUPS:
            bfile = bfile_for_group(paths, group_label)
            if not Path(str(bfile) + ".fam").exists():
                raise FileNotFoundError(f"Expected existing QC genotype set: {bfile}.fam")
            subprocess.run([
                paths.plink19,
                "--bfile", str(bfile),
                "--pca", str(int(max_pcs)),
                "--threads", str(int(plink_threads)),
                "--out", str(bfile) + ".pca",
            ], check=True)

    meta = filter_meta_to_group_qc_genotypes(meta, paths)
    zqc = zqc.loc[:, meta.sample_id.astype(str)].copy()

    manual_gpcs = manual_gpcs or DEFAULT_MANUAL_GPCS
    meta, gpc_meta = add_genotype_pcs_by_group(
        meta, paths, pc_mode=pc_mode, manual_gpcs=manual_gpcs,
        max_pcs=max_pcs, tw_alpha=tw_alpha,
        twstats_exe=twstats_exe,
        twtable=twtable,
    )

    # Expression PCs are intentionally NOT calculated here. v8 selects K
    # with the Buja-Eyuboglu algorithm in PCAForQTL::runBE() in a separate
    # reproducible step after the final genotype-QC sample sets are known.
    epc_meta = {"status": "pending_PCAForQTL_BE"}

    # Deconvolution universe: broad TPM matrix, but restricted to final retained samples.
    tpm_deconv, ann_deconv = filter_tpm_to_gene_symbols(
        rna_geneid, ann_all, mean_tpm_threshold=deconv_mean_tpm, autosome_only=False
    )
    raw_tpm = tpm_deconv.loc[:, [c for c in tpm_deconv.columns if c in set(meta.rna_column_id.astype(str))]].copy()
    rid_to_sid = dict(zip(meta.rna_column_id.astype(str), meta.sample_id.astype(str)))
    raw_tpm = raw_tpm.rename(columns=rid_to_sid)
    raw_tpm = raw_tpm.loc[:, meta.sample_id.astype(str)]
    raw_out = raw_tpm.T.copy()
    raw_out.index.name = "sample_id"
    raw_out = raw_out.reset_index()
    raw_out.to_parquet(
        paths.data_dir / "expression" / "amp_pd_expression_tpm_raw_qc.parquet",
        index=False, compression="zstd",
    )

    expr = zqc.T.copy()
    expr.index.name = "sample_id"
    expr = expr.reset_index()
    expr.to_parquet(paths.data_dir / "expression" / "amp_pd_expression_z.parquet",
                    index=False, compression="zstd")
    meta.to_parquet(paths.data_dir / "metadata" / "amp_pd_samples.parquet",
                    index=False, compression="zstd")
    meta.to_csv(paths.data_dir / "metadata" / "amp_pd_samples.csv", index=False)

    ann_qtl.to_parquet(paths.data_dir / "annotation" / "genes_used.parquet",
                       index=False, compression="zstd")
    ann_deconv.to_parquet(paths.data_dir / "annotation" / "genes_deconvolution_universe.parquet",
                          index=False, compression="zstd")
    location_cols = ["chr", "seqnames", "start", "end", "width", "strand", "gene_id", "gene_name", "gene_type"]
    ann_qtl[location_cols].to_parquet(
        paths.data_dir / "annotation" / "gene_locations.parquet",
        index=False, compression="zstd",
    )

    # Cohort-specific metadata plus combined replication metadata.
    for code, name in COHORT_CODE_TO_NAME.items():
        if code == "REP":
            m = meta[meta.cohort.astype(str).isin(["PD", "BF"])].copy()
        else:
            m = meta[meta.cohort.astype(str) == code].copy()
        m.to_parquet(paths.data_dir / "metadata" / f"{code}_{name}_samples.parquet",
                     index=False, compression="zstd")

    pc_report = {
        "genotype_pc_selection": gpc_meta,
        "expression_pcs": epc_meta,
    }
    (paths.data_dir / "metadata" / "pc_selection.json").write_text(json.dumps(pc_report, indent=2))

    summary = {
        "qtl_mean_tpm": float(qtl_mean_tpm),
        "deconv_mean_tpm": float(deconv_mean_tpm),
        "qtl_autosome_only": bool(autosome_only),
        "n_qtl_genes": int(zqc.shape[0]),
        "n_deconv_genes": int(raw_tpm.shape[0]),
        "n_samples": int(zqc.shape[1]),
        "n_participants": int(meta.participant_id.nunique()),
        "cohorts": {
            c: {
                "samples": int((meta.cohort == c).sum()),
                "participants": int(meta.loc[meta.cohort == c, "participant_id"].nunique()),
            }
            for c in sorted(meta.cohort.unique())
        },
        "genotype_qc_groups": {
            g: {
                "cohorts": cohorts,
                "bfile": str(bfile_for_group(paths, g)),
                "n_participants": int(
                    meta.loc[meta.cohort.astype(str).isin(cohorts), "participant_id"].nunique()
                ),
            }
            for g, cohorts in GENOTYPE_QC_GROUPS.items()
        },
        "pc_selection": pc_report,
        "expression_outliers_removed": int(qcd.outlier_any.sum()),
        "age_min": float(meta.age.min()),
        "age_max": float(meta.age.max()),
    }
    (paths.data_dir / "preprocess_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
