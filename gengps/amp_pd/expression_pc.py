from __future__ import annotations
from pathlib import Path
import json, subprocess, re
import numpy as np
import pandas as pd

from .config import (
    AmpPdPaths, LEGACY_TIMES_BY_COHORT, EXPRESSION_PC_BE_GROUPS,
    DEFAULT_BE_B, DEFAULT_BE_ALPHA, DEFAULT_BE_SEED, DEFAULT_BE_CORES,
)


def _complete_grid_rows(meta: pd.DataFrame, cohort: str, times):
    """Return one deterministic sample row per participant x required time."""
    m = meta[meta["cohort"].astype(str).eq(str(cohort))].copy()
    m["followup_time"] = pd.to_numeric(m["followup_time"], errors="coerce")
    m = m[m["followup_time"].isin([float(x) for x in times])].copy()
    m = m.sort_values(["participant_id", "followup_time", "sample_id"])
    m = m.drop_duplicates(["participant_id", "followup_time"], keep="first")
    req = set(map(float, times))
    good = []
    for pid, g in m.groupby("participant_id", sort=False):
        if req.issubset(set(g["followup_time"].astype(float))):
            good.append(str(pid))
    m = m[m["participant_id"].astype(str).isin(set(good))].copy()
    return m.sort_values(["participant_id", "followup_time", "sample_id"])


def select_group_metadata(meta: pd.DataFrame, group: str):
    if group == "PP_FULL":
        return meta[meta["cohort"].astype(str).eq("PP")].copy()
    if group == "PP_LEGACY":
        return _complete_grid_rows(meta, "PP", LEGACY_TIMES_BY_COHORT["PP"])
    if group == "PD_LEGACY":
        return _complete_grid_rows(meta, "PD", LEGACY_TIMES_BY_COHORT["PD"])
    if group == "PD_FULL":
        return meta[meta["cohort"].astype(str).eq("PD")].copy()
    if group == "REP_FULL":
        return meta[meta["cohort"].astype(str).isin(["PD", "BF"])].copy()
    raise KeyError(f"Unknown expression-PC group: {group}")


def _genotype_pc_columns(meta: pd.DataFrame, group: str):
    prefix = "gPC_PP_" if group.startswith("PP_") else "gPC_REP_"
    cols = [c for c in meta.columns if str(c).startswith(prefix)]
    def key(c):
        m = re.search(r"(\d+)$", str(c))
        return int(m.group(1)) if m else 0
    return sorted(cols, key=key)


def prepare_be_inputs(paths: AmpPdPaths, groups=None):
    """Prepare donor-balanced BE inputs plus sample-level projection inputs.

    Why donor-balanced?
    -------------------
    PCAForQTL::runBE() independently permutes the observations within each
    feature. Treating repeated longitudinal visits from the same participant as
    independent BE observations violates exchangeability and can inflate the
    number of selected PCs. v8.8 therefore gives every participant exactly one
    row for BE selection by averaging that participant's retained visits.

    K selection:
      participant x gene matrix -> PCAForQTL::runBE()

    PC covariates used in QTL:
      the donor-level PCA loadings are projected back onto every sample/visit,
      yielding one PC score vector per RNA sample.
    """
    groups = list(groups or EXPRESSION_PC_BE_GROUPS.keys())
    meta = pd.read_parquet(paths.data_dir / "metadata" / "amp_pd_samples.parquet")
    expr_path = paths.data_dir / "expression" / "amp_pd_expression_z.parquet"
    outdir = paths.data_dir / "expression_pcs_BE"
    outdir.mkdir(parents=True, exist_ok=True)

    expr = pd.read_parquet(expr_path)
    expr["sample_id"] = expr["sample_id"].astype(str)
    expr = expr.set_index("sample_id")
    gene_cols = expr.columns.astype(str).tolist()

    manifest = {}
    for group in groups:
        gm = select_group_metadata(meta, group)
        gm = gm.sort_values(
            ["cohort", "participant_id", "followup_time", "sample_id"]
        ).copy()
        gm["sample_id"] = gm["sample_id"].astype(str)
        gm["participant_id"] = gm["participant_id"].astype(str)

        gm = gm[gm["sample_id"].isin(expr.index)].copy()
        gm = gm.drop_duplicates("sample_id", keep="first")
        sids = gm["sample_id"].tolist()

        Xs = expr.loc[sids, gene_cols].to_numpy(dtype=np.float64, copy=True)
        if not np.isfinite(Xs).all():
            raise ValueError(f"{group}: non-finite sample-level expression values.")

        sample_df = pd.DataFrame(Xs, index=sids, columns=gene_cols)
        sample_df.insert(0, "participant_id", gm.set_index("sample_id").loc[sids, "participant_id"].values)

        # One equally weighted observation per participant for BE.
        donor_df = sample_df.groupby("participant_id", sort=True)[gene_cols].mean()
        donor_ids = donor_df.index.astype(str).tolist()
        Xd = donor_df.to_numpy(dtype=np.float64, copy=True)

        if not np.isfinite(Xd).all():
            raise ValueError(f"{group}: non-finite donor-mean expression values.")
        if len(donor_ids) < 10:
            raise ValueError(f"{group}: only {len(donor_ids)} participants available.")

        sample_npy = outdir / f"{group}.expression_sample.npy"
        donor_npy = outdir / f"{group}.expression_donor_mean.npy"
        np.save(sample_npy, Xs, allow_pickle=False)
        np.save(donor_npy, Xd, allow_pickle=False)

        sample_id_file = outdir / f"{group}.sample_ids.txt"
        donor_id_file = outdir / f"{group}.donor_ids.txt"
        sample_id_file.write_text("\n".join(sids) + "\n")
        donor_id_file.write_text("\n".join(donor_ids) + "\n")

        sample_map = gm[["sample_id", "participant_id", "cohort", "followup_time"]].copy()
        sample_map.to_csv(outdir / f"{group}.sample_to_donor.csv", index=False)

        gpc = _genotype_pc_columns(gm, group)
        known = pd.DataFrame({
            "sample_id": gm["sample_id"].astype(str),
            "participant_id": gm["participant_id"].astype(str),
            "Age": pd.to_numeric(gm["age_at_baseline"], errors="coerce"),
            "Sex": pd.to_numeric(gm["sex_binary"], errors="coerce"),
            "RIN": pd.to_numeric(gm["RIN"], errors="coerce"),
            "Diagnosis": pd.to_numeric(gm["diagnosis_binary"], errors="coerce"),
        })
        for i, c in enumerate(gpc, start=1):
            known[f"gPC{i}"] = pd.to_numeric(gm[c], errors="coerce").to_numpy()
        known_file = outdir / f"{group}.known_covariates.csv"
        known.to_csv(known_file, index=False)

        visit_counts = gm.groupby("participant_id").size()
        manifest[group] = {
            "n_samples": int(Xs.shape[0]),
            "n_donors_BE": int(Xd.shape[0]),
            "n_genes": int(Xs.shape[1]),
            "median_visits_per_donor": float(visit_counts.median()),
            "min_visits_per_donor": int(visit_counts.min()),
            "max_visits_per_donor": int(visit_counts.max()),
            "sample_expression_npy": str(sample_npy),
            "donor_mean_expression_npy": str(donor_npy),
            "sample_ids": str(sample_id_file),
            "donor_ids": str(donor_id_file),
            "known_covariates": str(known_file),
            "n_genotype_pcs": len(gpc),
            "BE_observation_unit": "participant",
            "BE_aggregation": "mean_across_retained_visits",
            "sample_PC_scoring": "projection_onto_donor_PCA_loadings",
        }

    (outdir / "BE_input_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def run_be_group(paths: AmpPdPaths, group: str, B=DEFAULT_BE_B,
                 alpha=DEFAULT_BE_ALPHA, seed=DEFAULT_BE_SEED,
                 cores=DEFAULT_BE_CORES, rscript="Rscript"):
    outdir = paths.data_dir / "expression_pcs_BE"
    runner = Path(__file__).resolve().parents[2] / "scripts_amp_pd" / "run_pca_for_qtl_BE.R"
    cmd = [
        rscript, str(runner),
        "--expr-donor-npy", str(outdir / f"{group}.expression_donor_mean.npy"),
        "--expr-sample-npy", str(outdir / f"{group}.expression_sample.npy"),
        "--donor-ids", str(outdir / f"{group}.donor_ids.txt"),
        "--sample-ids", str(outdir / f"{group}.sample_ids.txt"),
        "--known-covariates", str(outdir / f"{group}.known_covariates.csv"),
        "--group", group,
        "--out-dir", str(outdir / group),
        "--B", str(int(B)),
        "--alpha", str(float(alpha)),
        "--seed", str(int(seed)),
        "--cores", str(int(cores)),
    ]
    subprocess.run(cmd, check=True)


def merge_be_pcs(paths: AmpPdPaths, groups=None):
    groups = list(groups or EXPRESSION_PC_BE_GROUPS.keys())
    meta_path = paths.data_dir / "metadata" / "amp_pd_samples.parquet"
    meta = pd.read_parquet(meta_path)

    # Remove only the PC columns for groups being recomputed. This allows a
    # staged diagnostic run (legacy groups first, full groups later) without
    # deleting previously merged groups.
    prefixes = [f"ePC_BE_{g}_" for g in groups]
    old = [
        c for c in meta.columns
        if any(str(c).startswith(pref) for pref in prefixes)
    ]
    meta = meta.drop(columns=old, errors="ignore")

    report_path = paths.data_dir / "metadata" / "expression_pc_BE_selection.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
        except Exception:
            report = {}
    else:
        report = {}

    outdir = paths.data_dir / "expression_pcs_BE"
    for group in groups:
        gdir = outdir / group
        pcfile = gdir / "top_BE_PCs.csv"
        summaryfile = gdir / "BE_summary.json"
        if not pcfile.exists():
            raise FileNotFoundError(pcfile)

        pcs = pd.read_csv(pcfile)
        pcs["sample_id"] = pcs["sample_id"].astype(str)
        rename = {
            c: f"ePC_BE_{group}_{i}"
            for i, c in enumerate([c for c in pcs.columns if c != "sample_id"], start=1)
        }
        pcs = pcs.rename(columns=rename)
        meta = meta.merge(pcs, on="sample_id", how="left", validate="one_to_one")

        if summaryfile.exists():
            report[group] = json.loads(summaryfile.read_text())
        else:
            report[group] = {"n_pcs": len(rename)}

    meta.to_parquet(meta_path, index=False, compression="zstd")
    meta.to_csv(paths.data_dir / "metadata" / "amp_pd_samples.csv", index=False)
    report_path.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def prepare_run_merge(paths: AmpPdPaths, groups=None, B=DEFAULT_BE_B,
                      alpha=DEFAULT_BE_ALPHA, seed=DEFAULT_BE_SEED,
                      cores=DEFAULT_BE_CORES, rscript="Rscript"):
    manifest = prepare_be_inputs(paths, groups=groups)
    groups = list(groups or manifest.keys())
    for group in groups:
        run_be_group(
            paths, group, B=B, alpha=alpha, seed=seed,
            cores=cores, rscript=rscript
        )
    report = merge_be_pcs(paths, groups=groups)
    return {"input_manifest": manifest, "BE_selection": report}
