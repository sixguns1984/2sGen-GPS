from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import AmpPdPaths, LEGACY_TIMES_BY_COHORT, CELL_COMPOSITION_COVARIATES


STAGE1_CACHE_VERSION = "manuscript_stage1_v4_binary_exactcibx_rawpdbp"


def _stage1_base(paths: AmpPdPaths, cohort: str) -> Path:
    cohort = str(cohort)
    if cohort not in {"PP", "PD"}:
        raise ValueError(f"Stage-I manuscript data are defined only for PP/PD, got {cohort!r}")
    return paths.data_dir / "stage1_manuscript" / cohort


def stage1_cache_paths(paths: AmpPdPaths, cohort: str) -> dict[str, Path]:
    b = _stage1_base(paths, cohort)
    common = {
        "base": b,
        "cov": b / "covariates.parquet",
        "expr": b / "expression_log2_tpm001.parquet",
        "geno_meta": b / "genotype_meta.json",
        "geno_index": b / "genotype_snp_index.sqlite",
        "manifest": b / "manifest.json",
        "cis": paths.data_dir / "stage1_manuscript" / "PP" / "cis_pairs.parquet",
        "cis_genes": paths.data_dir / "stage1_manuscript" / "PP" / "cis_genes.txt",
        "pp_snp_extract": paths.data_dir / "stage1_manuscript" / "PP" / "ppmi_genotype_snps.txt",
    }
    if cohort == "PP":
        common["geno_dat"] = b / "genotype_int8.dat"
    else:
        prefix = b / "pdbp_raw_wgs_ppmi_snp_universe"
        common.update({
            "geno_prefix": prefix,
            "geno_bed": Path(str(prefix) + ".bed"),
            "geno_bim": Path(str(prefix) + ".bim"),
            "geno_fam": Path(str(prefix) + ".fam"),
        })
    return common


def require_stage1_cache(paths: AmpPdPaths, cohort: str, require_cis: bool = False) -> dict[str, Path]:
    cohort = str(cohort)
    p = stage1_cache_paths(paths, cohort)
    required = ["cov", "expr", "geno_meta", "geno_index", "manifest"]
    required += ["geno_dat"] if cohort == "PP" else ["geno_bed", "geno_bim", "geno_fam"]
    if cohort == "PP":
        required += ["pp_snp_extract"]
    if require_cis:
        required += ["cis", "cis_genes"]
    missing = [str(p[k]) for k in required if not p[k].exists()]
    if missing:
        raise FileNotFoundError(
            "Stage-I manuscript cache is incomplete. Run "
            "scripts_amp_pd/00_prepare_stage1_manuscript_inputs.py first. Missing:\n  "
            + "\n  ".join(missing)
        )
    info = json.loads(p["manifest"].read_text())
    if info.get("cache_version") != STAGE1_CACHE_VERSION:
        raise RuntimeError(
            f"Stage-I cache version mismatch for {cohort}: "
            f"{info.get('cache_version')!r} != {STAGE1_CACHE_VERSION!r}. "
            "Re-run 00_prepare_stage1_manuscript_inputs.py --force."
        )
    return p


def stage1_covariates(method: str, cohort: str, meta: pd.DataFrame, cell_adjust: bool = False, include_cell_composition: bool = True) -> list[str]:
    """Primary Stage-I covariates.

    The submitted MPTGA keeps the manuscript covariates exactly and now includes
    the six non-redundant broad CIBERSORTx composition covariates *in the primary
    model* (not as a separate sensitivity analysis): age, sex, RIN, diagnosis,
    8 genotype PCs, 33/37 expression PCs, and six standardized CLR composition
    terms. Comparator methods remain at their method-default adjustment so that
    the benchmark changes only the requested genotype coding/data source.
    """
    if method != "submitted_followup_mptga":
        return []

    n_epc = 33 if str(cohort) == "PP" else 37
    wanted = ["age", "sex_binary", "RIN", "diagnosis_binary"]
    wanted += [f"pca{i}" for i in range(1, 9)]
    wanted += [f"PC{i}" for i in range(1, n_epc + 1)]
    if include_cell_composition:
        wanted += list(CELL_COMPOSITION_COVARIATES)

    missing = [c for c in wanted if c not in meta.columns]
    if missing:
        raise KeyError(f"Stage-I primary covariates missing for {cohort}: {missing}")
    bad = []
    for c in wanted:
        v = pd.to_numeric(meta[c], errors="coerce")
        if v.isna().any() or v.nunique(dropna=True) <= 1:
            bad.append(c)
    if bad:
        raise ValueError(
            f"Stage-I primary covariates are missing/non-varying for {cohort}: {bad}. "
            "The prepared manuscript input is analysis-ready and must not be silently filtered."
        )
    return wanted


def load_stage1_observations(paths: AmpPdPaths, cohort: str, genes: Sequence[str]):
    p = require_stage1_cache(paths, cohort, require_cis=(str(cohort) == "PP"))
    meta = pd.read_parquet(p["cov"])
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta["participant_id"] = meta["participant_id"].astype(str)

    schema = set(pq.ParquetFile(p["expr"]).schema_arrow.names)
    genes = [str(g) for g in genes if str(g) in schema]
    expr = pd.read_parquet(p["expr"], columns=["sample_id"] + genes)
    expr["sample_id"] = expr["sample_id"].astype(str)
    expr = expr.set_index("sample_id")

    meta = meta.set_index("sample_id", drop=False)
    if meta.index.has_duplicates:
        raise RuntimeError(f"Duplicate Stage-I sample_id values in {cohort} covariates")
    if expr.index.has_duplicates:
        raise RuntimeError(f"Duplicate Stage-I sample_id values in {cohort} expression")
    missing_samples = meta.index.difference(expr.index)
    if len(missing_samples):
        raise RuntimeError(
            f"{cohort} Stage-I expression is missing {len(missing_samples)} covariate samples; "
            f"first={missing_samples[:5].tolist()}"
        )
    expr = expr.reindex(meta.index)
    meta.index = expr.index

    expected_times = list(LEGACY_TIMES_BY_COHORT[str(cohort)])
    counts = meta.groupby("participant_id", sort=False)["followup_time"].agg(list)
    bad = [pid for pid, vals in counts.items() if list(map(float, vals)) != expected_times]
    if bad:
        raise RuntimeError(
            f"{cohort} Stage-I strict visit order mismatch for {len(bad)} participants; first={bad[:5]}. "
            f"Expected exact order {expected_times}."
        )
    return meta, expr


class _SqliteVariantIndexMixin:
    def _lookup_rows(self, snpids: Sequence[str], fail_missing: bool = True) -> dict[str, int]:
        snpids = list(dict.fromkeys(map(str, snpids)))
        found: dict[str, int] = {}
        if not snpids:
            return found
        uri = f"file:{self.index_path}?mode=ro"
        con = sqlite3.connect(uri, uri=True)
        try:
            # SQLite commonly defaults to 999 bind parameters; keep headroom.
            step = 800
            for st in range(0, len(snpids), step):
                ss = snpids[st:st + step]
                q = "SELECT snpid, row_idx FROM snp_index WHERE snpid IN (%s)" % ",".join("?" * len(ss))
                for s, r in con.execute(q, ss):
                    found[str(s)] = int(r)
        finally:
            con.close()
        if fail_missing:
            missing = [s for s in snpids if s not in found]
            if missing:
                raise KeyError(
                    f"{len(missing)} SNPs absent from Stage-I {self.cohort} genotype cache; first={missing[:5]}"
                )
        return found

    def available_snps(self, snpids: Sequence[str]) -> set[str]:
        return set(self._lookup_rows(snpids, fail_missing=False))


class Stage1MemmapGenotypeSource(_SqliteVariantIndexMixin):
    """Fast random-access PPMI reader for the manuscript CSV-derived int8 cache."""

    def __init__(self, paths: AmpPdPaths, cohort: str = "PP"):
        self.cohort = str(cohort)
        if self.cohort != "PP":
            raise ValueError("Stage1MemmapGenotypeSource is the PPMI cache; use the PDBP raw-WGS source for PD")
        p = require_stage1_cache(paths, self.cohort, require_cis=False)
        self.dat_path = p["geno_dat"]
        self.index_path = p["geno_index"]
        info = json.loads(p["geno_meta"].read_text())
        self.participants = list(map(str, info["participants"]))
        self.n_snps = int(info["n_snps"])
        self.n_participants = int(info["n_participants"])
        self.pid_to_col = {pid: i for i, pid in enumerate(self.participants)}
        self._mm = np.memmap(
            self.dat_path, mode="r", dtype=np.int8,
            shape=(self.n_snps, self.n_participants), order="C",
        )

    def read(self, participant_ids, snpids, mean_impute=False, dtype=np.float64):
        participant_ids = list(map(str, participant_ids))
        snpids = list(map(str, snpids))
        missing_p = [p for p in participant_ids if p not in self.pid_to_col]
        if missing_p:
            raise KeyError(f"{len(missing_p)} PPMI participants absent from Stage-I cache; first={missing_p[:5]}")
        row_map = self._lookup_rows(snpids, fail_missing=True)
        rows = np.asarray([row_map[s] for s in snpids], dtype=np.int64)
        cols = np.asarray([self.pid_to_col[p] for p in participant_ids], dtype=np.int64)
        # SNP ids are requested in lexical/cis-map order, whereas the memmap rows
        # follow the original CSV order.  Reading millions of tiny random memmap
        # rows is much slower than a monotone access pattern, so read row-sorted
        # and restore the requested SNP order afterwards.
        order = np.argsort(rows, kind="mergesort")
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order), dtype=order.dtype)
        block = np.asarray(self._mm[rows[order], :][:, cols])
        X = np.asarray(block[inv_order, :], dtype=dtype).T
        bad = ~np.isin(X, [-1.0, 0.0, 1.0, 2.0])
        if bad.any():
            raise ValueError(f"PP Stage-I cache contains invalid hard calls: {np.unique(X[bad])[:10].tolist()}")
        return pd.DataFrame(X, index=pd.Index(participant_ids, name="participant_id"), columns=snpids)


class Stage1PDBPBedGenotypeSource(_SqliteVariantIndexMixin):
    """PDBP replication genotypes extracted from the original AMP-PD WGS BED.

    The preparation step makes a bit-packed PLINK BED containing the exact 346
    manuscript PDBP participants and the complete PPMI Stage-I SNP universe.  It
    is therefore fair to every benchmark method and much faster/smaller than a
    dense CSV. Missing BED calls are converted to the original legacy code -1;
    the binary encoder subsequently maps -1 -> (1,1).
    """

    def __init__(self, paths: AmpPdPaths, cohort: str = "PD"):
        self.cohort = str(cohort)
        if self.cohort != "PD":
            raise ValueError("Stage1PDBPBedGenotypeSource is defined only for PD")
        p = require_stage1_cache(paths, "PD", require_cis=False)
        self.index_path = p["geno_index"]
        info = json.loads(p["geno_meta"].read_text())
        self.participants = list(map(str, info["participants"]))
        self.pid_to_row = {pid: i for i, pid in enumerate(self.participants)}
        self.n_snps = int(info["n_snps"])
        try:
            from bed_reader import open_bed
        except ImportError as e:
            raise ImportError(
                "bed-reader is required for fast manuscript Stage-I PDBP replication. "
                "Install requirements.txt before running Stage-I."
            ) from e
        self._bed = open_bed(str(p["geno_bed"]))

    def read(self, participant_ids, snpids, mean_impute=False, dtype=np.float64):
        participant_ids = list(map(str, participant_ids))
        snpids = list(map(str, snpids))
        missing_p = [p for p in participant_ids if p not in self.pid_to_row]
        if missing_p:
            raise KeyError(f"{len(missing_p)} PDBP participants absent from raw-WGS Stage-I cache; first={missing_p[:5]}")
        row_map = self._lookup_rows(snpids, fail_missing=True)
        si = np.asarray([self.pid_to_row[p] for p in participant_ids], dtype=np.int64)
        vi = np.asarray([row_map[s] for s in snpids], dtype=np.int64)
        # bed-reader accepts float32/float64; for Stage-I hard calls we read
        # float32 and then compact to int8 after converting missing calls to -1.
        read_dtype = np.float32 if np.dtype(dtype) == np.dtype(np.int8) else dtype
        order = np.argsort(vi, kind="mergesort")
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(len(order), dtype=order.dtype)
        Xs = np.asarray(self._bed.read(index=np.s_[si, vi[order]], dtype=read_dtype), dtype=read_dtype)
        X = Xs[:, inv_order]
        # Preserve the original BinaryEncoder missing category rather than mean-imputing.
        X[np.isnan(X)] = -1.0
        bad = ~np.isin(X, [-1.0, 0.0, 1.0, 2.0])
        if bad.any():
            raise ValueError(f"PD raw-WGS cache contains invalid hard calls: {np.unique(X[bad])[:10].tolist()}")
        X = np.asarray(X, dtype=dtype)
        return pd.DataFrame(X, index=pd.Index(participant_ids, name="participant_id"), columns=snpids)


# One source object per scheduler worker process avoids reopening a multi-million-
# variant BED/BIM/index for every gene chunk.
_GENO_SOURCE_CACHE: dict[tuple[str, str], object] = {}


def get_stage1_genotype_source(paths: AmpPdPaths, cohort: str):
    key = (str(Path(paths.root).resolve()), str(cohort))
    src = _GENO_SOURCE_CACHE.get(key)
    if src is None:
        src = Stage1MemmapGenotypeSource(paths, "PP") if str(cohort) == "PP" else Stage1PDBPBedGenotypeSource(paths, "PD")
        _GENO_SOURCE_CACHE[key] = src
    return src


def stage1_gene_list(paths: AmpPdPaths) -> list[str]:
    p = require_stage1_cache(paths, "PP", require_cis=True)
    return [x.strip() for x in p["cis_genes"].read_text().splitlines() if x.strip()]


def stage1_expression_genes(paths: AmpPdPaths, cohort: str) -> set[str]:
    p = require_stage1_cache(paths, cohort)
    return set(pq.ParquetFile(p["expr"]).schema_arrow.names) - {"sample_id"}
