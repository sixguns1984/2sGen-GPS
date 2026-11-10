#!/usr/bin/env python3
"""Prepare manuscript-compatible Stage-I PPMI/PDBP inputs for V8.9.

Stage-I deliberately uses the already-QCed manuscript covariate/expression files.
PPMI genotype hard calls are converted from the original dense CSV to an int8
random-access cache.  PDBP genotype is *not* read from the old 748,400-SNP
replication subset: instead, PLINK extracts the complete PPMI Stage-I SNP universe
from the original AMP-PD WGS BED for the exact 346 manuscript PDBP participants.
This makes replication fair for every benchmark method.

The six non-redundant broad CIBERSORTx composition terms are merged by sample_id
and become primary submitted-MPTGA covariates.  They are not a separate
sensitivity analysis.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gengps.amp_pd.config import (
    AmpPdPaths,
    LEGACY_TIMES_BY_COHORT,
    CELL_COMPOSITION_COVARIATES,
)
from gengps.amp_pd.stage1_data import STAGE1_CACHE_VERSION, stage1_cache_paths


DEFAULTS = {
    "PP": {
        "cov": "/public2/data/luojunfeng/2sTGCS_binary/exposure/ppmi/teqtl.cov.csv",
        "geno": "/public2/data/luojunfeng/2sTGCS_binary/exposure/ppmi/cis.genotype.matrix.csv",
        "expr": "/public2/data/luojunfeng/2sTGCS_binary/exposure/ppmi/tpm.5t.adfpass.csv",
        "cis": "/public2/data/luojunfeng/2sTGCS_binary/exposure/ppmi/cis_snpgene_pair.csv",
        "n_participants": 293,
        "n_epc": 33,
    },
    "PD": {
        "cov": "/public2/data/luojunfeng/2sTGCS_binary/exposure/pdbp/teqtl.cov.csv",
        "expr": "/public2/data/luojunfeng/2sTGCS_binary/exposure/pdbp/tpm.5t.rep.csv",
        "n_participants": 346,
        "n_epc": 37,
    },
}


def log(msg):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg, flush=True)


def _norm_geno_pid(x: str, cohort: str | None = None) -> str:
    x = str(x).strip()
    if x.startswith("PP."):
        return "PP-" + x[3:]
    if x.startswith("PD."):
        return "PD-" + x[3:]
    if x.startswith("PP-") or x.startswith("PD-"):
        return x
    # AMP-PD FAM files may use the cohort-specific subject identifier without
    # the leading study code used by the RNA files.
    if cohort == "PP" and x.startswith("PP"):
        return "PP-" + x
    if cohort == "PD" and x.startswith("PD"):
        return "PD-" + x
    return x.replace(".", "-", 1) if "." in x else x


def _merge_cell_covariates(cov: pd.DataFrame, cohort: str, cell_cov_path: Path) -> pd.DataFrame:
    if not cell_cov_path.exists():
        raise FileNotFoundError(
            f"Stage-I exact cell-composition covariates not found: {cell_cov_path}. "
            "Run 00a_export_stage1_cibersortx_lm22_inputs.py, 00b_run_stage1_cibersortx_lm22.sh, "
            "and 00c_import_stage1_cibersortx_lm22.py before preparing Stage-I inputs. "
            "The current V8.9 metadata is intentionally not used because its sample universe differs from the manuscript 293/346 x 5 sample sets."
        )
    cols = ["sample_id"] + list(CELL_COMPOSITION_COVARIATES)
    pf = pq.ParquetFile(cell_cov_path)
    missing_cols = [c for c in cols if c not in pf.schema_arrow.names]
    if missing_cols:
        raise KeyError(
            f"{cell_cov_path} lacks required primary Stage-I cell-composition covariates {missing_cols}. "
            "Run scripts_amp_pd/00c_import_stage1_cibersortx_lm22.py first."
        )
    cm = pd.read_parquet(cell_cov_path, columns=cols)
    cm["sample_id"] = cm["sample_id"].astype(str)
    if cm["sample_id"].duplicated().any():
        dup = cm.loc[cm.sample_id.duplicated(False), "sample_id"].head().tolist()
        raise RuntimeError(f"Duplicate sample_id in cell-composition metadata; first={dup}")
    out = cov.merge(cm, on="sample_id", how="left", validate="one_to_one", sort=False)
    bad = out[CELL_COMPOSITION_COVARIATES].isna().any(axis=1)
    if bad.any():
        ex = out.loc[bad, "sample_id"].head(10).tolist()
        raise RuntimeError(
            f"{cohort}: {int(bad.sum())} manuscript samples lack Stage-I exact CIBERSORTx covariates; first={ex}. "
            "Stage-I uses cell composition in the primary model and will not silently drop or impute samples. "
            "Re-export/rerun CIBERSORTx on the exact manuscript sample universe."
        )
    return out


def _prepare_cov(cohort: str, src: Path, out: Path, cell_cov_path: Path) -> pd.DataFrame:
    log(f"Reading {cohort} manuscript covariates: {src}")
    cov = pd.read_csv(src, index_col=0)
    req = ["cohort", "participant_id", "sample_id", "visit", "time", "RIN", "sex", "age", "diagnosis"]
    req += [f"pca{i}" for i in range(1, 9)]
    req += [f"PC{i}" for i in range(1, DEFAULTS[cohort]["n_epc"] + 1)]
    missing = [c for c in req if c not in cov.columns]
    if missing:
        raise KeyError(f"{cohort} covariate CSV missing required columns: {missing}")

    cov = cov.copy()
    cov["participant_id"] = cov["participant_id"].astype(str)
    cov["sample_id"] = cov["sample_id"].astype(str)
    cov["time"] = pd.to_numeric(cov["time"], errors="raise").astype(float)
    cov["followup_time"] = cov["time"].astype(float)
    cov["visit"] = cov["time"].astype(float)

    sex_map = {"Female": 0.0, "Male": 1.0, "F": 0.0, "M": 1.0, "0": 0.0, "1": 1.0}
    cov["sex_binary"] = cov["sex"].astype(str).map(sex_map)
    if cov["sex_binary"].isna().any():
        bad = cov.loc[cov.sex_binary.isna(), "sex"].astype(str).unique().tolist()
        raise ValueError(f"Unmapped {cohort} sex labels: {bad}")

    def diag_code(v):
        s = str(v).strip().lower()
        if s in {"idiopathic pd", "parkinson's disease", "parkinsons disease", "pd"}:
            return 1.0
        if "no pd" in s or "no parkinson" in s or s in {"control", "healthy control", "hc"}:
            return 0.0
        return np.nan

    cov["diagnosis_binary"] = cov["diagnosis"].map(diag_code)
    if cov["diagnosis_binary"].isna().any():
        bad = cov.loc[cov.diagnosis_binary.isna(), "diagnosis"].astype(str).unique().tolist()
        raise ValueError(f"Unmapped {cohort} diagnosis labels: {bad}")

    n_expected = DEFAULTS[cohort]["n_participants"]
    if cov.participant_id.nunique() != n_expected:
        raise RuntimeError(f"{cohort} participant count mismatch: {cov.participant_id.nunique()} != {n_expected}")
    expected_times = list(LEGACY_TIMES_BY_COHORT[cohort])
    groups = cov.groupby("participant_id", sort=False)["time"].agg(list)
    bad_order = [p for p, t in groups.items() if list(map(float, t)) != expected_times]
    if bad_order:
        raise RuntimeError(
            f"{cohort} strict visit order mismatch for {len(bad_order)} participants; first={bad_order[:5]}"
        )
    if len(cov) != n_expected * len(expected_times):
        raise RuntimeError(f"{cohort} observation count is not exactly N x 5: {len(cov)}")

    # Primary cell-composition adjustment comes from a Stage-I-specific CIBERSORTx run
    # on the exact manuscript sample universe. Never borrow the current V8.9 metadata
    # because that universe may omit manuscript samples.
    cov = _merge_cell_covariates(cov, cohort, cell_cov_path)

    numeric = ["age", "RIN", "sex_binary", "diagnosis_binary"]
    numeric += [f"pca{i}" for i in range(1, 9)]
    numeric += [f"PC{i}" for i in range(1, DEFAULTS[cohort]["n_epc"] + 1)]
    numeric += list(CELL_COMPOSITION_COVARIATES)
    for c in numeric:
        cov[c] = pd.to_numeric(cov[c], errors="raise").astype(float)
        if not np.isfinite(cov[c]).all():
            raise RuntimeError(f"{cohort} non-finite values in required covariate {c}")

    out.parent.mkdir(parents=True, exist_ok=True)
    cov.to_parquet(out, index=False, compression="zstd")
    log(
        f"Wrote {out} rows={len(cov):,}; participants={n_expected}; "
        f"primary cell covariates={len(CELL_COMPOSITION_COVARIATES)}"
    )
    return cov


def _prepare_expression(cohort: str, src: Path, cov: pd.DataFrame, out: Path):
    log(f"Reading {cohort} expression: {src}")
    x = pd.read_csv(src, index_col=0)
    wanted = cov["sample_id"].astype(str).tolist()
    if cohort == "PP":
        x.index = x.index.astype(str); x.columns = x.columns.astype(str)
        missing = [s for s in wanted if s not in x.columns]
        if missing:
            raise RuntimeError(f"PP expression missing {len(missing)} covariate samples; first={missing[:5]}")
        x = x.loc[:, wanted].T
    else:
        x.index = x.index.astype(str); x.columns = x.columns.astype(str)
        missing = [s for s in wanted if s not in x.index]
        if missing:
            raise RuntimeError(f"PD expression missing {len(missing)} covariate samples; first={missing[:5]}")
        x = x.reindex(wanted)
    if x.index.tolist() != wanted:
        raise RuntimeError(f"{cohort} expression/covariate sample order failed exact alignment")
    vals = x.to_numpy(dtype=np.float64, copy=False)
    if np.nanmin(vals) < 0 or not np.isfinite(vals).all():
        raise ValueError(f"{cohort} manuscript TPM matrix contains invalid values")
    vals = np.log2(vals + 0.01)
    y = pd.DataFrame(vals, index=x.index, columns=x.columns)
    y.insert(0, "sample_id", y.index.astype(str))
    y.to_parquet(out, index=False, compression="zstd")
    log(f"Wrote {out} shape={y.shape}; transform=log2(TPM+0.01); dtype=float64")


def _init_snp_db(db: Path):
    if db.exists(): db.unlink()
    con = sqlite3.connect(db)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("CREATE TABLE snp_index (snpid TEXT PRIMARY KEY, row_idx INTEGER NOT NULL) WITHOUT ROWID")
    return con


def _prepare_pp_genotype(src: Path, cov: pd.DataFrame, dat: Path, meta: Path, db: Path,
                         snp_extract: Path, chunk_rows: int = 50000):
    log(f"Preparing PPMI random-access genotype cache from {src}")
    header = pd.read_csv(src, nrows=0).columns.tolist()
    csv_pids = [_norm_geno_pid(c, "PP") for c in header[1:]]
    cov_pids = cov.drop_duplicates("participant_id", keep="first")["participant_id"].astype(str).tolist()
    if len(csv_pids) != len(set(csv_pids)):
        raise RuntimeError("PP genotype header has duplicate normalized participant IDs")
    if set(csv_pids) != set(cov_pids):
        missing = sorted(set(cov_pids) - set(csv_pids))[:10]
        extra = sorted(set(csv_pids) - set(cov_pids))[:10]
        raise RuntimeError(f"PP genotype participant mismatch; missing={missing}, extra={extra}")
    pos = {p: i for i, p in enumerate(csv_pids)}
    order = [pos[p] for p in cov_pids]

    for p in [dat, meta, db, snp_extract]:
        if p.exists(): p.unlink()
    dat.parent.mkdir(parents=True, exist_ok=True)
    con = _init_snp_db(db)
    n = 0
    allowed = {-1, 0, 1, 2}
    with open(dat, "wb") as fout, open(snp_extract, "w") as fsnp:
        for ich, z in enumerate(pd.read_csv(src, index_col=0, chunksize=int(chunk_rows))):
            snps = z.index.astype(str).tolist()
            if len(snps) != len(set(snps)):
                raise RuntimeError(f"PP duplicate SNP IDs within genotype chunk {ich}")
            a = z.to_numpy(copy=False)[:, order]
            a = np.asarray(a, dtype=float)
            if not np.isfinite(a).all() or not np.all(np.equal(a, np.round(a))):
                raise RuntimeError(f"PP genotype matrix has non-finite/non-integer hard calls in chunk {ich}")
            a = np.asarray(np.round(a), dtype=np.int8)
            seen = set(np.unique(a).tolist())
            if not seen.issubset(allowed):
                raise RuntimeError(f"PP genotype matrix contains values outside -1/0/1/2: {sorted(seen-allowed)}")
            a.tofile(fout)
            fsnp.write("\n".join(snps) + "\n")
            con.executemany("INSERT INTO snp_index(snpid,row_idx) VALUES (?,?)",
                            ((s, n + i) for i, s in enumerate(snps)))
            n += len(snps)
            if ich % 10 == 0:
                log(f"PP genotype rows cached: {n:,}")
    con.commit(); con.close()
    info = {
        "cache_version": STAGE1_CACHE_VERSION,
        "source": str(src), "n_snps": int(n), "n_participants": int(len(cov_pids)),
        "participants": cov_pids, "dtype": "int8", "layout": "SNP_x_participant_C_order",
        "hard_call_values": [-1, 0, 1, 2], "missing_code": -1,
        "binary_mapping": {"0": "00", "1": "01", "2": "10", "-1": "11"},
    }
    meta.write_text(json.dumps(info, indent=2))
    exp = n * len(cov_pids)
    if dat.stat().st_size != exp:
        raise RuntimeError(f"PP genotype cache byte-size mismatch: {dat.stat().st_size} != {exp}")
    log(f"Wrote PP genotype cache: SNPs={n:,}, participants={len(cov_pids)}")
    return n


def _resolve_keep_rows(raw_fam: Path, cov_pids: list[str], cohort: str) -> pd.DataFrame:
    fam = pd.read_csv(raw_fam, sep=r"\s+", header=None,
                      names=["FID", "IID", "PAT", "MAT", "SEX", "PHENO"], dtype=str)
    candidates: dict[str, list[int]] = {}
    for i, row in fam.iterrows():
        for val in [row.FID, row.IID]:
            key = _norm_geno_pid(val, cohort)
            candidates.setdefault(key, []).append(i)
    rows = []
    for pid in cov_pids:
        hits = sorted(set(candidates.get(pid, [])))
        if len(hits) != 1:
            raise RuntimeError(f"Raw WGS FAM mapping for {pid} returned {len(hits)} rows: {hits[:5]}")
        rows.append(hits[0])
    if len(set(rows)) != len(rows):
        raise RuntimeError("Raw WGS FAM mapping is not one-to-one for manuscript participants")
    return fam.iloc[rows].copy()


def _build_bim_sqlite(bim: Path, db: Path):
    con = _init_snp_db(db)
    n = 0
    batch = []
    with open(bim, "r") as fh:
        for i, line in enumerate(fh):
            parts = line.rstrip("\n").split()
            if len(parts) < 2:
                raise RuntimeError(f"Malformed BIM line {i+1}: {line[:100]!r}")
            batch.append((str(parts[1]), i))
            if len(batch) >= 100000:
                con.executemany("INSERT INTO snp_index(snpid,row_idx) VALUES (?,?)", batch)
                batch.clear()
            n = i + 1
    if batch:
        con.executemany("INSERT INTO snp_index(snpid,row_idx) VALUES (?,?)", batch)
    con.commit(); con.close()
    return n


def _prepare_pd_raw_wgs(paths: AmpPdPaths, cov: pd.DataFrame, pp_snp_extract: Path,
                        out_prefix: Path, meta: Path, db: Path, plink19: str,
                        threads: int, memory_mb: int):
    raw_prefix = Path(paths.wgs_bfile)
    for ext in [".bed", ".bim", ".fam"]:
        if not Path(str(raw_prefix) + ext).exists():
            raise FileNotFoundError(str(raw_prefix) + ext)
    cov_pids = cov.drop_duplicates("participant_id", keep="first")["participant_id"].astype(str).tolist()
    keep = out_prefix.parent / "pdbp_346_keep.txt"
    keep_rows = _resolve_keep_rows(Path(str(raw_prefix) + ".fam"), cov_pids, "PD")
    keep_rows[["FID", "IID"]].to_csv(keep, sep="\t", header=False, index=False)

    for ext in [".bed", ".bim", ".fam", ".log", ".nosex"]:
        q = Path(str(out_prefix) + ext)
        if q.exists(): q.unlink()
    if db.exists(): db.unlink()
    if meta.exists(): meta.unlink()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(plink19), "--bfile", str(raw_prefix), "--keep", str(keep),
        "--extract", str(pp_snp_extract), "--make-bed", "--allow-no-sex",
        "--threads", str(max(1, int(threads))), "--memory", str(max(1000, int(memory_mb))),
        "--out", str(out_prefix),
    ]
    log("Extracting full PDBP Stage-I SNP universe from original AMP-PD WGS with PLINK")
    log("COMMAND: " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    fam_out = pd.read_csv(str(out_prefix) + ".fam", sep=r"\s+", header=None,
                          names=["FID", "IID", "PAT", "MAT", "SEX", "PHENO"], dtype=str)
    participants = []
    target = set(cov_pids)
    for _, row in fam_out.iterrows():
        opts = [_norm_geno_pid(row.FID, "PD"), _norm_geno_pid(row.IID, "PD")]
        hits = [x for x in opts if x in target]
        if not hits:
            raise RuntimeError(f"Cannot map extracted PDBP FAM row FID={row.FID}, IID={row.IID}")
        participants.append(hits[0])
    if len(participants) != len(cov_pids) or set(participants) != set(cov_pids):
        raise RuntimeError(
            f"Extracted raw-WGS PDBP participant mismatch: got {len(participants)}, expected {len(cov_pids)}"
        )
    if len(set(participants)) != len(participants):
        raise RuntimeError("Duplicate participant mapping in extracted PDBP BED")

    n_snps = _build_bim_sqlite(Path(str(out_prefix) + ".bim"), db)
    info = {
        "cache_version": STAGE1_CACHE_VERSION,
        "source_raw_wgs_bfile": str(raw_prefix),
        "snp_extract_source": str(pp_snp_extract),
        "n_snps": int(n_snps), "n_participants": int(len(participants)),
        "participants": participants,
        "storage": "PLINK1 BED bit-packed; random-access via bed-reader",
        "hard_call_values": [-1, 0, 1, 2], "missing_code": -1,
        "binary_mapping": {"0": "00", "1": "01", "2": "10", "-1": "11"},
    }
    meta.write_text(json.dumps(info, indent=2))
    log(
        f"Wrote raw-WGS PDBP replication genotype cache: SNPs={n_snps:,}; "
        f"participants={len(participants)}; BED={out_prefix}.bed"
    )
    return n_snps


def _prepare_cis(src: Path, out: Path, genes_out: Path, chunk_rows: int = 1_000_000):
    log(f"Preparing PPMI cis pair Parquet from {src}")
    if out.exists(): out.unlink()
    if genes_out.exists(): genes_out.unlink()
    writer = None; genes = []; seen_genes = set(); n = 0
    try:
        for ich, z in enumerate(pd.read_csv(src, usecols=["snpid", "geneid"], chunksize=int(chunk_rows))):
            z["snpid"] = z["snpid"].astype(str); z["geneid"] = z["geneid"].astype(str)
            if z[["snpid", "geneid"]].isna().any().any():
                raise RuntimeError("Missing snpid/geneid in cis pair source")
            for g in pd.unique(z["geneid"]):
                if g not in seen_genes:
                    seen_genes.add(g); genes.append(g)
            tab = pa.Table.from_pandas(z[["snpid", "geneid"]], preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out, tab.schema, compression="zstd")
            writer.write_table(tab, row_group_size=250_000)
            n += len(z)
            if ich % 10 == 0: log(f"cis pairs cached: {n:,}")
    finally:
        if writer is not None: writer.close()
    genes_out.write_text("\n".join(genes) + "\n")
    log(f"Wrote {out}: pairs={n:,}; genes={len(genes):,}")
    return n, len(genes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--genotype-chunk-rows", type=int, default=50000)
    ap.add_argument("--cis-chunk-rows", type=int, default=1000000)
    ap.add_argument("--plink-threads", type=int, default=16)
    ap.add_argument("--plink-memory-mb", type=int, default=32000)
    ap.add_argument("--raw-wgs-bfile", default=None,
                    help="Original full AMP-PD WGS PLINK prefix. Default: AmpPdPaths.wgs_bfile")
    ap.add_argument("--cell-covariates", default=None,
                    help="Stage-I exact CIBERSORTx covariate parquet. Default: data/stage1_manuscript/cell_deconvolution/cibersortx_lm22/stage1_cell_covariates.parquet")
    ap.add_argument("--cell-metadata", default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--pp-cov", default=DEFAULTS["PP"]["cov"])
    ap.add_argument("--pp-geno", default=DEFAULTS["PP"]["geno"])
    ap.add_argument("--pp-expr", default=DEFAULTS["PP"]["expr"])
    ap.add_argument("--pp-cis", default=DEFAULTS["PP"]["cis"])
    ap.add_argument("--pd-cov", default=DEFAULTS["PD"]["cov"])
    ap.add_argument("--pd-expr", default=DEFAULTS["PD"]["expr"])
    a = ap.parse_args()

    paths = AmpPdPaths(root=a.root)
    if a.raw_wgs_bfile:
        paths.wgs_bfile = a.raw_wgs_bfile
    if a.cell_covariates and a.cell_metadata:
        raise ValueError("Use only --cell-covariates; --cell-metadata is a deprecated alias")
    cell_cov = Path(a.cell_covariates or a.cell_metadata) if (a.cell_covariates or a.cell_metadata) else (
        paths.data_dir / "stage1_manuscript" / "cell_deconvolution" / "cibersortx_lm22" / "stage1_cell_covariates.parquet"
    )

    src = {
        "PP": {"cov": Path(a.pp_cov), "geno": Path(a.pp_geno), "expr": Path(a.pp_expr)},
        "PD": {"cov": Path(a.pd_cov), "expr": Path(a.pd_expr)},
    }
    cis_src = Path(a.pp_cis)
    for d in src.values():
        for p in d.values():
            if not p.exists(): raise FileNotFoundError(p)
    if not cis_src.exists(): raise FileNotFoundError(cis_src)

    # The cache version is intentionally bumped whenever genotype/covariate semantics change.
    pp = stage1_cache_paths(paths, "PP"); pdp = stage1_cache_paths(paths, "PD")
    for p in [pp["base"], pdp["base"]]: p.mkdir(parents=True, exist_ok=True)

    # PPMI: exact manuscript covariate/expression files + original dense genotype CSV.
    rebuild_pp = a.force or not all(x.exists() for x in [pp["cov"], pp["expr"], pp["geno_dat"], pp["geno_meta"], pp["geno_index"], pp["pp_snp_extract"], pp["manifest"]])
    if not rebuild_pp:
        try:
            rebuild_pp = json.loads(pp["manifest"].read_text()).get("cache_version") != STAGE1_CACHE_VERSION
        except Exception:
            rebuild_pp = True
    if rebuild_pp:
        cov_pp = _prepare_cov("PP", src["PP"]["cov"], pp["cov"], cell_cov)
        _prepare_expression("PP", src["PP"]["expr"], cov_pp, pp["expr"])
        npp = _prepare_pp_genotype(src["PP"]["geno"], cov_pp, pp["geno_dat"], pp["geno_meta"], pp["geno_index"], pp["pp_snp_extract"], a.genotype_chunk_rows)
        info_pp = {
            "cache_version": STAGE1_CACHE_VERSION, "cohort": "PP",
            "n_participants": 293, "n_observations": 1465,
            "times": list(LEGACY_TIMES_BY_COHORT["PP"]), "n_genotype_pcs": 8,
            "n_expression_pcs": 33, "n_primary_cell_covariates": len(CELL_COMPOSITION_COVARIATES),
            "primary_cell_covariates": list(CELL_COMPOSITION_COVARIATES),
            "expression_transform": "log2(TPM+0.01)", "genotype_coding": "legacy_binary",
            "genotype_source": str(src["PP"]["geno"]), "n_genotype_snps": int(npp),
            "covariate_source": str(src["PP"]["cov"]), "expression_source": str(src["PP"]["expr"]),
            "cell_covariate_source": str(cell_cov),
        }
        pp["manifest"].write_text(json.dumps(info_pp, indent=2))
    else:
        log("PP Stage-I cache already prepared; use --force to rebuild")
        cov_pp = pd.read_parquet(pp["cov"])
        info_pp = json.loads(pp["manifest"].read_text())

    # PDBP: exact manuscript covariate/expression files, but genotype is freshly
    # extracted from the original full WGS for the complete PPMI SNP universe.
    rebuild_pd = a.force or not all(x.exists() for x in [pdp["cov"], pdp["expr"], pdp["geno_bed"], pdp["geno_bim"], pdp["geno_fam"], pdp["geno_meta"], pdp["geno_index"], pdp["manifest"]])
    if not rebuild_pd:
        try:
            rebuild_pd = json.loads(pdp["manifest"].read_text()).get("cache_version") != STAGE1_CACHE_VERSION
        except Exception:
            rebuild_pd = True
    if rebuild_pd:
        cov_pd = _prepare_cov("PD", src["PD"]["cov"], pdp["cov"], cell_cov)
        _prepare_expression("PD", src["PD"]["expr"], cov_pd, pdp["expr"])
        npd = _prepare_pd_raw_wgs(paths, cov_pd, pp["pp_snp_extract"], pdp["geno_prefix"], pdp["geno_meta"], pdp["geno_index"], paths.plink19, a.plink_threads, a.plink_memory_mb)
        info_pd = {
            "cache_version": STAGE1_CACHE_VERSION, "cohort": "PD",
            "n_participants": 346, "n_observations": 1730,
            "times": list(LEGACY_TIMES_BY_COHORT["PD"]), "n_genotype_pcs": 8,
            "n_expression_pcs": 37, "n_primary_cell_covariates": len(CELL_COMPOSITION_COVARIATES),
            "primary_cell_covariates": list(CELL_COMPOSITION_COVARIATES),
            "expression_transform": "log2(TPM+0.01)", "genotype_coding": "legacy_binary",
            "genotype_source": str(paths.wgs_bfile), "genotype_extraction": "raw_WGS_x_exact346_x_full_PPMI_SNP_universe",
            "n_genotype_snps": int(npd), "covariate_source": str(src["PD"]["cov"]),
            "expression_source": str(src["PD"]["expr"]), "cell_covariate_source": str(cell_cov),
        }
        pdp["manifest"].write_text(json.dumps(info_pd, indent=2))
    else:
        log("PD Stage-I raw-WGS cache already prepared; use --force to rebuild")
        info_pd = json.loads(pdp["manifest"].read_text())

    if (not pp["cis"].exists()) or (not pp["cis_genes"].exists()) or a.force:
        n_pairs, n_genes = _prepare_cis(cis_src, pp["cis"], pp["cis_genes"], a.cis_chunk_rows)
    else:
        n_pairs = int(pq.ParquetFile(pp["cis"]).metadata.num_rows)
        n_genes = len([x for x in pp["cis_genes"].read_text().splitlines() if x.strip()])
        log("PP Stage-I cis cache already prepared; use --force to rebuild")

    expr_genes = set(pq.ParquetFile(pp["expr"]).schema_arrow.names) - {"sample_id"}
    cis_genes = [x for x in pp["cis_genes"].read_text().splitlines() if x.strip()]
    missing_cis_genes = [g for g in cis_genes if g not in expr_genes]
    if missing_cis_genes:
        raise RuntimeError(f"PP cis map contains genes absent from manuscript expression; first={missing_cis_genes[:10]}")

    summary = {
        "cache_version": STAGE1_CACHE_VERSION, "PP": info_pp, "PD": info_pd,
        "cis_source": str(cis_src), "n_cis_pairs": int(n_pairs), "n_cis_genes": int(n_genes),
        "benchmark_genotype_mode": "binary: 0=00,1=01,2=10,-1=11",
        "pdbp_old_748400_snp_subset_used": False,
        "cell_composition_is_primary_covariate": True,
    }
    out = paths.data_dir / "stage1_manuscript" / "prepare_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    log(f"DONE. Summary: {out}")


if __name__ == "__main__":
    main()
