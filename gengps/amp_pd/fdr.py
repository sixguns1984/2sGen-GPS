from __future__ import annotations
from pathlib import Path
import json
import os
import shutil
import duckdb


def _qident(x: str) -> str:
    return '"' + str(x).replace('"', '""') + '"'


def _table_exists(con, name: str) -> bool:
    q = "SELECT count(*) FROM information_schema.tables WHERE table_name = ?"
    return bool(con.execute(q, [name]).fetchone()[0])


def _view_exists(con, name: str) -> bool:
    q = "SELECT count(*) FROM information_schema.views WHERE table_name = ?"
    return bool(con.execute(q, [name]).fetchone()[0])


def _set_duckdb_runtime(
    con,
    temp_dir: Path,
    threads: int | None = None,
    memory_limit: str | None = None,
):
    """Configure DuckDB for large out-of-core BH-FDR jobs.

    The QTL filesystem may have limited free space, so every spill/sort temporary
    file is forced to ``temp_dir``.  A deliberately conservative memory limit
    makes DuckDB spill *before* it reaches the process/system ceiling.  Fewer
    threads also reduce per-thread sort/hash buffers substantially.

    Both settings can be overridden without editing code:
      GEN_GPS_DUCKDB_THREADS=4
      GEN_GPS_DUCKDB_MEMORY=20GB
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    if threads is None:
        threads = int(os.environ.get("GEN_GPS_DUCKDB_THREADS", "2"))
    if memory_limit is None:
        memory_limit = os.environ.get("GEN_GPS_DUCKDB_MEMORY", "16GB")

    escaped = str(temp_dir).replace("'", "''")
    con.execute(f"SET temp_directory = '{escaped}'")
    con.execute(f"PRAGMA threads={max(1, int(threads))}")
    con.execute(f"SET memory_limit='{memory_limit}'")
    try:
        con.execute("SET preserve_insertion_order=false")
    except Exception:
        pass


def _parquet_row_count(con, glob: str) -> int:
    return int(con.execute(
        f"SELECT count(*) FROM read_parquet('{glob}', union_by_name=true)"
    ).fetchone()[0])


def _sql_path(path: Path | str) -> str:
    return str(path).replace("'", "''")


def finalize_method_fdr(
    chunk_dir: str,
    out_dir: str,
    temp_root: str = "/public/labdata/luojunfeng/tmp",
    work_label: str | None = None,
    reuse_legacy_work_db: bool = True,
    duckdb_threads: int | None = None,
    duckdb_memory_limit: str | None = None,
    fdr_strategy: str = "stagewise_hierarchical",
    alpha: float = 0.05,
):
    """Out-of-core global BH correction over every cis pair for one method/cohort.

    v8.9 large-scan / resume behavior
    ---------------------------------
    * QTL chunk Parquets are the durable compute outputs and are never deleted.
    * All DuckDB sort/spill and temporary BH mapping Parquets are written below
      ``temp_root`` (default /public/labdata/luojunfeng/tmp).
    * If ``final/fdr_work.duckdb`` from an interrupted legacy run contains a
      complete ``base`` table, it is reused, so FDR can resume without GPU scans.
    * BH ranking is performed on only (Variant_ID, Ensembl_ID, p-value), rather
      than carrying the full wide QTL row through the global sort.  This reduces
      RAM and spill volume dramatically for ~10^8-pair scans.
    * DuckDB defaults to 2 threads / 16GB for finalization, forcing early external
      spilling instead of failing near a ~30GB process ceiling.  Override with
      GEN_GPS_DUCKDB_THREADS and GEN_GPS_DUCKDB_MEMORY if desired.
    * Temporary BH mapping files are removed only after all final outputs have
      been written successfully.
    """
    if fdr_strategy not in {"global_global", "stagewise_hierarchical"}:
        raise ValueError(
            "fdr_strategy must be one of {'global_global','stagewise_hierarchical'}"
        )
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be between 0 and 1")

    chunk_dir = Path(chunk_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    glob = str(chunk_dir / "chunk_*.parquet")

    if not list(chunk_dir.glob("chunk_*.parquet")):
        raise FileNotFoundError(f"No QTL chunk parquet files found in {chunk_dir}")

    temp_root = Path(temp_root)
    label = work_label or (out_dir.parent.name + "__" + out_dir.name)
    safe_label = "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
    temp_dir = temp_root / "2sGenGPS_duckdb" / safe_label
    temp_dir.mkdir(parents=True, exist_ok=True)

    temporal_bh = temp_dir / "temporal_bh.parquet"
    dynamic_bh = temp_dir / "dynamic_bh.parquet"

    # Clear stale DuckDB spill files left by an interrupted finalization for this
    # exact cohort/method.  The directory is job-specific, so this never touches
    # other methods or the user's temp root.
    for stale in temp_dir.glob("duckdb_temp_storage_*"):
        try:
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
            else:
                stale.unlink()
        except OSError:
            pass

    # Remove incomplete BH mapping files from a prior failed finalization. They
    # are temporary and cheap relative to the durable QTL chunks.
    for p in [temporal_bh, dynamic_bh]:
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    legacy_db = out_dir / "fdr_work.duckdb"
    use_legacy = bool(reuse_legacy_work_db and legacy_db.exists())
    con = duckdb.connect(str(legacy_db)) if use_legacy else duckdb.connect()
    _set_duckdb_runtime(
        con,
        temp_dir,
        threads=duckdb_threads,
        memory_limit=duckdb_memory_limit,
    )

    # Count chunk rows and verify an interrupted persistent base table before reuse.
    n_chunks = _parquet_row_count(con, glob)
    if n_chunks == 0:
        con.close()
        return 0

    source_name = "base"
    reused_base = False
    if use_legacy and _table_exists(con, "base"):
        n_base = int(con.execute("SELECT count(*) FROM base").fetchone()[0])
        if n_base == n_chunks:
            reused_base = True
        else:
            con.execute("DROP TABLE base")

    if not reused_base:
        if _view_exists(con, source_name):
            con.execute("DROP VIEW base")
        con.execute(
            f"CREATE TEMP VIEW base AS SELECT * FROM read_parquet('{_sql_path(glob)}', union_by_name=true)"
        )

    n = n_chunks
    cols = [r[1] for r in con.execute("PRAGMA table_info('base')").fetchall()]
    if "Pvalue" not in cols:
        con.close()
        raise KeyError("Pvalue column is missing from QTL chunks/base")
    for key in ["Variant_ID", "Ensembl_ID"]:
        if key not in cols:
            con.close()
            raise KeyError(f"{key} column is missing from QTL chunks/base")

    # Temporal/global BH.  Crucially, the global sort carries only pair IDs and
    # p-values, not the full ~30-column QTL record.
    con.execute(f"""
        COPY (
          WITH r AS (
            SELECT Variant_ID, Ensembl_ID, Pvalue,
                   row_number() OVER (ORDER BY Pvalue ASC) AS _rank,
                   count(*) OVER () AS _m
            FROM base
            WHERE isfinite(Pvalue)
          ), q AS (
            SELECT Variant_ID, Ensembl_ID, _rank,
                   least(1.0, Pvalue * _m / _rank) AS _qraw
            FROM r
          )
          SELECT Variant_ID, Ensembl_ID,
                 least(1.0, min(_qraw) OVER (
                   ORDER BY _rank DESC
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                 )) AS FDR
          FROM q
        ) TO '{_sql_path(temporal_bh)}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    has_dynamic = "Interaction_Pvalue" in cols

    # Stage-wise / hierarchical confirmation constants.  The implementation follows
    # the two-stage screening-confirmation construction used by Heller/stageR:
    #   stage I: BH on the global screening (overall temporal genetic-effect) tests;
    #   stage II: for the R screened pairs, test the single confirmation hypothesis
    #             (genotype-by-time interaction) at alpha_II = alpha * R / G.
    # With one confirmation hypothesis per screened SNP-gene pair, the corresponding
    # stage-wise adjusted confirmation p-value is p_int * G / R (capped at 1).
    # This is deliberately *not* naive BH inside the selected tQTL subset.
    n_screen_tests = int(con.execute(
        "SELECT count(*) FROM base WHERE isfinite(Pvalue)"
    ).fetchone()[0])
    n_screen_rejected = int(con.execute(
        f"SELECT count(*) FROM read_parquet('{_sql_path(temporal_bh)}') "
        f"WHERE isfinite(FDR) AND FDR < {alpha}"
    ).fetchone()[0])
    confirmation_alpha = (
        alpha * n_screen_rejected / n_screen_tests if n_screen_tests > 0 else 0.0
    )
    stagewise_multiplier = (
        n_screen_tests / n_screen_rejected if n_screen_rejected > 0 else float("inf")
    )

    if has_dynamic and fdr_strategy == "global_global":
        # Legacy conservative strategy: independent global interaction BH across
        # the complete discovery cis-pair universe.
        con.execute(f"""
            COPY (
              WITH r AS (
                SELECT Variant_ID, Ensembl_ID, Interaction_Pvalue,
                       row_number() OVER (ORDER BY Interaction_Pvalue ASC) AS _rank,
                       count(*) OVER () AS _m
                FROM base
                WHERE isfinite(Interaction_Pvalue)
              ), q AS (
                SELECT Variant_ID, Ensembl_ID, _rank,
                       least(1.0, Interaction_Pvalue * _m / _rank) AS _qraw
                FROM r
              )
              SELECT Variant_ID, Ensembl_ID,
                     least(1.0, min(_qraw) OVER (
                       ORDER BY _rank DESC
                       ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                     )) AS Interaction_FDR
              FROM q
            ) TO '{_sql_path(dynamic_bh)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

    all_results = out_dir / "all_results.parquet"
    sig_tqtl = out_dir / "significant_tqtl.parquet"
    sig_union = out_dir / "significant_union.parquet"  # backward-compatible alias
    sig_temporal = out_dir / "significant_temporal.parquet"  # backward-compatible alias
    sig_dynamic = out_dir / "significant_dynamic.parquet"
    sig_additive = out_dir / "significant_additive.parquet"
    sig_intermediate = out_dir / "significant_intermediate.parquet"

    # A failed COPY can leave a partial parquet. Remove only finalization outputs;
    # durable QTL chunks are never touched.
    targets = [
        all_results, sig_tqtl, sig_union, sig_temporal,
        sig_dynamic, sig_additive, sig_intermediate,
    ]
    for target in targets:
        try:
            target.unlink()
        except FileNotFoundError:
            pass

    if has_dynamic and fdr_strategy == "global_global":
        con.execute(f"""
            COPY (
              SELECT b.*, t.FDR, d.Interaction_FDR
              FROM base b
              LEFT JOIN read_parquet('{_sql_path(temporal_bh)}') t
                USING (Variant_ID, Ensembl_ID)
              LEFT JOIN read_parquet('{_sql_path(dynamic_bh)}') d
                USING (Variant_ID, Ensembl_ID)
            ) TO '{_sql_path(all_results)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    elif has_dynamic and fdr_strategy == "stagewise_hierarchical":
        # One confirmation hypothesis per screened pair.  Keep the historical
        # Interaction_FDR column name for downstream compatibility, but also write
        # the explicit Interaction_FDR_stagewise alias so the provenance is clear.
        if n_screen_rejected > 0:
            expr = (
                f"CASE WHEN isfinite(t.FDR) AND t.FDR < {alpha:.17g} "
                f"AND isfinite(b.Interaction_Pvalue) "
                f"THEN least(1.0, b.Interaction_Pvalue * {float(stagewise_multiplier):.17g}) "
                f"ELSE NULL END"
            )
        else:
            expr = (
                f"CASE WHEN isfinite(t.FDR) AND t.FDR < {alpha:.17g} "
                f"AND isfinite(b.Interaction_Pvalue) THEN 1.0 ELSE NULL END"
            )
        con.execute(f"""
            COPY (
              SELECT b.*, t.FDR,
                     {expr} AS Interaction_FDR,
                     {expr} AS Interaction_FDR_stagewise
              FROM base b
              LEFT JOIN read_parquet('{_sql_path(temporal_bh)}') t
                USING (Variant_ID, Ensembl_ID)
            ) TO '{_sql_path(all_results)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    else:
        con.execute(f"""
            COPY (
              SELECT b.*, t.FDR
              FROM base b
              LEFT JOIN read_parquet('{_sql_path(temporal_bh)}') t
                USING (Variant_ID, Ensembl_ID)
            ) TO '{_sql_path(all_results)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

    # Final discovery definitions:
    #   tQTL             := global temporal BH-FDR < alpha
    #   dynamic tQTL     := tQTL AND strategy-specific confirmed interaction
    #   additive tQTL    := tQTL AND interaction P > 0.05
    #   intermediate     := tQTL AND interaction P <= 0.05 but not confirmed dynamic
    # `significant_union` is retained only as a legacy filename and is now exactly
    # the same set as `significant_tqtl` because dynamic/additive are tQTL subclasses.
    con.execute(f"""
        COPY (
          SELECT * FROM read_parquet('{_sql_path(all_results)}')
          WHERE isfinite(FDR) AND FDR < {alpha}
        ) TO '{_sql_path(sig_tqtl)}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    for alias in [sig_temporal, sig_union]:
        con.execute(f"""
            COPY (
              SELECT * FROM read_parquet('{_sql_path(sig_tqtl)}')
            ) TO '{_sql_path(alias)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

    if has_dynamic:
        con.execute(f"""
            COPY (
              SELECT * FROM read_parquet('{_sql_path(sig_tqtl)}')
              WHERE isfinite(Interaction_FDR) AND Interaction_FDR < {alpha}
            ) TO '{_sql_path(sig_dynamic)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        con.execute(f"""
            COPY (
              SELECT * FROM read_parquet('{_sql_path(sig_tqtl)}')
              WHERE isfinite(Interaction_Pvalue) AND Interaction_Pvalue > 0.05
            ) TO '{_sql_path(sig_additive)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        con.execute(f"""
            COPY (
              SELECT * FROM read_parquet('{_sql_path(sig_tqtl)}')
              WHERE isfinite(Interaction_Pvalue)
                AND Interaction_Pvalue <= 0.05
                AND (NOT isfinite(Interaction_FDR) OR Interaction_FDR >= {alpha})
            ) TO '{_sql_path(sig_intermediate)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
        """)

    # Persist explicit counts so downstream code and reviewers can audit the exact
    # hierarchy without inferring it from filenames.
    n_tqtl = int(con.execute(
        f"SELECT count(*) FROM read_parquet('{_sql_path(sig_tqtl)}')"
    ).fetchone()[0])
    n_dynamic = n_additive = n_intermediate = None
    if has_dynamic:
        n_dynamic = int(con.execute(
            f"SELECT count(*) FROM read_parquet('{_sql_path(sig_dynamic)}')"
        ).fetchone()[0])
        n_additive = int(con.execute(
            f"SELECT count(*) FROM read_parquet('{_sql_path(sig_additive)}')"
        ).fetchone()[0])
        n_intermediate = int(con.execute(
            f"SELECT count(*) FROM read_parquet('{_sql_path(sig_intermediate)}')"
        ).fetchone()[0])
    info = {
        "n_pairs": int(n),
        "chunk_dir": str(chunk_dir),
        "out_dir": str(out_dir),
        "duckdb_temp_dir": str(temp_dir),
        "duckdb_threads": int(os.environ.get("GEN_GPS_DUCKDB_THREADS", "2")) if duckdb_threads is None else int(duckdb_threads),
        "duckdb_memory_limit": os.environ.get("GEN_GPS_DUCKDB_MEMORY", "16GB") if duckdb_memory_limit is None else str(duckdb_memory_limit),
        "fdr_strategy": str(fdr_strategy),
        "alpha": float(alpha),
        "bh_strategy": (
            "temporal_global_and_interaction_global_v4"
            if fdr_strategy == "global_global"
            else "stagewise_hierarchical_screen_confirm_v1"
        ),
        "tqtl_definition": f"global temporal BH-FDR<{alpha:g}",
        "dynamic_definition": (
            f"global temporal BH-FDR<{alpha:g} AND global interaction BH-FDR<{alpha:g}"
            if fdr_strategy == "global_global"
            else (
                f"global temporal BH-FDR<{alpha:g} AND stage-wise interaction confirmation "
                f"P<alpha*R/G={confirmation_alpha:.12g}"
            )
        ),
        "hierarchical_screening_n_tests_G": int(n_screen_tests),
        "hierarchical_screening_n_rejected_R": int(n_screen_rejected),
        "hierarchical_confirmation_alpha": float(confirmation_alpha),
        "hierarchical_interaction_adjustment_multiplier_G_over_R": (
            None if n_screen_rejected == 0 else float(stagewise_multiplier)
        ),
        "hierarchical_note": (
            None if fdr_strategy == "global_global" else
            "Heller/stageR-style screening-confirmation procedure; one interaction confirmation "
            "hypothesis per screened SNP-gene pair. Validate empirical error control under LD by simulation."
        ),
        "additive_definition": "temporal tQTL AND Interaction_Pvalue>0.05",
        "n_tqtl": int(n_tqtl),
        "n_dynamic": None if n_dynamic is None else int(n_dynamic),
        "n_additive": None if n_additive is None else int(n_additive),
        "n_intermediate": None if n_intermediate is None else int(n_intermediate),
        "legacy_work_db_reused": bool(reused_base),
        "legacy_work_db_path": str(legacy_db) if use_legacy else None,
    }
    (out_dir / "fdr_finalize_summary.json").write_text(json.dumps(info, indent=2))

    con.close()

    # Clean temporary BH mapping Parquets and stale spill artifacts only after all
    # final outputs were written successfully.
    for p in [temporal_bh, dynamic_bh]:
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    if legacy_db.exists():
        try:
            legacy_db.unlink()
        except OSError:
            pass
    legacy_tmp = Path(str(legacy_db) + ".tmp")
    if legacy_tmp.exists():
        shutil.rmtree(legacy_tmp, ignore_errors=True)
    try:
        if temp_dir.exists() and not any(temp_dir.iterdir()):
            temp_dir.rmdir()
    except OSError:
        pass
    return n
