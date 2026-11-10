#!/usr/bin/env python
from pathlib import Path
import argparse, json, os, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gengps.amp_pd.config import AmpPdPaths
from gengps.amp_pd.fdr import finalize_method_fdr


def main():
    ap = argparse.ArgumentParser(
        description="Resume/finalize global BH-FDR from existing QTL chunk Parquets without rerunning QTL scans."
    )
    ap.add_argument("--root", default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt")
    ap.add_argument("--cohort", default="PP", help="PP, PD, REP, etc.")
    ap.add_argument("--method", required=True, help="Result directory label, e.g. submitted_followup_mptga")
    ap.add_argument("--tmp-dir", default=os.environ.get("TMPDIR", "/public/labdata/luojunfeng/tmp"))
    ap.add_argument("--duckdb-threads", type=int, default=None,
                    help="DuckDB FDR threads; default 2 or GEN_GPS_DUCKDB_THREADS")
    ap.add_argument("--duckdb-memory", default=None,
                    help="DuckDB memory limit, e.g. 16GB; default GEN_GPS_DUCKDB_MEMORY or 16GB")
    ap.add_argument("--fdr-strategy", choices=["stagewise_hierarchical", "global_global"],
                    default="stagewise_hierarchical")
    ap.add_argument("--fdr-alpha", type=float, default=0.05)
    a = ap.parse_args()

    paths = AmpPdPaths(root=a.root)
    base = paths.qtl_dir / a.cohort / a.method
    chunk_dir = base / "chunks"
    final_dir = base / "final"
    if not chunk_dir.exists():
        raise FileNotFoundError(chunk_dir)

    n = finalize_method_fdr(
        str(chunk_dir), str(final_dir), temp_root=a.tmp_dir,
        work_label=f"{a.cohort}__{a.method}", reuse_legacy_work_db=True,
        duckdb_threads=a.duckdb_threads, duckdb_memory_limit=a.duckdb_memory,
        fdr_strategy=a.fdr_strategy, alpha=a.fdr_alpha,
    )
    print(json.dumps({
        "cohort": a.cohort,
        "method": a.method,
        "n_pairs_finalized": int(n),
        "final_dir": str(final_dir),
        "tmp_dir": a.tmp_dir,
        "fdr_strategy": a.fdr_strategy,
        "fdr_alpha": a.fdr_alpha,
    }, indent=2))


if __name__ == "__main__":
    main()
