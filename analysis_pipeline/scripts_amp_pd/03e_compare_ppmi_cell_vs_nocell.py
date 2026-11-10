#!/usr/bin/env python
"""Compare the primary +cell and no-cell Stage-I submitted MPTGA PPMI scans.

The comparison reports discovery counts under the current finalizer and also
recomputes a *global* BH cutoff for Interaction_Pvalue over all 77M pairs for
each model, so the effect of cell-composition adjustment is not confounded by
changing the interaction-BH family.
"""
from pathlib import Path
import argparse
import json
import math
import os

import duckdb
import pandas as pd


def q(s: str) -> str:
    return str(s).replace("'", "''")


def scalar(con, sql):
    return con.execute(sql).fetchone()[0]


def summarize_one(con, path: Path, label: str):
    pp = q(path)
    n_all = int(scalar(con, f"SELECT count(*) FROM read_parquet('{pp}')"))
    n_valid = int(scalar(con, f"""
        SELECT count(*) FROM read_parquet('{pp}')
        WHERE Interaction_Pvalue IS NOT NULL
          AND Interaction_Pvalue >= 0 AND Interaction_Pvalue <= 1
    """))
    n_tqtl = int(scalar(con, f"SELECT count(*) FROM read_parquet('{pp}') WHERE FDR < 0.05"))

    cutoff = scalar(con, f"""
        WITH ranked AS (
            SELECT Interaction_Pvalue AS p,
                   row_number() OVER (ORDER BY Interaction_Pvalue) AS r
            FROM read_parquet('{pp}')
            WHERE Interaction_Pvalue IS NOT NULL
              AND Interaction_Pvalue >= 0 AND Interaction_Pvalue <= 1
        )
        SELECT max(p) FROM ranked
        WHERE p <= 0.05 * r / {n_valid}
    """)

    if cutoff is None:
        n_global_interaction = 0
        n_global_dynamic = 0
    else:
        n_global_interaction = int(scalar(con, f"""
            SELECT count(*) FROM read_parquet('{pp}')
            WHERE Interaction_Pvalue <= {float(cutoff)}
        """))
        n_global_dynamic = int(scalar(con, f"""
            SELECT count(*) FROM read_parquet('{pp}')
            WHERE FDR < 0.05 AND Interaction_Pvalue <= {float(cutoff)}
        """))

    thresholds = {}
    for x in [1e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]:
        thresholds[f"{x:g}"] = int(scalar(con, f"""
            SELECT count(*) FROM read_parquet('{pp}')
            WHERE FDR < 0.05 AND Interaction_Pvalue < {x}
        """))

    n_covars = scalar(con, f"SELECT min(n_covariates_used) FROM read_parquet('{pp}')")
    max_covars = scalar(con, f"SELECT max(n_covariates_used) FROM read_parquet('{pp}')")
    cell_modes = [r[0] for r in con.execute(
        f"SELECT DISTINCT stage1_cell_mode FROM read_parquet('{pp}') ORDER BY 1"
    ).fetchall()] if "stage1_cell_mode" in [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{pp}')").fetchall()] else []

    return {
        "label": label,
        "path": str(path),
        "n_pairs": n_all,
        "n_valid_interaction_p": n_valid,
        "n_tqtl_global_temporal_fdr_lt_0_05": n_tqtl,
        "global_interaction_bh_raw_p_cutoff": None if cutoff is None else float(cutoff),
        "n_global_interaction_bh_hits": n_global_interaction,
        "n_global_dynamic_tqtl": n_global_dynamic,
        "n_covariates_min": None if n_covars is None else int(n_covars),
        "n_covariates_max": None if max_covars is None else int(max_covars),
        "stage1_cell_mode": cell_modes,
        "within_tqtl_raw_interaction_p_counts": thresholds,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt")
    p.add_argument("--tmp-dir", default="/public/labdata/luojunfeng/tmp")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--memory", default="16GB")
    a = p.parse_args()

    root = Path(a.root)
    primary = root / "qtl/PP/submitted_followup_mptga/final/all_results.parquet"
    nocell = root / "qtl/PP/submitted_followup_mptga__nocell_audit/final/all_results.parquet"
    for f in [primary, nocell]:
        if not f.exists():
            raise FileNotFoundError(f)

    tmp = Path(a.tmp_dir) / "2sGenGPS_duckdb" / "PP__cell_vs_nocell_audit"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"PRAGMA threads={int(a.threads)}")
    con.execute(f"SET memory_limit='{q(a.memory)}'")
    con.execute(f"SET temp_directory='{q(tmp)}'")
    con.execute("SET preserve_insertion_order=false")
    try:
        out = {
            "primary_cell_adjusted": summarize_one(con, primary, "primary_cell_adjusted"),
            "nocell_audit": summarize_one(con, nocell, "nocell_audit"),
        }
    finally:
        con.close()

    p1 = out["primary_cell_adjusted"]
    p0 = out["nocell_audit"]
    out["contrast"] = {
        "delta_n_tqtl_cell_minus_nocell": p1["n_tqtl_global_temporal_fdr_lt_0_05"] - p0["n_tqtl_global_temporal_fdr_lt_0_05"],
        "delta_n_global_dynamic_cell_minus_nocell": p1["n_global_dynamic_tqtl"] - p0["n_global_dynamic_tqtl"],
        "ratio_global_dynamic_cell_over_nocell": None if p0["n_global_dynamic_tqtl"] == 0 else p1["n_global_dynamic_tqtl"] / p0["n_global_dynamic_tqtl"],
    }

    outdir = root / "qtl/PP/cell_vs_nocell_audit"
    outdir.mkdir(parents=True, exist_ok=True)
    j = outdir / "comparison_summary.json"
    j.write_text(json.dumps(out, indent=2))

    rows = []
    for key in ["primary_cell_adjusted", "nocell_audit"]:
        z = out[key]
        row = {
            "analysis": key,
            "n_pairs": z["n_pairs"],
            "n_tqtl": z["n_tqtl_global_temporal_fdr_lt_0_05"],
            "global_interaction_bh_raw_p_cutoff": z["global_interaction_bh_raw_p_cutoff"],
            "n_global_dynamic_tqtl": z["n_global_dynamic_tqtl"],
            "n_covariates_min": z["n_covariates_min"],
            "n_covariates_max": z["n_covariates_max"],
        }
        for thr, n in z["within_tqtl_raw_interaction_p_counts"].items():
            row[f"n_tqtl_interaction_p_lt_{thr}"] = n
        rows.append(row)
    pd.DataFrame(rows).to_csv(outdir / "comparison_summary.csv", index=False)

    print(json.dumps(out, indent=2))
    print(f"\nWrote: {j}")
    print(f"Wrote: {outdir / 'comparison_summary.csv'}")


if __name__ == "__main__":
    main()
