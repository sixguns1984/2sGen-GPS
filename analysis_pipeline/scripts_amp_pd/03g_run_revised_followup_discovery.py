#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gengps.amp_pd.config import AmpPdPaths, REVISED_FOLLOWUP_METHODS
from gengps.amp_pd.scheduler import run_method_partitioned


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt")
    p.add_argument("--methods", default=",".join(REVISED_FOLLOWUP_METHODS))
    p.add_argument("--gpus", default="0,1,2")
    p.add_argument("--cpu-workers", type=int, default=12)
    p.add_argument("--genes-per-chunk", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--gpu-batch-map", default="0:4096,1:2048,2:2048")
    p.add_argument("--min-gpu-batch", type=int, default=256)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--tmp-dir", default=os.environ.get("TMPDIR", "/public2/data/luojunfeng/2sGenGPS_revised_chatgpt/tmp"))
    p.add_argument("--rho-points", type=int, default=100)
    p.add_argument("--ctar-reference-interval", type=float, default=1.0,
                   help="Time unit (years) for rho_ref. Corr(dt)=rho_ref**(dt/reference_interval).")
    p.add_argument("--ctar-rho-max", type=float, default=0.99)
    p.add_argument("--legacy-genotype-mode", choices=["binary"], default="binary",
                   help="Revised Stage-I is fixed to the historical two-channel binary hard-call coding.")
    p.add_argument("--fdr-strategy", choices=["stagewise_hierarchical", "global_global"],
                   default="stagewise_hierarchical")
    p.add_argument("--fdr-alpha", type=float, default=0.05)
    p.add_argument("--gene-list", default="", help="Optional newline-delimited gene list for a smoke/pilot run.")
    p.add_argument("--max-genes", type=int, default=0, help="If >0 and no --gene-list, use the first N distinct cis genes for a pilot.")
    a = p.parse_args()

    paths = AmpPdPaths(root=a.root)
    gpus = [int(x) for x in a.gpus.split(",") if x]

    cis_path = None
    if a.gene_list or a.max_genes > 0:
        import duckdb
        base_cis = paths.data_dir / "cis" / "cis_1000kb_PP.parquet"
        pilot_dir = paths.data_dir / "stage1_ctar" / "pilot"
        pilot_dir.mkdir(parents=True, exist_ok=True)
        if a.gene_list:
            genes = [x.strip() for x in Path(a.gene_list).read_text().splitlines() if x.strip()]
        else:
            con = duckdb.connect()
            try:
                q = str(base_cis).replace("'", "''")
                genes = [str(x[0]) for x in con.execute(
                    f"SELECT DISTINCT CAST(geneid AS VARCHAR) FROM read_parquet('{q}') LIMIT {int(a.max_genes)}"
                ).fetchall()]
            finally:
                con.close()
        if not genes:
            raise RuntimeError("Pilot gene set is empty")
        cis_path = pilot_dir / "cis_pilot.parquet"
        con = duckdb.connect()
        try:
            con.register("genes_df", __import__("pandas").DataFrame({"geneid": genes}))
            q = str(base_cis).replace("'", "''")
            outq = str(cis_path).replace("'", "''")
            con.execute(f"COPY (SELECT c.* FROM read_parquet('{q}') c INNER JOIN genes_df g USING(geneid)) TO '{outq}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        finally:
            con.close()
        print(f"Pilot cis map: {cis_path}; genes={len(genes)}", flush=True)

    rows = []
    for method in [x for x in a.methods.split(",") if x]:
        if method not in REVISED_FOLLOWUP_METHODS:
            raise ValueError(f"This wrapper is for revised raw-derived methods only: {REVISED_FOLLOWUP_METHODS}")
        print(f"\n### revised raw-derived PPMI discovery: {method}", flush=True)
        z = run_method_partitioned(
            paths, "PP", method, cis_path=cis_path,
            genes_per_chunk=a.genes_per_chunk,
            gpus=gpus,
            cpu_workers=a.cpu_workers,
            batch_size=a.batch_size,
            gpu_batch_map=a.gpu_batch_map,
            min_gpu_batch=a.min_gpu_batch,
            rho_points=a.rho_points,
            legacy_genotype_mode=a.legacy_genotype_mode,
            dtype=a.dtype,
            cell_adjust=True,
            finalize=True,
            temp_root=a.tmp_dir,
            data_mode="current_v89",
            result_label=(method + "__pilot" if cis_path is not None else method),
            ctar_reference_interval=a.ctar_reference_interval,
            ctar_rho_max=a.ctar_rho_max,
            fdr_strategy=a.fdr_strategy,
            fdr_alpha=a.fdr_alpha,
        )
        rows.append(z)
        print(json.dumps(z, indent=2), flush=True)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
