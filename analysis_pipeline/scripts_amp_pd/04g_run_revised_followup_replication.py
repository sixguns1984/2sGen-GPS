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
from gengps.amp_pd.scheduler import run_method_partitioned, build_replication_cis_from_discovery


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
    p.add_argument("--ctar-reference-interval", type=float, default=1.0)
    p.add_argument("--ctar-rho-max", type=float, default=0.99)
    p.add_argument("--legacy-genotype-mode", choices=["binary"], default="binary",
                   help="Revised Stage-I is fixed to the historical two-channel binary hard-call coding.")
    p.add_argument("--fdr-strategy", choices=["stagewise_hierarchical", "global_global"],
                   default="stagewise_hierarchical")
    p.add_argument("--fdr-alpha", type=float, default=0.05)
    a = p.parse_args()

    paths = AmpPdPaths(root=a.root)
    gpus = [int(x) for x in a.gpus.split(",") if x]
    rows = []
    for method in [x for x in a.methods.split(",") if x]:
        if method not in REVISED_FOLLOWUP_METHODS:
            raise ValueError(f"This wrapper is for revised raw-derived methods only: {REVISED_FOLLOWUP_METHODS}")
        cmap, n = build_replication_cis_from_discovery(paths, method, "PD", data_mode="current_v89")
        if n == 0:
            print(f"No PPMI discovery hits available in PDBP genotype set for {method}", flush=True)
            continue
        print(f"\n### revised raw-derived PDBP replication: {method}; pairs={n}", flush=True)
        z = run_method_partitioned(
            paths, "PD", method, cis_path=cmap,
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
            result_label=method,
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
