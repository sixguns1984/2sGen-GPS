#!/usr/bin/env python
"""Run a Stage-I PPMI submitted-MPTGA calibration without cell-composition covariates.

This is an internal audit, not a second primary/sensitivity analysis.  It uses the
same strict manuscript PPMI data, cis-pair universe, binary hard-call encoding,
100-point legacy rho grid, FP64 engine, and AR(1) sufficient-statistic solver as
the primary Stage-I submitted MPTGA.  The only intended model difference is the
omission of the six CIBERSORTx broad-cell CLR covariates.

Output is isolated under:
  qtl/PP/submitted_followup_mptga__nocell_audit/
so it cannot overwrite or reuse primary +cell results.
"""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_CODE))

from gengps.amp_pd.config import AmpPdPaths, LEGACY_TIMES_BY_COHORT
from gengps.amp_pd.scheduler import run_method_partitioned


def parse_floats(x):
    return [float(v) for v in str(x).split(",") if str(v).strip() != ""]


def main():
    p = argparse.ArgumentParser(
        description="PPMI Stage-I submitted MPTGA no-cell internal calibration audit"
    )
    p.add_argument("--root", default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt")
    p.add_argument("--gpus", default="0,1,2")
    p.add_argument("--cpu-workers", type=int, default=12)
    p.add_argument("--genes-per-chunk", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--gpu-batch-map", default="0:8192,1:4096,2:4096")
    p.add_argument("--min-gpu-batch", type=int, default=512)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--tmp-dir", default=os.environ.get("TMPDIR", "/public/labdata/luojunfeng/tmp"))
    p.add_argument("--rho-points", type=int, default=100)
    p.add_argument("--legacy-times", default=",".join(map(str, LEGACY_TIMES_BY_COHORT["PP"])))
    a = p.parse_args()

    if a.dtype != "float64":
        p.error("The Stage-I no-cell audit must use --dtype float64 to match the primary analysis.")
    if a.rho_points != 100:
        p.error("The Stage-I no-cell audit must use --rho-points 100 (0.00,...,0.99).")

    paths = AmpPdPaths(root=a.root)
    gpus = [int(x) for x in a.gpus.split(",") if x != ""]

    out = run_method_partitioned(
        paths,
        "PP",
        "submitted_followup_mptga",
        genes_per_chunk=a.genes_per_chunk,
        gpus=gpus,
        cpu_workers=a.cpu_workers,
        batch_size=a.batch_size,
        gpu_batch_map=a.gpu_batch_map,
        min_gpu_batch=a.min_gpu_batch,
        rho_points=a.rho_points,
        legacy_times=parse_floats(a.legacy_times),
        legacy_genotype_mode="binary",
        dtype=a.dtype,
        cell_adjust=False,
        result_label="submitted_followup_mptga__nocell_audit",
        finalize=True,
        temp_root=a.tmp_dir,
        data_mode="stage1_manuscript",
        stage1_cell_mode="nocell_audit",
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
