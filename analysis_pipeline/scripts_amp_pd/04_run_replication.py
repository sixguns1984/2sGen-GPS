#!/usr/bin/env python
from pathlib import Path
import argparse, json, sys, os
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gengps.amp_pd.config import AmpPdPaths, FOLLOWUP_METHODS, AGE_METHODS, LEGACY_TIMES_BY_COHORT
from gengps.amp_pd.scheduler import run_method_partitioned, build_replication_cis_from_discovery

def parse_floats(x):
    return [float(v) for v in x.split(",") if v != ""]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt")
    p.add_argument("--methods", default=",".join(FOLLOWUP_METHODS))
    p.add_argument("--gpus", default="0,1,2")
    p.add_argument("--cpu-workers", type=int, default=12)
    p.add_argument("--genes-per-chunk", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--gpu-batch-map", default="0:8192,1:4096,2:4096")
    p.add_argument("--min-gpu-batch", type=int, default=512)
    p.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    p.add_argument("--tmp-dir", default=os.environ.get("TMPDIR", "/public/labdata/luojunfeng/tmp"),
                   help="Temporary/spill directory, including DuckDB global-FDR sorts.")
    p.add_argument("--rho-points", type=int, default=100, help="Stage-I manuscript MPTGA uses the exact legacy 100-point rho grid 0.00,...,0.99.")
    p.add_argument("--legacy-times", default="", help="Optional PDBP override; default is 0,0.5,1,1.5,2.")
    p.add_argument("--spline-df", type=int, default=5)
    p.add_argument("--variance-df", type=int, default=5)
    p.add_argument("--legacy-genotype-mode", choices=["binary","dummies","dosage"], default="binary",
                   help="Stage-I benchmark default is the original BinaryEncoder coding: 0=00,1=01,2=10,-1=11.")
    p.add_argument("--cell-adjust", action="store_true")
    p.add_argument("--data-mode", choices=["stage1_manuscript","current_v89"], default="stage1_manuscript",
                   help="Stage-I defaults to the QCed manuscript PPMI/PDBP matrices; Stage-II/III are unchanged.")
    p.add_argument("--fdr-strategy", choices=["stagewise_hierarchical","global_global"],
                   default="stagewise_hierarchical",
                   help="Finalization policy for replication result files. Replication summary itself uses nominal PDBP P values.")
    p.add_argument("--fdr-alpha", type=float, default=0.05)
    a = p.parse_args()
    if a.data_mode == "stage1_manuscript" and a.cell_adjust:
        p.error("Stage-I now includes cell-composition covariates in the primary submitted MPTGA model; remove --cell-adjust.")

    paths = AmpPdPaths(root=a.root)
    gpus = [int(x) for x in a.gpus.split(",") if x != ""]
    out = []
    for method in [x for x in a.methods.split(",") if x]:
        disc_label = method + ("__celladj" if a.cell_adjust else "")
        cohorts = ["PD"] if method in FOLLOWUP_METHODS else ["REP"]
        for cohort in cohorts:
            cmap, n = build_replication_cis_from_discovery(paths, disc_label, cohort, data_mode=a.data_mode)
            if n == 0:
                print(f"No PPMI discovery hits available in target genotype BIM for {method} -> {cohort}")
                continue
            print(f"\n### Replication {cohort}: {method}; discovery pairs available={n}", flush=True)
            z = run_method_partitioned(
                paths, cohort, method, cis_path=cmap,
                genes_per_chunk=a.genes_per_chunk,
                gpus=gpus, cpu_workers=a.cpu_workers,
                batch_size=a.batch_size, gpu_batch_map=a.gpu_batch_map,
                min_gpu_batch=a.min_gpu_batch,
                rho_points=a.rho_points,
                legacy_times=(parse_floats(a.legacy_times) if a.legacy_times
                              else LEGACY_TIMES_BY_COHORT.get(cohort)),
                spline_df=a.spline_df, variance_df=a.variance_df,
                legacy_genotype_mode=a.legacy_genotype_mode,
                dtype=a.dtype, cell_adjust=a.cell_adjust, finalize=True, temp_root=a.tmp_dir,
                data_mode=a.data_mode,
                fdr_strategy=a.fdr_strategy, fdr_alpha=a.fdr_alpha,
            )
            out.append(z)
            print(json.dumps(z, indent=2), flush=True)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
