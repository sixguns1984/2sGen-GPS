#!/usr/bin/env python
from pathlib import Path
import argparse, json, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gengps.amp_pd.config import (
    AmpPdPaths, EXPRESSION_PC_BE_GROUPS,
    DEFAULT_BE_B, DEFAULT_BE_ALPHA, DEFAULT_BE_SEED, DEFAULT_BE_CORES,
)
from gengps.amp_pd.expression_pc import prepare_run_merge

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt")
    p.add_argument("--groups", default=",".join(EXPRESSION_PC_BE_GROUPS.keys()),
                   help="Comma-separated: PP_FULL,PP_LEGACY,PD_LEGACY,REP_FULL")
    p.add_argument("--B", type=int, default=DEFAULT_BE_B)
    p.add_argument("--alpha", type=float, default=DEFAULT_BE_ALPHA)
    p.add_argument("--seed", type=int, default=DEFAULT_BE_SEED)
    p.add_argument("--cores", type=int, default=DEFAULT_BE_CORES)
    p.add_argument("--rscript", default="Rscript")
    a = p.parse_args()

    groups = [x for x in a.groups.split(",") if x]
    paths = AmpPdPaths(root=a.root)
    result = prepare_run_merge(
        paths,
        groups=groups,
        B=a.B,
        alpha=a.alpha,
        seed=a.seed,
        cores=a.cores,
        rscript=a.rscript,
    )
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
