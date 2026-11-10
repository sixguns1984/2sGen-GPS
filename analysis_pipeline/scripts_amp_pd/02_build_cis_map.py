#!/usr/bin/env python
from pathlib import Path
import argparse, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gengps.amp_pd.config import AmpPdPaths
from gengps.amp_pd.cis import prepare_cis_map

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt")
    p.add_argument("--window", type=int, default=1_000_000)
    a = p.parse_args()
    paths = AmpPdPaths(root=a.root)
    summary = prepare_cis_map(paths, window=a.window)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
