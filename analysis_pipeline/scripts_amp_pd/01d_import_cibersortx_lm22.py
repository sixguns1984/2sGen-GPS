#!/usr/bin/env python
from pathlib import Path
import argparse, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gengps.amp_pd.config import AmpPdPaths
from gengps.amp_pd.deconvolution import import_cibersortx_lm22_results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt")
    p.add_argument("--ppmi", required=True, help="PPMI CIBERSORTx Fractions result file")
    p.add_argument("--pdbp", required=True, help="PDBP CIBERSORTx Fractions result file")
    p.add_argument("--biofind", required=True, help="BioFIND CIBERSORTx Fractions result file")
    p.add_argument("--pseudocount", type=float, default=1e-5)
    a = p.parse_args()
    paths = AmpPdPaths(root=a.root)
    report = import_cibersortx_lm22_results(
        paths, a.ppmi, a.pdbp, a.biofind, pseudocount=a.pseudocount
    )
    print(report.to_string(index=False))

if __name__ == "__main__":
    main()
