#!/usr/bin/env python
from pathlib import Path
import argparse, json, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gengps.amp_pd.config import AmpPdPaths, DEFAULT_MANUAL_GPCS
from gengps.amp_pd.preprocess import save_preprocessed_expression_and_metadata
from gengps.amp_pd.eigensoft import parse_map

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt")
    p.add_argument("--skip-plink", action="store_true",
                   help="Reuse existing data/genotype/PP and data/genotype/REP QC BED sets.")
    p.add_argument("--force-plink", action="store_true",
                   help="Overwrite/recreate PP and REP PLINK QC outputs.")
    p.add_argument("--age-min", type=float, default=30)
    p.add_argument("--age-max", type=float, default=90)
    p.add_argument("--qtl-mean-tpm", type=float, default=1.0,
                   help="Primary QTL expression threshold; default TPM>=1.")
    p.add_argument("--deconv-mean-tpm", type=float, default=1.0,
                   help=("Threshold for the broad raw-TPM cache written during preprocessing. "
                         "The v8.9 LM22 exporter rebuilds its own CIBERSORTx-only matrix from "
                         "source TPM at mean TPM>=0.01, so this preprocessing cache threshold "
                         "does not define the final LM22 mixture universe."))
    p.add_argument("--include-sex-chr", action="store_true",
                   help="Do not restrict QTL gene universe to autosomes.")
    p.add_argument("--pc-mode", choices=["eigensoft", "eigensoft_or_manual", "manual"],
                   default="eigensoft",
                   help="Production default: PLINK PCA eigenvalues + EIGENSOFT 6.1.4 twstats Tracy-Widom selection.")
    p.add_argument("--manual-gpcs", default="PP:8,REP:8",
                   help="Manual genotype PC counts or fallback, e.g. PP:8,REP:8.")
    p.add_argument("--max-pcs", type=int, default=20)
    p.add_argument("--tw-alpha", type=float, default=0.05)
    p.add_argument("--twstats", default="twstats",
                   help="twstats command from the current PATH; this matches the working interactive command.")
    p.add_argument("--twtable", default="twtable",
                   help="Argument passed to EIGENSOFT 6.1.4 -t; default literal token 'twtable', matching the known working command.")
    p.add_argument("--plink-threads", type=int, default=16)
    a = p.parse_args()

    paths = AmpPdPaths(root=a.root)
    summary = save_preprocessed_expression_and_metadata(
        paths,
        run_plink=not a.skip_plink,
        force_plink=a.force_plink,
        age_min=a.age_min,
        age_max=a.age_max,
        qtl_mean_tpm=a.qtl_mean_tpm,
        deconv_mean_tpm=a.deconv_mean_tpm,
        autosome_only=not a.include_sex_chr,
        pc_mode=a.pc_mode,
        manual_gpcs=parse_map(a.manual_gpcs, DEFAULT_MANUAL_GPCS),
        max_pcs=a.max_pcs,
        tw_alpha=a.tw_alpha,
        twstats_exe=a.twstats,
        twtable=a.twtable,
        plink_threads=a.plink_threads,
    )
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
