#!/usr/bin/env python
from pathlib import Path
import argparse, sys, json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gengps.amp_pd.config import AmpPdPaths
from gengps.amp_pd.deconvolution import export_cibersortx_inputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt",
    )
    p.add_argument(
        "--lm22",
        required=True,
        help="LM22 signature-matrix path.",
    )
    p.add_argument(
        "--deconv-mean-tpm",
        type=float,
        default=0.01,
        help=(
            "CIBERSORTx-only mean TPM threshold. Default 0.01 after the LM22 "
            "threshold-sensitivity audit. This does not modify the QTL TPM>=1 universe."
        ),
    )
    p.add_argument(
        "--signature-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write only genes shared with LM22 to the mixture files. Default: true. "
            "Use --no-signature-only to export the entire deconvolution gene universe."
        ),
    )
    a = p.parse_args()

    paths = AmpPdPaths(root=a.root)
    outputs = export_cibersortx_inputs(
        paths,
        split_by_cohort=True,
        lm22_path=a.lm22,
        deconv_mean_tpm=a.deconv_mean_tpm,
        signature_only=a.signature_only,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
