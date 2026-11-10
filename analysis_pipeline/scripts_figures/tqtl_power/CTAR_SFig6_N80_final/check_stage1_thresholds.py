#!/usr/bin/env python3
"""Read Stage-I thresholds only, without running SFig6 power grids."""
import argparse
from pathlib import Path
from ctar_sfig6_n80 import resolve_thresholds
ap=argparse.ArgumentParser()
ap.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
ap.add_argument('--loci-file',default=None)
a=ap.parse_args()
qtl=Path(a.loci_file) if a.loci_file else Path(a.root)/'qtl/PP/submitted_followup_ctar_mptga/final/significant_tqtl.parquet'
print('CTAR QTL path:',qtl)
print('QTL file exists:',qtl.exists())
print('Finalization JSON exists:',(qtl.parent/'fdr_finalize_summary.json').exists())
print('Full scan file exists:',(qtl.parent/'all_results.parquet').exists())
resolve_thresholds(qtl)
