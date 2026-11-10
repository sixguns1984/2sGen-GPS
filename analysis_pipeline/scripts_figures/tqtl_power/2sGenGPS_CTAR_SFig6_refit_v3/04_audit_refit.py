#!/usr/bin/env python3
"""Hard audit against the original Stage-I overall LRT, before publication."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',required=True);p.add_argument('--output',required=True)
    p.add_argument('--max-median-abs-error',type=float,default=2.0)
    p.add_argument('--min-pairs',type=int,default=5)
    p.add_argument('--allow-unvalidated',action='store_true')
    a=p.parse_args()
    if not Path(a.results).exists() or Path(a.results).stat().st_size==0:
        raise SystemExit(f'[NO FITS] Cannot audit empty result: {a.results}; inspect locus_failures.csv before rerunning')
    try:d=pd.read_csv(a.results)
    except pd.errors.EmptyDataError:
        raise SystemExit(f'[NO FITS] Empty result CSV: {a.results}; inspect locus_failures.csv')
    pp=d[d.cohort=='PP']
    if 'lrt_abs_diff' not in pp:pp=pp.iloc[0:0]
    else:pp=pp[np.isfinite(pd.to_numeric(pp.lrt_abs_diff,errors='coerce'))]
    med=float(pp.lrt_abs_diff.median()) if len(pp) else None
    quant=float(pp.lrt_abs_diff.quantile(.9)) if len(pp) else None
    report={'n_ppmi_reproduced':len(pp),'median_abs_lrt_error':med,'p90_abs_lrt_error':quant,
            'pass': bool(len(pp)>=a.min_pairs and med is not None and med<=a.max_median_abs_error),
            'note':'Fixed-rho conditional CTAR may differ from production rho-profiled scan. Align model and covariates if mismatch is large.'}
    Path(a.output).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    if not report['pass'] and not a.allow_unvalidated:
        raise SystemExit('[FAIL] Refit LRT not verified. Do not use percentage as CTAR power until exact Stage-I matching is established.')
if __name__=='__main__':main()
