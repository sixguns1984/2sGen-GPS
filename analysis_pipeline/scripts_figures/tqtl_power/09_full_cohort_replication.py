#!/usr/bin/env python3
"""Full (not sampled) PPMI -> PDBP CTAR replication from already finalized Stage-I audit.
No CTAR refit. Full PPMI BH discoveries + PDBP matched same SNP-gene tests.
"""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True);ap.add_argument('--figroot',required=True)
    a=ap.parse_args();root=Path(a.root);out=Path(a.figroot)/'power_analysis_CTAR_v5';out.mkdir(parents=True,exist_ok=True)
    f=root/'single_iv_causal/iv_validation/single_iv_02_replication_direction_audit.csv'
    if not f.is_file():raise FileNotFoundError(f'Existing Stage1 FULL replication-direction audit missing: {f}; do not substitute the LD-pruned eligible list')
    df=pd.read_csv(f,low_memory=False)
    req=['gene','lead_snp','ppmi_tqtl_p','ppmi_temporal_fdr','pdbp_tqtl_p']
    if not set(req).issubset(df):raise RuntimeError(f'{f}: missing {set(req)-set(df)}')
    if df.duplicated(['gene','lead_snp']).any():raise RuntimeError('Duplicate full Stage1 SNP-gene pairs')
    q=pd.to_numeric(df.ppmi_temporal_fdr,errors='coerce');p=pd.to_numeric(df.pdbp_tqtl_p,errors='coerce')
    df=df.loc[q.lt(.05)&np.isfinite(pd.to_numeric(df.ppmi_tqtl_p,errors='coerce'))].copy()
    p=pd.to_numeric(df.pdbp_tqtl_p,errors='coerce')
    matched=p.notna()&p.between(0,1,inclusive='both')
    n_disc=len(df);n_matched=int(matched.sum());n_rep=int((matched&(p<.05)).sum())
    if not n_matched:raise RuntimeError('No PDBP matched full Stage1 candidates')
    tr=pd.to_numeric(df.get('pdbp_direction_value',pd.Series(np.nan,index=df.index)),errors='coerce')
    direction=(matched&(p<.05)&(tr>0))
    summ={'source':str(f),'PPMI_global_FDR_significant_pairs':n_disc,
          'PDBP_matched_testable_pairs':n_matched,
          'PDBP_nominal_P_less_0_05_pairs':n_rep,
          'PDBP_nominal_replication_rate':n_rep/n_matched,
          'PDBP_nominal_and_trajectory_concordant_count':int(direction.sum()),
          'PDBP_nominal_and_trajectory_concordant_rate':float(direction.sum()/n_matched),
          'denominator':'PDBP matched PPMI BH-significant SNP-gene tests; no LD pruning; identical full Stage1 audit',
          'no_model_refit':True}
    pd.DataFrame([summ]).to_csv(out/'full_PPMI_PDBP_replication_summary.csv',index=False)
    (out/'full_replication_manifest.json').write_text(json.dumps(summ,indent=2))
    print('[FULL DISCOVERY/REPLICATION]',summ,flush=True)
if __name__=='__main__':main()
