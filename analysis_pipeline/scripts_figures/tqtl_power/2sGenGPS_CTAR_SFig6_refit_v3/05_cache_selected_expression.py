#!/usr/bin/env python3
"""Optional but strongly recommended: one-pass export of expression for selected genes.
Designed for large real parquet matrices; no normalization or numerical changes.
"""
import argparse,json
from pathlib import Path
import pandas as pd
import sys
CODE=Path(__file__).resolve().parents[3]
if str(CODE) not in sys.path:sys.path.insert(0,str(CODE))
from gengps.amp_pd.config import AmpPdPaths
from gengps.amp_pd.real_qtl import resolve_expression_gene_columns

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    a.add_argument('--expression');a.add_argument('--selected-file');a.add_argument('--outdir');a.add_argument('--overwrite',action='store_true')
    x=a.parse_args();root=Path(x.root)
    out=Path(x.outdir or root/'manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_refit_v3');out.mkdir(parents=True,exist_ok=True)
    selected=pd.read_csv(x.selected_file or out/'selected_loci.csv')
    genes=set(selected.gene.astype(str))
    inp=Path(x.expression or root/'data/expression/amp_pd_expression_z.parquet')
    dest=out/'expression_selected.parquet'
    if dest.exists() and not x.overwrite:
        print('[REUSE] Existing expression cache',dest)
        return
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    mapping=resolve_expression_gene_columns(AmpPdPaths(root=str(root)),genes)
    physical=list(dict.fromkeys(mapping.values()))
    source=ds.dataset(str(inp),format='parquet')
    cols=source.schema.names
    if 'sample_id' not in cols:raise ValueError('Production expression missing sample_id')
    missing=set(physical)-set(cols)
    if missing:raise ValueError(f'Production expression columns missing: {sorted(missing)[:20]}')
    mode='wide_symbol_mapped'
    scanner=source.scanner(columns=['sample_id']+physical,batch_size=5000)
    writer=None;rows=0
    try:
        for batch in scanner.to_batches():
            if not len(batch):continue
            if writer is None:writer=pq.ParquetWriter(dest,batch.schema,compression='zstd')
            writer.write_batch(batch);rows+=len(batch)
    finally:
        if writer:writer.close()
    if not rows:raise ValueError('No expression rows written; check identifiers')
    report={'mode':mode,'source':str(inp),'output':str(dest),'n_selected_genes':len(genes),'output_rows':rows}
    (out/'expression_cache_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
