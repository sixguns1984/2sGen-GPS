#!/usr/bin/env python3
"""Read-only Stage-I alignment audit before bootstrap. Never edits original Stage-I files."""
import argparse,json
from pathlib import Path
import pandas as pd
from production_adapter import ProductionMetadata,ProductionExpression,ProductionBedGenotypes,build_production_locus

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--root',required=True)
    a.add_argument('--outdir',required=True)
    a.add_argument('--n-pairs',type=int,default=12)
    x=a.parse_args()
    root=Path(x.root);out=Path(x.outdir)
    loci=pd.read_csv(out/'selected_loci.csv').head(x.n_pairs)
    m={co:ProductionMetadata(root,co) for co in ('PP','PD')}
    g={co:ProductionBedGenotypes(root,co,m[co].data.participant_id,loci.variant.astype(str).unique()) for co in m}
    expr_path=(out/'expression_selected.parquet' if (out/'expression_selected.parquet').is_file() else None)
    e=ProductionExpression(root,loci.gene.astype(str).unique(),cache_file=expr_path)
    audit=[]
    for r in loci.itertuples(index=False):
        for co in ('PP','PD'):
            if co=='PD' and getattr(r,'rep_match','both')!='both':continue
            try:
                z=build_production_locus(m[co],g[co],e,r.gene,r.variant)
                expected=(getattr(r,'n_covariates_used',None) if co=='PP' else None)
                if expected is not None and pd.notna(expected) and int(expected)!=z['n_covariates']:
                    raise ValueError(f'Stage-I nuisance count {expected} vs refit {z["n_covariates"]}')
                audit.append(dict(gene=r.gene,variant=r.variant,cohort=co,status='PASS',n_donors=z['n_donors'],n_obs=z['n_obs'],n_covariates=z['n_covariates'],genotypes=str(z['g_counts'])))
            except Exception as ex:
                audit.append(dict(gene=r.gene,variant=r.variant,cohort=co,status='FAIL',reason=str(ex)[:350]))
    pd.DataFrame(audit).to_csv(out/'stage1_input_alignment_audit.csv',index=False)
    report={'metadata':{co:m[co].summary for co in m},'genotype_matches':{co:g[co].coverage for co in g},
            'expression_missing_genes':sorted(e.unknown)[:30],
            'valid':sum(q['status']=='PASS' for q in audit),'total':len(audit)}
    (out/'stage1_input_alignment_audit.json').write_text(json.dumps(report,indent=2,default=str))
    print('[ALIGNMENT AUDIT] valid',report['valid'],'/',report['total'])
    for q in audit:
        print(q['cohort'],q['gene'],q['variant'],q['status'],q.get('reason',''))
    if report['valid']<2:raise SystemExit('[FAIL] Not enough valid matched loci; see stage1_input_alignment_audit.csv. Do not run bootstrap.')
if __name__=='__main__': main()
