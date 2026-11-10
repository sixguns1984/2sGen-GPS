#!/usr/bin/env python3
"""Preflight check of real 2sGenGPS CTAR Stage-I data; never runs a power grid."""
from pathlib import Path
import argparse, json, sys
from ctar_sfig6_n80 import (meta_templates,resolve_thresholds,load_input_loci,
                             empirical_replication)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    ap.add_argument('--n-pairs',type=int,default=1500)
    ap.add_argument('--min-pairs',type=int,default=1000)
    ap.add_argument('--seed',type=int,default=20261030)
    ap.add_argument('--loci-file',default=None)
    ap.add_argument('--metadata',default=None)
    ap.add_argument('--plink-prefix',default=None)
    args=ap.parse_args()
    root=Path(args.root)
    qtl=Path(args.loci_file) if args.loci_file else root/'qtl/PP/submitted_followup_ctar_mptga/final/significant_tqtl.parquet'
    rep=root/'qtl/PD/submitted_followup_ctar_mptga/final/all_results.parquet'
    meta=Path(args.metadata) if args.metadata else root/'data/metadata/amp_pd_samples.parquet'
    prefix=args.plink_prefix or str(root/'data/genotype/PP/amp_pd_PP_qc')
    print('[AUDIT] Real CTAR discovery:',qtl,flush=True)
    print('[AUDIT] Real CTAR replication:',rep,flush=True)
    print('[AUDIT] Metadata:',meta,flush=True)
    print('[AUDIT] PPMI genotype:',prefix,flush=True)
    for file in (qtl,meta,Path(prefix+'.bim'),Path(prefix+'.bed')):
        if not file.is_file():raise FileNotFoundError(f'[AUDIT] Missing actual project input: {file}')
    groups=meta_templates(meta)
    ns={key:len(z) for key,z in groups.items()}
    print('[AUDIT] Independent participants:',ns,flush=True)
    disc,inter,source=resolve_thresholds(qtl)
    print('[AUDIT] Now validating Ensembl_ID/Variant_ID and 1,500 true PPMI MAF values...',flush=True)
    loci=load_input_loci(qtl,args.seed,args.n_pairs,args.min_pairs,prefix)
    result=empirical_replication(loci,rep)
    report={'PPMI_QTL':str(qtl),'PDBP_replication':str(rep),'metadata':str(meta),
            'plink_prefix':prefix,'n_participants':ns,
            'alpha_disc':disc,'alpha_inter':inter,'threshold_source':source,
            'sampling':loci.attrs.get('loci_audit',{}),'replication':result,
            'status':'PASS'}
    out=root/'manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_N80_final'
    out.mkdir(parents=True,exist_ok=True)
    path=out/'input_audit.json'
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False,default=float),encoding='utf-8')
    print('\n[AUDIT PASS] All primary real inputs accepted. Saved:',path,flush=True)
    print('You can now run: nohup bash run_sfigure6_final.sh > "$ROOT/logs/sfigure6_ctar_N80_final.log" 2>&1 &',flush=True)


if __name__=='__main__':main()
