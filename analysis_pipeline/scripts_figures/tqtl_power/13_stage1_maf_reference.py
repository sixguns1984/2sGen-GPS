#!/usr/bin/env python3
"""Descriptive MAF of frozen validated single-IV SNPs in PPMI, without QC/filtering.
Uses the original Stage1 genotype source and actual eligible participants.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--root',required=True);a.add_argument('--code',required=True);a.add_argument('--figroot',required=True)
    a.add_argument('--chunk',type=int,default=400);a.add_argument('--refresh',action='store_true')
    args=a.parse_args();root=Path(args.root);out=Path(args.figroot)/'power_analysis_CTAR_v5';out.mkdir(parents=True,exist_ok=True)
    f=out/'MAF_frozen_Stage1_IVs.csv'
    if f.is_file() and not args.refresh:
        z=pd.read_csv(f)
        print(f'[MAF] Reused {f}: median={z.MAF.median():.4f}; n={len(z)}',flush=True)
        return
    source=root/'single_iv_causal/iv_validation/single_iv_primary_eligible.csv'
    if not source.is_file():raise FileNotFoundError(source)
    s=pd.read_csv(source,usecols=['lead_snp'],dtype=str).lead_snp.dropna().drop_duplicates().tolist()
    sys.path.insert(0,args.code)
    from gengps.amp_pd.config import AmpPdPaths
    from gengps.amp_pd.real_qtl import (load_cohort_metadata,bfile_for_cohort,
          _augment_fixed_genotype_pcs,_prepare_irregular_followup_rows,method_covariates,_drop_missing_covariates)
    from gengps.amp_pd.genotype import make_genotype_source
    paths=AmpPdPaths(root=str(root)); meta=load_cohort_metadata(paths,'PP')
    meta=_augment_fixed_genotype_pcs(paths,'PP',meta,n_pcs=8)
    meta,_,_=_prepare_irregular_followup_rows(meta,pd.DataFrame(index=meta.index))
    cov=method_covariates('submitted_followup_ctar_mptga','PP',meta,cell_adjust=True)
    meta,_=_drop_missing_covariates(meta,pd.DataFrame(index=meta.index),cov)
    pids=pd.Index(meta.participant_id.astype(str).unique(),name='participant_id')
    src=make_genotype_source(str(bfile_for_cohort(paths,'PP')),paths.plink19,prefer_bed_reader=True)
    rows=[];errors=[]
    for ix in range(0,len(s),args.chunk):
        names=s[ix:ix+args.chunk]
        # A few bad SNP names must not invalidate the entire batch.
        try:g=src.read(pids,names,mean_impute=False,dtype=np.float64)
        except Exception as e:
            errors.append({'batch_start':ix,'snps':','.join(names[:5]),'error':str(e)});continue
        for snp in names:
            if snp not in g:continue
            x=pd.to_numeric(g[snp],errors='coerce').to_numpy(float)
            x=x[np.isfinite(x)&np.isin(x,[0.,1.,2.])]
            if len(x)<1:continue
            p=float(x.mean()/2.);maf=min(p,1-p)
            rows.append({'lead_snp':snp,'MAF':maf,'n_called':len(x),
                         'n_0':int((x==0).sum()),'n_1':int((x==1).sum()),'n_2':int((x==2).sum())})
        print(f'[MAF] {min(ix+args.chunk,len(s))}/{len(s)}',flush=True)
    z=pd.DataFrame(rows)
    if z.empty:raise RuntimeError('Could not extract MAFs; check genotype reference and participant ID matching')
    z.to_csv(f,index=False)
    pd.DataFrame(errors).to_csv(out/'MAF_read_errors.csv',index=False)
    (out/'MAF_manifest.json').write_text(json.dumps({'source':str(source),
        'n_candidates':len(s),'n_maf':len(z),'n_ppmi_participants':len(pids),
        'reference_maf_median':float(z.MAF.median()),
        'unit':'minor allele frequency in PPMI Stage1 eligible participants; descriptive only, not an additional SNP filter',
        'limitation':'MAF distribution of frozen validated IV pool, not necessarily all genome-wide SNPs'},indent=2))
    print(f'[MAF] n={len(z)}, median={z.MAF.median():.4f}',flush=True)

if __name__=='__main__':main()
