#!/usr/bin/env python3
"""Select 5,000 real overall CTAR tQTLs and export matching PP/PD genotypes.
No dynamic-QTL inference is performed. All selected SNP-gene pairs are preserved
for PPMI; PDBP matching is audited separately.
"""
import argparse, json, subprocess, shutil, re
from pathlib import Path
import numpy as np
import pandas as pd

GENE=('Ensembl_ID','gene_id','gene','phenotype_id')
VAR=('Variant_ID','variant_id','SNP','snp','rsid')

def findcol(cols, choices, required=True):
    norm=lambda x: str(x).strip().lower().replace('_','').replace('-','')
    index={norm(c):c for c in cols}
    for c in choices:
        if norm(c) in index:return index[norm(c)]
    if required:raise ValueError('Missing '+str(choices)+'; available fields: '+str(cols[:80]))
    return None

def iterate(path, cols, batch=100000):
    path=Path(path)
    if path.suffix.lower()=='.csv':
        yield from pd.read_csv(path, usecols=cols, chunksize=batch)
    else:
        import pyarrow.dataset as ds
        data=ds.dataset(str(path),format='parquet')
        for b in data.to_batches(columns=cols,batch_size=batch):yield b.to_pandas()

def columns(path):
    p=Path(path)
    if p.suffix.lower()=='.csv':return list(pd.read_csv(p,nrows=0).columns)
    import pyarrow.dataset as ds
    return ds.dataset(str(p),format='parquet').schema.names

def select_loci(path,n,seed):
    cols=columns(path)
    gene=findcol(cols,GENE);var=findcol(cols,VAR)
    p=findcol(cols,('Pvalue','overall_pvalue','temporal_pvalue'))
    fdr=findcol(cols,('FDR','overall_FDR','qvalue'),False)
    rho=findcol(cols,('rho','rho_ref'),False)
    lrt=findcol(cols,('LRT','overall_LRT'),False)
    n_cov=findcol(cols,('n_covariates_used',),False)
    use=list(dict.fromkeys(x for x in (gene,var,p,fdr,rho,lrt,n_cov) if x))
    rng=np.random.default_rng(seed);keep=[];passed=0;max_p=0.
    for chunk in iterate(path,use):
        chunk=chunk.rename(columns={gene:'gene',var:'variant',p:'pvalue',**({fdr:'fdr'} if fdr else {}),**({rho:'rho'} if rho else {}),**({lrt:'lrt'} if lrt else {}),**({n_cov:'n_covariates_used'} if n_cov else {})})
        chunk['pvalue']=pd.to_numeric(chunk['pvalue'],errors='coerce')
        mask=chunk['pvalue'].between(0,1,inclusive='both')
        if fdr:
            chunk['fdr']=pd.to_numeric(chunk.fdr,errors='coerce')
            mask &= chunk.fdr.lt(.05)
        elif 'significant_tqtl' not in str(path):
            raise ValueError('Unfiltered discovery results lack FDR; provide significant_tqtl.parquet')
        x=chunk.loc[mask].copy()
        if not len(x):continue
        passed+=len(x);max_p=max(max_p,float(x.pvalue.max()))
        x['variant']=x['variant'].astype(str);x['gene']=x['gene'].astype(str)
        x['priority']=rng.random(len(x))
        keep.append(x)
        if sum(len(i) for i in keep)>max(n*3,18000):
            keep=[pd.concat(keep,ignore_index=True).nsmallest(n,'priority')]
    if not keep:raise ValueError('No overall significant temporal tQTLs identified')
    result=pd.concat(keep,ignore_index=True).nsmallest(n,'priority').drop(columns=['priority'])
    if result.duplicated(['variant','gene']).any():
        raise ValueError('Duplicated Variant_ID+Ensembl_ID in discovery: check source')
    return result.reset_index(drop=True), {'total_significant':int(passed),'sampled':len(result),
          'observed_max_raw_p_of_all_hits':max_p,'columns':dict(gene=gene,variant=var,p=p,fdr=fdr)}

def match_rep(path,selected):
    cols=columns(path)
    gene=findcol(cols,GENE);variant=findcol(cols,VAR)
    p=findcol(cols,('Pvalue','overall_pvalue'))
    rho=findcol(cols,('rho','rho_ref'),False)
    lrt=findcol(cols,('LRT','Overall_LRT'),False)
    use=[x for x in (gene,variant,p,rho,lrt) if x]
    variants=set(selected.variant.astype(str)); pairs=set(zip(selected.variant.astype(str),selected.gene.astype(str)))
    matches=[]
    for chunk in iterate(path,use):
        x=chunk[chunk[variant].astype(str).isin(variants)]
        if len(x):
            x=x.loc[[tuple(z) in pairs for z in zip(x[variant].astype(str),x[gene].astype(str))]]
            if len(x):matches.append(x.rename(columns={gene:'gene',variant:'variant',p:'pd_pvalue',**({rho:'pd_rho'} if rho else {}),**({lrt:'pd_lrt'} if lrt else {})}))
    if not matches:raise ValueError('No matching PDBP results; check selected locus keys and replication file')
    rep=pd.concat(matches,ignore_index=True)
    if rep.duplicated(['variant','gene']).any():raise ValueError('Duplicate PDBP locus results')
    selected=selected.merge(rep,on=['variant','gene'],how='left',validate='one_to_one',indicator='rep_match')
    selected['pd_nominal_replicated']=pd.to_numeric(selected.pd_pvalue,errors='coerce')<.05
    return selected

def resolve_plink(plink):
    """Use the already-installed project PLINK1.9. Do not assume PLINK2 exists."""
    requested=str(plink or '/usr/bin/plink1.9')
    if '/' in requested:
        path=Path(requested)
        if not path.is_file():
            raise FileNotFoundError(f'PLINK executable does not exist: {requested}')
        if not path.stat().st_mode & 0o111:
            raise PermissionError(f'PLINK is not executable: {requested}')
        binary=str(path)
    else:
        binary=shutil.which(requested)
        if not binary:raise RuntimeError(f'PLINK executable {requested!r} not on PATH. Project PLINK1.9: /usr/bin/plink1.9')
    version=subprocess.run([binary,'--version'],capture_output=True,text=True,check=False)
    message=(version.stdout+' '+version.stderr).lower()
    if not re.search(r'plink\s+v?1\.9|plink\s+1\.9|v1\.9',message):
        raise RuntimeError(f'Expected PLINK 1.9 but got {message.strip()[:180]!r} from {binary}. Use --plink /usr/bin/plink1.9')
    return binary


def export_plink(prefix,variants,out,plink):
    prefix=Path(prefix)
    for suffix in ('.bed','.bim','.fam'):
        if not Path(str(prefix)+suffix).exists():raise FileNotFoundError(str(prefix)+suffix)
    binary=resolve_plink(plink)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    snps=out.parent/(out.stem+'_extract.txt')
    wanted=set(str(v).strip() for v in variants if pd.notna(v) and str(v).strip())
    if not wanted:raise ValueError('No SNPs requested for PLINK extraction')
    # Stream BIM once: verify actual SNP IDs in each cohort before extraction.
    present=set()
    with Path(str(prefix)+'.bim').open() as bim:
        for line in bim:
            parts=line.split()
            if len(parts)>1 and parts[1] in wanted: present.add(parts[1])
    if not present: raise ValueError(f'Zero requested SNP IDs matched {prefix}.bim. Check Variant_ID naming / build.')
    snps.write_text('\n'.join(sorted(present))+'\n')
    print(f'[BIM QC] requested={len(wanted)} present={len(present)} absent={len(wanted-present)}',flush=True)
    # Do not reuse stale PLINK2 or other allele-orientation dosage files merely
    # because their .raw pathname exists. Always validate the requested SNP set.
    raw=Path(str(out)+'.raw')
    if raw.exists():
        header=raw.open().readline().split()
        meta={'FID','IID','PAT','MAT','SEX','PHENOTYPE'}
        snp_columns=[c for c in header if c not in meta]
        got=set(c.rsplit('_',1)[0] for c in snp_columns)
        if present.issubset(got):
            print(f'[REUSE VERIFIED] {raw}: {len(present)} BIM-matched SNP IDs in genotype header',flush=True)
            return
        print(f'[REBUILD] stale .raw lacks {len(present-got)} BIM-matched SNP IDs: {raw}',flush=True)
        raw.unlink()
    # PLINK 1.9 equivalent of PLINK2 --export A is --recode A.
    cmd=[binary,'--bfile',str(prefix),'--extract',str(snps),'--recode','A','--out',str(out)]
    print('[PLINK1.9]',' '.join(cmd),flush=True)
    subprocess.run(cmd,check=True)
    if not raw.exists():raise FileNotFoundError('PLINK 1.9 did not produce '+str(raw))
    with raw.open() as f: header=f.readline().split()
    if not {'FID','IID'}.issubset(header):raise ValueError(f'{raw}: FID/IID missing from PLINK .raw header')
    got={c.rsplit('_',1)[0] for c in header if c not in {'FID','IID','PAT','MAT','SEX','PHENOTYPE'}}
    print(f'[PLINK1.9 QC] requested={len(wanted)} exported={len(got & wanted)} missing={len(wanted-got)}',flush=True)
    if len(got & present)==0:
        raise ValueError(f'No requested SNPs exported to {raw}. Check Variant_ID vs BIM SNP column.')
    missing=sorted(wanted-got)
    if missing:
        (out.parent/(out.stem+'_unmatched_variants.txt')).write_text('\n'.join(missing)+'\n')
        print(f'[WARN] {len(missing)} SNPs not in {prefix}.bim; saved missing list; per-locus fit will audit.',flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    ap.add_argument('--discovery-file');ap.add_argument('--replication-file')
    ap.add_argument('--pp-genotype');ap.add_argument('--pd-genotype')
    ap.add_argument('--outdir');ap.add_argument('--n-pairs',type=int,default=5000)
    ap.add_argument('--seed',type=int,default=20261030);ap.add_argument('--plink','--plink2',dest='plink',default='/usr/bin/plink1.9',help='PLINK 1.9 executable (legacy --plink2 spelling accepted as alias)')
    ap.add_argument('--skip-plink',action='store_true')
    a=ap.parse_args();root=Path(a.root)
    pp=a.discovery_file or root/'qtl/PP/submitted_followup_ctar_mptga/final/significant_tqtl.parquet'
    pdfile=a.replication_file or root/'qtl/PD/submitted_followup_ctar_mptga/final/all_results.parquet'
    out=Path(a.outdir or root/'manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_refit_v3');out.mkdir(parents=True,exist_ok=True)
    selected,info=select_loci(pp,a.n_pairs,a.seed)
    selected=match_rep(pdfile,selected)
    selected.to_csv(out/'selected_loci.csv',index=False)
    meta={'discovery_file':str(pp),'replication_file':str(pdfile),**info,
        'matched_PDBP':int((selected.rep_match=='both').sum()),'selected_PPMI':len(selected)}
    (out/'selection_audit.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False))
    if not a.skip_plink:
        pp_prefix=a.pp_genotype or root/'data/genotype/PP/amp_pd_PP_qc'
        pd_prefix=a.pd_genotype or root/'data/genotype/REP/amp_pd_REP_qc'
        for label,prefix in [('PP',pp_prefix),('PD',pd_prefix)]:
            export_plink(prefix,selected.variant, out/f'genotypes_{label}',a.plink)
    print(json.dumps(meta,indent=2,ensure_ascii=False),flush=True)
    print('[PREPARED]',out,flush=True)
if __name__=='__main__':main()
