#!/usr/bin/env python3
"""Build donor x major cell type pseudobulks from five raw-count H5ADs."""
import argparse, json, re, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

TARGET_SNPS = ('rs2253588', 'rs35162690')
CELL_TYPES = ('Oli','Immune','Inh','OPC','Ast','Vast','Exc')
REGIONS = ('GPI', 'DMNX', 'PFC', 'PMC', 'PVC')

def donor_id(x):
    s=str(x).strip()
    return re.sub(r'-(?:PVC|GPI|DMNX|PFC|PMC)[-_](?:D?WGS)$', '', s, flags=re.I)

def check_unique(df, key, label):
    dup=df.duplicated(key,keep=False)
    if dup.any():
        raise ValueError(f'{label} contains duplicate {key}: {df.loc[dup,key].head(12).tolist()}')

def prepare_clinical(path, out):
    x=pd.read_csv(path,low_memory=False)
    req=['IID','age_at_baseline','SEX','case_control_other_latest']
    missing=set(req)-set(x.columns)
    if missing: raise KeyError(f'Clinical input is missing {missing}')
    x['donor']=x.IID.map(donor_id)
    x['status']=x.case_control_other_latest.astype(str).str.strip().str.title()
    x=x[x.status.isin(['Case','Control'])].copy()
    x['age']=pd.to_numeric(x.age_at_baseline, errors='coerce')
    x['sex']=x.SEX.astype(str).str.strip().str.title()
    x=x[x.sex.isin(['Female','Male']) & x.age.notna()].copy()
    check_unique(x,'donor','clinical')
    y=x[['donor','age','sex','status']].copy()
    y.to_csv(out/'clinical_case_control.csv',index=False)
    return y

def make_genotype(plink_prefix, genotype_raw, out, plink, variant_map='', reference_build='auto', offline=False, allow_position_only=False,
                  tqtl_ppmi_bim='', tqtl_pdbp_bim=''):
    """Extract the exact target sites by resolved BIM IDs without altering source PLINK files."""
    import runpy
    resolve_module = runpy.run_path(str(Path(__file__).with_name('00_resolve_snp_ids.py')))
    run_audit = resolve_module['run_audit']
    prefix=Path(plink_prefix)
    if genotype_raw:
        raw=Path(genotype_raw)
        if not raw.is_file():raise FileNotFoundError(raw)
        # Existing raw files can contain rsIDs or the resolved BIM IDs.
        audit=out/'snp_BIM_resolution_audit.csv'
        resolution=pd.read_csv(audit) if audit.is_file() else pd.DataFrame()
    else:
        missing=[str(prefix)+ext for ext in ('.bed','.bim','.fam') if not Path(str(prefix)+ext).is_file()]
        if missing:raise FileNotFoundError('Missing PLINK inputs: '+str(missing))
        mapped,errors=run_audit(str(prefix),out,variant_map,offline,reference_build,
                               allow_position_only=allow_position_only,
                               tqtl_ppmi_bim=tqtl_ppmi_bim,tqtl_pdbp_bim=tqtl_pdbp_bim)
        if errors: raise ValueError('Could not resolve BOTH target SNPs. See SNP resolution audit.')
        resolution=pd.DataFrame(mapped)
        ids=resolution.set_index('rsid').loc[list(TARGET_SNPS),'bim_id'].astype(str).tolist()
        (out/'snps_to_extract.txt').write_text('\n'.join(ids)+'\n')
        raw_prefix=out/'genotypes'
        cmd=[plink,'--bfile',str(prefix),'--extract',str(out/'snps_to_extract.txt'),
             '--keep-allele-order','--allow-no-sex','--recode','A','--out',str(raw_prefix)]
        print('PLINK COMMAND:', ' '.join(cmd),flush=True)
        subprocess.run(cmd,check=True)
        raw=raw_prefix.with_suffix('.raw')
    df=pd.read_csv(raw,sep=r'\s+',low_memory=False)
    ident='IID' if 'IID' in df else '#IID' if '#IID' in df else None
    if not ident:raise KeyError('PLINK .raw missing IID')
    df['donor']=df[ident].map(donor_id)
    check_unique(df,'donor','PLINK genotype (after donor ID normalization)')
    rs2id = (resolution.set_index('rsid')['bim_id'].astype(str).to_dict()
             if not resolution.empty else {})
    colmap={}
    for snp in TARGET_SNPS:
        ids=[snp] + ([rs2id[snp]] if snp in rs2id and rs2id[snp]!=snp else [])
        matches=[]
        for vid in ids:
            matches.extend([c for c in df.columns if c.startswith(vid+'_')])
        matches=list(dict.fromkeys(matches))
        if len(matches)!=1:
            raise ValueError(f'{snp}: expected one PLINK .raw dosage column for SNP/alias IDs {ids}, got {matches}')
        colmap[snp]=matches[0]
    y=df[['donor']+list(colmap.values())].rename(columns={v:k for k,v in colmap.items()}).copy()
    meta={}
    for snp in TARGET_SNPS:
        y[snp]=pd.to_numeric(y[snp],errors='coerce')
        if ((y[snp].dropna()<0) | (y[snp].dropna()>2)).any():
            raise ValueError(f'{snp}: allele dosage outside [0,2]')
        vid=rs2id.get(snp,snp)
        allele=colmap[snp][len(vid)+1:] if colmap[snp].startswith(vid+'_') else 'unknown'
        meta[snp]={'bim_id':vid,'plink_raw_column':colmap[snp],'counted_allele':allele}
    y.to_csv(out/'snp_dosages.csv',index=False)
    (out/'dosage_alleles.json').write_text(json.dumps(meta,indent=2))
    print('[GENOTYPE] resolved dosages:',meta,flush=True)
    return y

def is_true(s):
    if pd.api.types.is_bool_dtype(s): return s.fillna(False).to_numpy(dtype=bool)
    return s.astype(str).str.lower().str.strip().isin(('true','1','yes','pass','passed')).to_numpy(dtype=bool)

def process_region(region, h5ad_file, eligible, out, chunk_size, min_cells):
    import anndata as ad
    print(f'[H5AD] {region} -> {h5ad_file}',flush=True)
    a=ad.read_h5ad(h5ad_file,backed='r')
    req=['IID','cell_type']
    if any(q not in a.obs.columns for q in req):
        raise KeyError(f'{region}: obs must contain {req}')
    if 'gene_name' not in a.var.columns: raise KeyError(f'{region}: var["gene_name"] missing')
    raw_names=a.var['gene_name'].astype('string').fillna('').str.strip().to_numpy(str)
    var_good=np.array([(bool(n) and n.lower() not in ('nan','none','na')) for n in raw_names])
    # Collapse duplicate gene symbols by SUMMING their counts, rather than arbitrarily dropping one.
    genes,codes=np.unique(raw_names[var_good],return_inverse=True)
    original_idx=np.flatnonzero(var_good)
    projection=sp.csr_matrix((np.ones(len(original_idx),dtype=np.int8), (original_idx,codes)),
                             shape=(a.n_vars,len(genes)))
    donors=a.obs['IID'].astype(str).map(donor_id)
    ctypes=a.obs['cell_type'].astype(str)
    valid=donors.isin(eligible.donor) & ctypes.isin(CELL_TYPES)
    if 'passed_qc' in a.obs.columns:
        qc=is_true(a.obs['passed_qc'])
        if qc.any(): valid=valid & qc
        else: raise ValueError(f'{region}: passed_qc exists but no cell is marked TRUE; check the QC coding before analysis')
    groups=pd.DataFrame({'donor':donors.loc[valid].to_numpy(),'cell_type':ctypes.loc[valid].to_numpy()})
    if groups.empty: raise ValueError(f'{region}: zero eligible cells after genotype/clinical/QC matching')
    groups=groups.drop_duplicates().sort_values(['cell_type','donor']).reset_index(drop=True)
    groups['group_idx']=np.arange(len(groups),dtype=int)
    index_map={(r.donor,r.cell_type):int(r.group_idx) for r in groups.itertuples(index=False)}
    group_idx=np.full(a.n_obs,-1,dtype=np.int32)
    for i in np.flatnonzero(np.asarray(valid,dtype=bool)):
        group_idx[i]=index_map[(donors.iloc[i],ctypes.iloc[i])]
    accum=np.zeros((len(groups),len(genes)),dtype=np.int64)
    n_cells=np.bincount(group_idx[group_idx>=0],minlength=len(groups))
    print(f'[REGION] {region}: input cells={a.n_obs}, eligible cells={len(group_idx[group_idx>=0])}, groups={len(groups)}, original genes={a.n_vars}, unique symbols={len(genes)}',flush=True)
    for i,(block,start,end) in enumerate(a.chunked_X(chunk_size=chunk_size)):
        local=group_idx[start:end]
        locidx=np.flatnonzero(local>=0)
        if not len(locidx): continue
        mat=block[locidx,:]
        if not sp.issparse(mat): mat=sp.csr_matrix(mat)
        else: mat=mat.tocsr()
        if mat.nnz:
            if mat.data.min()<0 or not np.allclose(mat.data[:min(10000,mat.nnz)],np.rint(mat.data[:min(10000,mat.nnz)]),atol=1e-5):
                raise ValueError(f'{region}: X is not raw nonnegative integer-like counts. Check the H5AD data layer!')
        m=sp.csr_matrix((np.ones(len(locidx),dtype=np.int8),
                        (local[locidx],np.arange(len(locidx)))),shape=(len(groups),len(locidx)))
        agg=(m@mat@projection).tocoo()
        np.add.at(accum,(agg.row,agg.col),np.rint(agg.data).astype(np.int64))
        if i and i%20==0: print(f'[PROGRESS] {region}: cells processed {end:,}/{a.n_obs:,}',flush=True)
    a.file.close()
    groups['n_cells']=n_cells
    groups=groups.merge(eligible,on='donor',how='left',validate='many_to_one')
    groups['library_size']=accum.sum(axis=1)
    groups['region']=region
    keep=groups['n_cells'].ge(min_cells) & groups.library_size.gt(0)
    audit={ 'region':region,'input_n_cells':int(len(group_idx)), 'matched_n_cells':int(valid.sum()),
            'n_unique_donors':int(groups.donor.nunique()),'n_groups_before_min_cells':int(len(groups)),
            'n_groups_kept':int(keep.sum()),'n_genes_before':int(len(raw_names)),
            'n_unique_gene_symbols':int(len(genes)),
            'n_collapsed_or_unannotated_genes':int(len(raw_names)-len(genes)),
            'n_cell_groups_below_min_cells':int((~keep).sum()),'min_cells':min_cells }
    regout=out/'pseudobulk'/region
    regout.mkdir(parents=True,exist_ok=True)
    sp.save_npz(regout/'counts.npz',sp.csr_matrix(accum[keep.to_numpy()]))
    pd.DataFrame({'gene':genes}).to_csv(regout/'genes.csv',index=False)
    groups.loc[keep].drop(columns='group_idx').to_csv(regout/'samples.csv',index=False)
    groups.to_csv(regout/'all_group_audit.csv',index=False)
    (regout/'audit.json').write_text(json.dumps(audit,indent=2))
    print('[REGION COMPLETE]',audit,flush=True)
    return audit

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    p.add_argument('--h5ad-dir',default='/public2/data/duwenxuan/AMP_PD_anno/try')
    p.add_argument('--clinical',default='/public2/data/luojunfeng/2sTGCS_binary/scQTL/AMP_PD_Postmortem_Brain_2024_8_30_dwx.csv')
    p.add_argument('--plink-prefix',default='/public/labdata/public_data/AMP_PD_Postmortem_Brain/vcf/merged.100s.new.fixed')
    p.add_argument('--genotype-raw',default='',help='Optional pre-exported PLINK --recode A .raw; skips BED/BIM extraction')
    p.add_argument('--tqtl-ppmi-bim',default='',help='Stage-I tQTL PPMI BIM for rsID locations')
    p.add_argument('--tqtl-pdbp-bim',default='',help='Stage-I tQTL PDBP BIM for rsID locations')
    p.add_argument('--variant-map',default='',help='Optional SNP coordinate TSV (rsid,chrom,pos,build,ref,alt)')
    p.add_argument('--reference-build',choices=('auto','GRCh37','GRCh38'),default='auto')
    p.add_argument('--offline',action='store_true',help='Do not query Ensembl; requires --variant-map for rs35162690 if absent from BIM')
    p.add_argument('--allow-position-only',action='store_true',help='Allow coordinate-only SNP matches lacking allele verification (discouraged)')
    p.add_argument('--plink',default='/usr/bin/plink1.9')
    p.add_argument('--outdir',default='')
    p.add_argument('--regions',nargs='+',choices=REGIONS,default=list(REGIONS))
    p.add_argument('--chunk-size',type=int,default=3000)
    p.add_argument('--min-cells',type=int,default=30)
    a=p.parse_args()
    out=Path(a.outdir) if a.outdir else Path(a.root)/'single_iv_causal'/'postmortem_snp_scDE'
    out.mkdir(parents=True,exist_ok=True)
    cli=prepare_clinical(a.clinical,out)
    geno=make_genotype(a.plink_prefix,a.genotype_raw,out,a.plink,a.variant_map,a.reference_build,a.offline,a.allow_position_only,
                       a.tqtl_ppmi_bim,a.tqtl_pdbp_bim)
    joint=cli.merge(geno,on='donor',how='inner',validate='one_to_one')
    joint.to_csv(out/'clinical_genotype_matched.csv',index=False)
    print('[MATCH] Case/Control eligible clinical:',len(cli),'genotype donors:',len(geno),'matched:',len(joint),flush=True)
    for snp in TARGET_SNPS:
        g=joint[snp].dropna()
        print('[DOSAGE]',snp,'valid_n',len(g),'hardcall counts',g.round(4).value_counts().head(10).to_dict(),flush=True)
    if not len(joint): raise ValueError('No clinical/genotype donors matched')
    audit=[]
    for region in a.regions:
        fname=Path(a.h5ad_dir)/f'{region}_subtype_count_try.h5ad'
        if not fname.is_file():raise FileNotFoundError(fname)
        audit.append(process_region(region,fname,joint,out,a.chunk_size,a.min_cells))
    pd.DataFrame(audit).to_csv(out/'regions_preparation_audit.csv',index=False)
    print('[DONE] pseudobulk output:',out,flush=True)

if __name__=='__main__':main()
