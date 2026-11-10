#!/usr/bin/env python3
"""AD MMSE Stage2: retain every available validated Stage1 IV with identical A1 dosage.

No additional MAF, HWE, genotype missingness, or LD QC is permitted for the IVs.
Ancestry PC QC happens independently via 04_prepare_ad_mmse_genomewide_pcs.py.
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
from pathlib import Path
import pandas as pd

BIM_COLS=['chrom','snp','cm','bp','a1','a2']
FAM_COLS=['FID','IID','PAT','MAT','SEX','PHENO']

def read_cov(path):
    x=pd.read_csv(path,low_memory=False)
    x=x.drop(columns=[c for c in x.columns if str(c).startswith('Unnamed:')],errors='ignore')
    for key in ('cohort','participant_id','time'):
        if key not in x:raise ValueError(f'Missing phenotype column {key}')
    x['cohort']=x.cohort.astype(str)
    x['participant_id']=x.participant_id.astype(str)
    return x

def read_fam(bf):
    return pd.read_csv(str(bf)+'.fam',sep=r'\s+',names=FAM_COLS,header=None,dtype=str)

def read_bim(bf):
    z=pd.read_csv(str(bf)+'.bim',sep=r'\s+',names=BIM_COLS,header=None,dtype=str)
    if z.snp.duplicated().any():
        raise RuntimeError(f'Duplicate variant IDs in {bf}.bim; cannot track 1-to-1 allele directions')
    return z

def match(cov,fam):
    f_idx={str(v):i for i,v in enumerate(fam.FID)}
    i_idx={str(v):i for i,v in enumerate(fam.IID)}
    rows=[]
    for pid in cov.participant_id.drop_duplicates().astype(str):
        if pid in i_idx:r=fam.iloc[i_idx[pid]];how='IID'
        elif pid in f_idx:r=fam.iloc[f_idx[pid]];how='FID'
        else:
            rows.append(dict(participant_id=pid,genotype_match='missing',FID='',IID=''))
            continue
        rows.append(dict(participant_id=pid,genotype_match=how,FID=str(r.FID),IID=str(r.IID)))
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--analysis-name',default='ad_mmse')
    p.add_argument('--phenotype-cov',required=True)
    p.add_argument('--candidate-file',required=True)
    p.add_argument('--raw-bfile',required=True)
    p.add_argument('--plink',default='plink1.9')
    a=p.parse_args()
    plink=shutil.which(a.plink)
    if not plink:raise FileNotFoundError(f'PLINK not found on PATH: {a.plink}')
    root=Path(a.root);out=root/'single_iv_causal'/a.analysis_name
    gdir=out/'genotype';gdir.mkdir(parents=True,exist_ok=True)
    cand=pd.read_csv(a.candidate_file,low_memory=False)
    if not {'gene','lead_snp','artifact_id'}.issubset(cand.columns):
        raise ValueError('Frozen candidate file must have gene, lead_snp, artifact_id')
    cov=read_cov(a.phenotype_cov)
    fam=read_fam(a.raw_bfile)
    matched=match(cov,fam)
    cc=cov[['cohort','participant_id']].drop_duplicates().merge(matched,on='participant_id',how='left',validate='many_to_one')
    cc.to_csv(out/'participant_genotype_match.csv',index=False)
    summ=cc.groupby('cohort').agg(participants=('participant_id','nunique'),genotype_matched=('genotype_match',lambda v:v.ne('missing').sum()),genotype_missing=('genotype_match',lambda v:v.eq('missing').sum())).reset_index()
    summ.to_csv(out/'participant_overlap_summary.csv',index=False)
    snps=set(cand.lead_snp.dropna().astype(str))
    if not snps:raise RuntimeError('Candidate file contains no Stage1 IVs')
    raw_bim=read_bim(a.raw_bfile)
    available=set(raw_bim.snp)
    absent=sorted(snps-available)
    if absent:
        (gdir/'missing_stage1_IVs.txt').write_text('\n'.join(absent)+'\n')
        raise RuntimeError(f'{len(absent)} frozen candidate SNPs absent from merged, harmonized input. '
                           f'No post-Stage1 IV filtering allowed; investigate {gdir/"missing_stage1_IVs.txt"}')
    names=sorted(snps)
    (gdir/'needed_single_iv_snps.txt').write_text('\n'.join(names)+'\n')
    keep=cc[cc.genotype_match.ne('missing')][['FID','IID']].drop_duplicates()
    if keep.empty:raise RuntimeError('No MMSE participants matched genotype FAM')
    keep.to_csv(gdir/'keep_single_iv_stage2.txt',sep='\t',header=False,index=False)
    pref=gdir/'single_iv_snps'
    # No --mind, --geno, --maf, --hwe, --indep-pairwise, or other new IV QC.
    # --keep filters participants to the MMSE cohort; --extract is limited to
    # eligible Stage1 IVs; --keep-allele-order preserves the Stage1-aligned A1.
    cmd=[plink,'--bfile',a.raw_bfile,
         '--keep',str(gdir/'keep_single_iv_stage2.txt'),
         '--extract',str(gdir/'needed_single_iv_snps.txt'),
         '--keep-allele-order','--make-bed','--out',str(pref)]
    print('[STAGE2 IV] NO additional IV QC. Command:', ' '.join(cmd),flush=True)
    with (gdir/'extract_frozen_IVs.log').open('w') as log:
        r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:
        log=(gdir/'extract_frozen_IVs.log').read_text(errors='replace').splitlines()
        raise RuntimeError(f'PLINK failed: {r.returncode}\n'+ '\n'.join(log[-45:]))
    extracted=read_bim(pref)
    source=raw_bim[raw_bim.snp.isin(snps)].set_index('snp')
    target=extracted.set_index('snp')
    if set(target.index)!=snps or len(target)!=len(snps):
        raise RuntimeError(f'IV loss detected during Stage2 extraction: requested={len(snps)} extracted={len(target)}. '
                           'No QC applied; investigate variant coding/PLINK log')
    key_fields=['chrom','bp','a1','a2']
    changed={}
    for k in key_fields:
        left=source.loc[names,k].astype(str).str.upper()
        right=target.loc[names,k].astype(str).str.upper()
        wrong=left.ne(right)
        if wrong.any():changed[k]=left.index[wrong].tolist()[:15]
    if changed:
        (gdir/'stage1_IV_direction_or_position_mismatch.json').write_text(json.dumps(changed,indent=2))
        raise RuntimeError(f'Frozen IV position/allele orientation changed during Stage2 extraction: {changed}')
    output_fam=read_fam(pref)
    if len(output_fam)!=len(keep):
        raise RuntimeError(f'Subject count mismatch after Stage2 IV extraction: {len(output_fam)} vs {len(keep)}')
    manifest={'analysis_name':a.analysis_name,'phenotype_cov':a.phenotype_cov,
              'candidate_file':a.candidate_file,'raw_bfile':a.raw_bfile,
              'source':'Frozen validated Stage1 IVs only; no Stage2 re-QC',
              'IV_QC_applied':False,'IV_filtering':{'mind':False,'geno':False,'maf':False,'hwe':False,'LD_pruning':False},
              'sample_selection':'keep participants with matched MMSE phenotype',
              'n_candidate_pairs':int(len(cand)),'n_candidate_genes':int(cand.gene.nunique()),
              'n_requested_frozen_IVs':int(len(snps)),'n_extracted_frozen_IVs':int(len(target)),
              'n_genotype_participants':int(len(output_fam)),
              'A1_allele_preserved':True,'A2_allele_preserved':True,
              'single_iv_bfile':str(pref)}
    (out/'single_iv_input_audit.json').write_text(json.dumps(manifest,indent=2))
    print('[IV PRESERVED]',f'{len(snps)}/{len(snps)} frozen IVs, A1/A2 unchanged; no IV re-QC',flush=True)
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':main()
