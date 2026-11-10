#!/usr/bin/env python3
"""AD MMSE Stage2: ancestry PCs from independent cohort genome-wide PLINK genotypes.

Never calculate PCs from the merged Stage2 candidate-IV-only genotype dataset.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

def normalize_id(s, cohort):
    s=str(s).strip()
    if cohort=='MAP':
        m=re.search(r'MAP\s*(\d+)',s,flags=re.I)
        if m:return m.group(1)
        return s if re.fullmatch(r'\d+',s) else ''
    m=re.search(r'\d{3}_S_\d{4}',s,flags=re.I)
    return m.group(0) if m else s

def read_fam(prefix):
    path=Path(str(prefix)+'.fam')
    if not path.is_file():raise FileNotFoundError(f'Genotype file missing: {path}')
    return pd.read_csv(path,sep=r'\s+',header=None,dtype=str,names=['FID','IID','PAT','MAT','SEX','PHENO'])

def read_cohort_participants(path):
    x=pd.read_csv(path,dtype=str,keep_default_na=False)
    for k in ['cohort','participant_id','genotype_match']:
        if k not in x.columns:raise ValueError(f'Missing {k} in {path}')
    return x.loc[x.genotype_match.ne('missing'),['cohort','participant_id']].drop_duplicates()

def map_genome_samples(fam,expected,cohort):
    x=fam[['FID','IID']].copy()
    valid=set(map(str,expected))
    from_iid=x.IID.map(lambda s:normalize_id(s,cohort))
    from_fid=x.FID.map(lambda s:normalize_id(s,cohort))
    # The original TOPMed FAM can encode the canonical MAP<projid> in FID,
    # while older filtered files sometimes encode it in IID.
    x['participant_id']=from_iid.where(from_iid.isin(valid),from_fid.where(from_fid.isin(valid),''))
    x=x.loc[x.participant_id.isin(valid)].copy()
    if x.participant_id.duplicated().any():
        dup=x.loc[x.participant_id.duplicated(False),'participant_id'].unique().tolist()[:10]
        raise ValueError(f'{cohort}: duplicate normalized IDs in genome-wide FAM: {dup}')
    absent=sorted(set(expected)-set(x.participant_id))
    print(f'[GENOME FAM] {cohort}: genome-wide FAM={len(fam)}, matched={len(x)}/{len(expected)}; missing={len(absent)}; first_missing={absent[:6]}',flush=True)
    return x

def call(cmd, log):
    print('+',' '.join(map(str,cmd)),flush=True)
    p=subprocess.run([str(q) for q in cmd],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    Path(log).write_text(p.stdout or '')
    if p.returncode:
        tail='\n'.join((p.stdout or '').splitlines()[-55:])
        raise RuntimeError(f'PLINK/twstats exit={p.returncode}; log={log}\n{tail}')

def twstats_read(path,max_pcs):
    rows=[]
    for line in Path(path).read_text(errors='replace').splitlines():
        z=line.strip().split()
        if len(z)<5:continue
        try:
            ix=int(float(z[0]));vals=[float(v) for v in z[1:]]
        except (ValueError,OverflowError):continue
        if 1<=ix<=max_pcs and len(vals)>=4:rows.append((ix,vals[3]))
    if not rows:raise ValueError(f'Cannot parse Tracy-Widom test values from {path}; inspect file before proceeding')
    return {i:p for i,p in rows}

def select_pcs(pmap,max_pcs,alpha,min_pcs):
    selected=[]
    for ix in range(1,max_pcs+1):
        if ix not in pmap or not np.isfinite(pmap[ix]) or pmap[ix]>=alpha:break
        selected.append(ix)
    # Predetermined minimum: prevent an empty genetic ancestry adjustment.
    selected=list(range(1,max(len(selected),min(min_pcs,max_pcs))+1))
    return selected

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',required=True)
    ap.add_argument('--analysis-name',default='ad_mmse')
    ap.add_argument('--map-genome-bfile',required=True)
    ap.add_argument('--adni-genome-bfile',required=True)
    ap.add_argument('--plink',default='plink1.9')
    ap.add_argument('--twstats',default='twstats')
    ap.add_argument('--twtable',default='twtable')
    ap.add_argument('--max-pcs',type=int,default=20)
    ap.add_argument('--minimum-pcs',type=int,default=5)
    ap.add_argument('--alpha',type=float,default=.05)
    ap.add_argument('--mind',type=float,default=.05)
    ap.add_argument('--geno',type=float,default=.05)
    ap.add_argument('--maf',type=float,default=.05)
    ap.add_argument('--hwe',type=float,default=1e-6)
    ap.add_argument('--prune-window',type=int,default=50)
    ap.add_argument('--prune-step',type=int,default=5)
    ap.add_argument('--prune-r2',type=float,default=.2)
    a=ap.parse_args()
    if not 1<=a.minimum_pcs<=a.max_pcs:ap.error('minimum-pcs must be in [1, max-pcs]')
    plink=shutil.which(a.plink)
    twstats=shutil.which(a.twstats)
    if not plink:raise FileNotFoundError(f'plink1.9 executable not found: {a.plink}')
    if not twstats:raise FileNotFoundError(f'twstats executable not found: {a.twstats}')
    sources={'MAP':a.map_genome_bfile,'ADNI':a.adni_genome_bfile}
    # Check both sources up-front; do not silently substitute IV-only merged genotypes.
    for coh,bf in sources.items():
        for ext in ('.bed','.bim','.fam'):
            if not Path(str(bf)+ext).exists():
                raise FileNotFoundError(f'{coh} genome-wide genotype absent: {bf}{ext}. Supply the actual original genome-wide PLINK prefix with --{coh.lower()}-genome-bfile.')
    root=Path(a.root)
    adir=root/'single_iv_causal'/a.analysis_name
    inputmatch=adir/'participant_genotype_match.csv'
    if not inputmatch.exists():raise FileNotFoundError(inputmatch)
    participants=read_cohort_participants(inputmatch)
    out=adir/'genotype_pcs';out.mkdir(parents=True,exist_ok=True)
    combined=[];cohort_results={}
    for cohort,bfile in sources.items():
        ids=participants.loc[participants.cohort.eq(cohort),'participant_id'].astype(str).tolist()
        fam=read_fam(bfile)
        matched=map_genome_samples(fam,ids,cohort)
        if len(matched)<30:raise RuntimeError(f'{cohort}: only {len(matched)} matched genome-wide samples. Do not use IV-only data as PC fallback; investigate genome FAM mapping.')
        cdir=out/cohort;cdir.mkdir(parents=True,exist_ok=True)
        matched.to_csv(cdir/'genomewide_sample_match.csv',index=False)
        keep=cdir/'genomewide_keep.txt'
        matched[['FID','IID']].to_csv(keep,sep='\t',index=False,header=False)
        qc=cdir/f'{cohort}.genomewide.qc'
        # --autosome avoids non-autosomal genotype/sex handling for ancestry PCA.
        call([plink,'--bfile',bfile,'--keep',keep,'--autosome','--mind',a.mind,'--geno',a.geno,'--maf',a.maf,'--hwe',a.hwe,'--make-bed','--out',qc],cdir/'01_genomewide_qc.log')
        qfam=read_fam(qc)
        qpeople=set(zip(qfam.FID,qfam.IID))
        matched=matched[[tuple(k) in qpeople for k in zip(matched.FID,matched.IID)]].copy()
        if len(matched)<30:raise RuntimeError(f'{cohort}: {len(matched)} genome-wide participants survive QC; inspect {cdir/"01_genomewide_qc.log"}')
        pr=cdir/f'{cohort}.genomewide.prune'
        call([plink,'--bfile',qc,'--indep-pairwise',a.prune_window,a.prune_step,a.prune_r2,'--out',pr],cdir/'02_genomewide_ld.log')
        prune_file=Path(str(pr)+'.prune.in')
        if not prune_file.exists() or prune_file.stat().st_size==0:raise RuntimeError(f'{cohort}: no variants for PCA after LD pruning')
        npruned=sum(1 for _ in prune_file.open())
        pca=cdir/f'{cohort}.genomewide.pca'
        npcs=min(a.max_pcs,len(matched)-1)
        call([plink,'--bfile',qc,'--extract',prune_file,'--pca',npcs,'--out',pca],cdir/'03_genomewide_pca.log')
        twout=cdir/f'{cohort}.genomewide.twstats.out'
        call([twstats,'-t',a.twtable,'-i',str(pca)+'.eigenval','-o',twout],cdir/'04_genomewide_twstats.log')
        pmap=twstats_read(twout,npcs)
        sel=select_pcs(pmap,npcs,a.alpha,a.minimum_pcs)
        pd.DataFrame({'PC':list(pmap),'twstats_p':[pmap[i] for i in pmap]}).to_csv(cdir/f'{cohort}.genomewide.twstats.parsed.csv',index=False)
        names=['FID','IID']+[f'gPC{i}' for i in range(1,npcs+1)]
        pcs=pd.read_csv(str(pca)+'.eigenvec',sep=r'\s+',header=None,names=names,dtype={'FID':str,'IID':str})
        for col in names[2:]:pcs[col]=pd.to_numeric(pcs[col],errors='coerce')
        joined=matched[['FID','IID','participant_id']].merge(pcs,on=['FID','IID'],how='inner',validate='one_to_one')
        joined.insert(0,'cohort',cohort)
        if joined.participant_id.nunique()!=len(matched):raise RuntimeError(f'{cohort}: some QC retained participants absent from PCA eigenvec')
        joined.to_csv(cdir/f'{cohort}.all_pcs.csv',index=False)
        combined.append(joined)
        cohort_results[cohort]={
            'n_input_participants':len(ids),'n_genomewide_fam_matched':int(len(keep.read_text().splitlines())),
            'n_after_qc':int(len(matched)),'n_pca_participants':int(len(joined)),
            'n_pca_components_calculated':npcs,'selected_n_pcs':len(sel),
            'selected_pc_indices':sel,'selected_columns':[f'gPC{i}' for i in sel],
            'selection_rule':f'genomewide_autosomal_LDpruned_TW_consecutive_with_predeclared_minimum_{a.minimum_pcs}',
            'n_ld_pruned_variants':npruned,'genomewide_bfile':str(bfile),
            'twstats_output':str(twout),'eigenvec':str(pca)+'.eigenvec'}
        print(f'[GENOMEWIDE PCA] {cohort}: n_people={len(joined)} n_pruned_SNP={npruned} TW-leading + min={a.minimum_pcs} selected_PCs={sel}',flush=True)
    pc=pd.concat(combined,ignore_index=True)
    if pc.duplicated(['cohort','participant_id']).any():raise RuntimeError('Duplicate subject keys in combined PCs')
    pcpath=out/'single_iv_genotype_pcs.csv'
    pc.to_csv(pcpath,index=False)
    manifest={'analysis_name':a.analysis_name,'source':'separate cohort high-density genome-wide genotype; not the merged single-IV genotype file',
              'cohorts':cohort_results,'combined_pc_file':str(pcpath),
              'genotype_QC':{'mind':a.mind,'geno':a.geno,'maf':a.maf,'hwe':a.hwe,'ld_r2':a.prune_r2},
              'min_pcs':a.minimum_pcs,'twstats_alpha':a.alpha}
    (out/'pc_selection.json').write_text(json.dumps(manifest,indent=2))
    print('[DONE] Genotype PCs from genome-wide MAP and ADNI source data:',pcpath,flush=True)

if __name__=='__main__':main()
