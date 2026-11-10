#!/usr/bin/env python3
"""Select overall CTAR temporal tQTLs from a PRE-REPLICATED pool.

Modes:
  representative: random 5k from Stage-I eligible (PP BH, PD P<.05,
                  allele-harmonized trajectory-r positive, PP LD pruning).
  phenotype-final: ALL unique IV-gene pairs appearing as primary candidates
                  in the three UPDRS final master files; NO random sampling.

Stage-I PPMI BH raw-P cutoff is computed from the WHOLE significant QTL table,
not the selected subset. This file does not rerun any Stage-I or Stage-II model.
"""
import argparse, json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from base_prepare_original import (GENE,VAR,findcol,iterate,columns,match_rep)

KEY=['gene','variant']
PHENOS=('updrs1','updrs2','updrs3')

def yes(s):return s.astype(str).str.lower().str.strip().isin(('true','1','yes'))

def load_pool(root,mode):
    stage=root/'single_iv_causal/iv_validation/single_iv_primary_eligible.csv'
    if not stage.is_file():raise FileNotFoundError(f'Missing Stage-I eligible IVs: {stage}')
    st=pd.read_csv(stage,low_memory=False,dtype={'gene':str,'lead_snp':str})
    if 'iv_primary_eligible' in st:st=st.loc[yes(st.iv_primary_eligible)].copy()
    if not {'gene','lead_snp'}.issubset(st):raise ValueError('Stage-I eligible IV list requires gene + lead_snp')
    st=st.rename(columns={'lead_snp':'variant'})
    if st.duplicated(KEY).any():raise ValueError('Stage-I eligible contains duplicate gene+variant')
    if mode=='representative':return st[KEY].copy(),str(stage),None
    pieces=[];paths={}
    for ph in PHENOS:
        base=root/'single_iv_causal'/ph/'standard_final'
        file=base/f'{ph}_FINAL_causal_IV_gene_master_FULL.csv'
        if not file.is_file():file=base/f'{ph}_FINAL_causal_IV_gene_master.csv'
        if not file.is_file():raise FileNotFoundError(f'{ph} final master missing: {file}')
        d=pd.read_csv(file,low_memory=False,dtype={'gene':str,'lead_snp':str})
        if not {'gene','lead_snp','primary_downstream_candidate'}.issubset(d):
            raise ValueError(f'{file}: gene/lead_snp/primary_downstream_candidate required')
        d=d.loc[yes(d.primary_downstream_candidate),['gene','lead_snp']].copy()
        if d.duplicated(['gene','lead_snp']).any():raise ValueError(f'Duplicate clinical IV: {file}')
        pieces.append(d.rename(columns={'lead_snp':'variant'}));paths[ph]=str(file)
    target=pd.concat(pieces,ignore_index=True).drop_duplicates(KEY)
    chk=target.merge(st[KEY],on=KEY,how='left',indicator=True)
    missing=chk.loc[chk._merge.ne('both'),KEY]
    if len(missing):raise ValueError(f'{len(missing)} final clinical candidates absent from Stage-I eligible IVs: {missing.head(10).to_dict("records")}')
    return target.reset_index(drop=True),str(stage),paths

def read_discovery(path,pool):
    schema=columns(path);g=findcol(schema,GENE);v=findcol(schema,VAR)
    p=findcol(schema,('Pvalue','overall_pvalue','temporal_pvalue'))
    f=findcol(schema,('FDR','overall_FDR','qvalue'),False)
    rho=findcol(schema,('rho','rho_ref'),False)
    lrt=findcol(schema,('LRT','overall_LRT'),False)
    nc=findcol(schema,('n_covariates_used',),False)
    use=list(dict.fromkeys(x for x in (g,v,p,f,rho,lrt,nc) if x))
    keys=set(zip(pool.gene.astype(str),pool.variant.astype(str)))
    variants=set(pool.variant.astype(str))
    pieces=[];nall=0;maxraw=0.
    for chunk in iterate(path,use):
        x=pd.to_numeric(chunk[p],errors='coerce')
        mask=x.between(0,1,inclusive='both')
        if f is not None:mask &= pd.to_numeric(chunk[f],errors='coerce').lt(.05)
        elif 'significant_tqtl' not in str(path):raise ValueError('Missing FDR in unscreened discovery results')
        if mask.any():
            nall+=int(mask.sum());maxraw=max(maxraw,float(x[mask].max()))
        z=chunk.loc[mask & chunk[v].astype(str).isin(variants)].copy()
        if not z.empty:
            z=z.loc[[pair in keys for pair in zip(z[g].astype(str),z[v].astype(str))]]
            if len(z):pieces.append(z)
    if not pieces:raise ValueError('None of the eligible pairs were present in PPMI significant results')
    selected=pd.concat(pieces,ignore_index=True).rename(columns={g:'gene',v:'variant',p:'pvalue',**({f:'fdr'} if f else {}),**({rho:'rho'} if rho else {}),**({lrt:'lrt'} if lrt else {}),**({nc:'n_covariates_used'} if nc else {})})
    selected[['gene','variant']]=selected[['gene','variant']].astype(str)
    if selected.duplicated(KEY).any():raise ValueError('Duplicated PPMI discovery gene+variant')
    out=pool.merge(selected,on=KEY,how='left',validate='one_to_one',indicator=True)
    if not out._merge.eq('both').all():
        miss=out.loc[out._merge.ne('both'),KEY].head(10)
        raise ValueError(f'{len(out)-out._merge.eq("both").sum()} candidate IVs lack PPMI significant results: {miss.to_dict("records")}')
    return out.drop(columns='_merge'),int(nall),maxraw

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    p.add_argument('--mode',choices=['representative','phenotype-final'],default='representative')
    p.add_argument('--n-pairs',type=int,default=5000)
    p.add_argument('--seed',type=int,default=20261030)
    p.add_argument('--outdir',required=True)
    a=p.parse_args();root=Path(a.root);out=Path(a.outdir)
    if (out/'locus_bootstrap_results.csv').exists() or (out/'_shards').exists():
        raise FileExistsError(f'{out} contains power fits; choose a NEW outdir to change sampling universe')
    pool,source,clinical=load_pool(root,a.mode)
    if a.mode=='representative':
        if a.n_pairs>len(pool):raise ValueError(f'Requested {a.n_pairs} from eligible population {len(pool)}')
        pool=pool.sample(n=a.n_pairs,random_state=a.seed).reset_index(drop=True)
    disc=root/'qtl/PP/submitted_followup_ctar_mptga/final/significant_tqtl.parquet'
    rep=root/'qtl/PD/submitted_followup_ctar_mptga/final/all_results.parquet'
    selected,nall,bh=read_discovery(disc,pool)
    selected=match_rep(rep,selected)
    unmatched=selected.rep_match.ne('both')
    not_sig=~selected.pd_nominal_replicated
    if unmatched.any() or not_sig.any():
        # Fail instead of silently changing the promised 5,000 or mixing unreplicated loci.
        bad=selected.loc[(unmatched|not_sig),KEY+['pd_pvalue','rep_match']].head(12)
        raise RuntimeError(f'Frozen Stage-I eligible IVs disagree with current PDBP results: {int(unmatched.sum())} unmatched, {int(not_sig.sum())} without PD P<0.05. Examples: {bad.to_dict("records")}')
    out.mkdir(parents=True,exist_ok=True)
    selected.to_csv(out/'selected_loci.csv',index=False)
    audit={'mode':a.mode,'source_pool':source,'clinical_sources':clinical,
           'population_stage1_eligible':None if a.mode=='phenotype-final' else len(pd.read_csv(source,usecols=['gene'])),
           'discovery_file':str(disc),'replication_file':str(rep),'total_significant':nall,
           'sampled':len(selected),'selected_PPMI':len(selected),'matched_PDBP':len(selected),
           'pdbp_nominal_replicated':int(selected.pd_nominal_replicated.sum()),
           'observed_max_raw_p_of_all_hits':bh,'random_seed':a.seed,
           'selection_conditioning':'All sampled pairs passed PPMI BH, PDBP nominal P<0.05, Stage-I trajectory direction and PPMI LD pruning. Bootstrap PD power is conditional on this selection.'}
    (out/'selection_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
    print(json.dumps(audit,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
