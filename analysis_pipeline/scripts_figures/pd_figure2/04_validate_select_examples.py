#!/usr/bin/env python3
"""Validate actual Stage-II component and outcome rows BEFORE choosing manuscript examples.

No requirement for a specific UPDRS3/UPDRS2/MoCA triplet when the real A-tier
candidate pool doesn't contain one. Prefer diverse phenotypes and significant
A-tier, then high MAF, topical genes, and stable genotype group sizes.
"""
import argparse, json
from pathlib import Path
import pandas as pd
from component_trajectories import load_example_from_stage2
from helpers_pd_figure2 import read_publication_table

def choose(args):
    rank=pd.read_csv(args.rank)
    pub=read_publication_table(args.table)
    # Match by immutable triple, never gene+phenotype alone.
    k=['gene','lead_snp','phenotype']
    rank['phenotype']=rank.pheno_norm.astype(str)
    cand=rank.merge(pub,on=k,how='inner',validate='many_to_many',suffixes=('','_publication'))
    cand=cand.sort_values('rank_score',ascending=False).drop_duplicates(k)
    good=[];audit=[]
    for row in cand.head(args.max_to_check).to_dict('records'):
        if str(row['phenotype']) not in ('updrs1','updrs2','updrs3','moca'):continue
        try:
            z,meta=load_example_from_stage2(args.root,pd.Series(row),cohort=args.cohort)
            counts=meta['genotype_counts']
            if sum(counts.values())<args.min_usable_donors:
                raise ValueError('Too few observed phenotype participants')
            if counts.get(1,0)<args.min_heterozygotes:
                raise ValueError(f'Heterozygotes n={counts.get(1,0)} < {args.min_heterozygotes}')
            # n2 is descriptive, avoid mandatory n>=5 if dataset cannot support it.
            score=float(row['rank_score']) + min(counts.get(2,0),20)*0.18
            row['_selection_score']=score
            good.append(row)
            audit.append(dict(gene=row['gene'],variant=row['lead_snp'],phenotype=row['phenotype'],status='PASS',n=meta['n_participants'],counts=counts,maf=row['MAF']))
        except Exception as exc:
            audit.append(dict(gene=row['gene'],variant=row['lead_snp'],phenotype=row['phenotype'],status='FAIL',reason=str(exc)[:180]))
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(audit).to_csv(str(Path(args.output).with_suffix('.candidate_audit.csv')),index=False)
    good.sort(key=lambda x:x['_selection_score'],reverse=True)
    chosen=[];seen=set();ph=set()
    # Maximize phenotype diversity first, with A-tier and MAF prioritized in ranked list.
    for pass_ in (1,2):
        for r in good:
            pair=(str(r['gene']),str(r['lead_snp']),str(r['phenotype']))
            if pair in seen or str(r['gene']) in {str(z['gene']) for z in chosen}:continue
            if pass_==1 and r['phenotype'] in ph:continue
            chosen.append(r);seen.add(pair);ph.add(r['phenotype'])
            if len(chosen)==3:break
        if len(chosen)==3:break
    if len(chosen)<3:
        raise RuntimeError(f'Only {len(chosen)} validated examples (from {len(good)} viable candidates, checked {len(audit)}). See candidate_audit.csv. Consider lowering MIN_MAF to 0.03 or broadening examples to B/C tiers; no made-up trajectory will be drawn.')
    out=pd.DataFrame(chosen).drop(columns=['_selection_score'])
    out.to_csv(args.output,index=False)
    print('[FIG2 VALIDATED EXACT SNP EXAMPLES]')
    print(out[['gene','lead_snp','phenotype','MAF','evidence_class','rank_score']].to_string(index=False))
    return out

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--root',required=True);a.add_argument('--rank',required=True);a.add_argument('--table',required=True)
    a.add_argument('--output',required=True);a.add_argument('--cohort',default='PP')
    a.add_argument('--max-to-check',type=int,default=120)
    a.add_argument('--min-heterozygotes',type=int,default=8)
    a.add_argument('--min-usable-donors',type=int,default=60)
    choose(a.parse_args())
if __name__=='__main__':main()
