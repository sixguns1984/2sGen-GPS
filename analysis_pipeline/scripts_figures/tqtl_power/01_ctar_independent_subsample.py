#!/usr/bin/env python3
"""Independent PDBP replication versus sample size for PPMI-discovered CTAR tQTLs.

Crucial design: candidate pairs depend ONLY on PPMI significant-tQTL file.
PDBP is neither used to select candidates nor to determine significance thresholds.
50/75/90% -> participant-level WITHOUT-replacement subsamples; 100% -> EXACT
full cohort, once. Original model and covariates, without new IV QC.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd

METHOD='submitted_followup_ctar_mptga'


def fingerprint(path: Path) -> str:
    s=path.stat()
    return f'{s.st_size}:{s.st_mtime_ns}'


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def scan_pdbp_bim_for_snps(bim: Path, targets: set[str]) -> set[str]:
    """Existence only, never use PDBP association results or re-QC SNPs."""
    if not bim.is_file():raise FileNotFoundError(bim)
    found=set()
    for chunk in pd.read_csv(bim,sep=r'\s+',header=None,usecols=[1],dtype=str,chunksize=250000):
        found.update(set(chunk.iloc[:,0]) & targets)
        if len(found)==len(targets):break
    return found


def select_ppmi_pairs(pp: pd.DataFrame, n_pairs: int, seed: int,
                      available_snps: set[str] | None = None) -> tuple[pd.DataFrame,int,int]:
    required={'Variant_ID','Ensembl_ID','Pvalue','FDR','rho'}
    if not required.issubset(pp.columns):
        raise RuntimeError(f'PPMI significant file missing {sorted(required-set(pp.columns))}')
    z=pp[list(required)].copy()
    z['Variant_ID']=z['Variant_ID'].astype(str)
    z['Ensembl_ID']=z['Ensembl_ID'].astype(str)
    for c in ['Pvalue','FDR','rho']:
        z[c]=pd.to_numeric(z[c],errors='coerce')
    z=z.loc[(z.FDR<.05)&np.isfinite(z.Pvalue)&(z.Pvalue>=0)&np.isfinite(z.rho)].copy()
    z=z.drop_duplicates(['Variant_ID','Ensembl_ID'],keep='first')
    total=len(z)
    if total==0:raise RuntimeError('No PPMI CTAR significant tQTLs with valid rho')
    if available_snps is not None:
        z=z[z.Variant_ID.isin(available_snps)].copy()
    testable=len(z)
    if testable==0:raise RuntimeError('No PPMI significant pairs have PDBP genotypes')
    z=z.sort_values(['Variant_ID','Ensembl_ID'],kind='mergesort').reset_index(drop=True)
    # Random and representative of all PPMI discoveries; NO use of PDBP outcome/P-value.
    rng=np.random.default_rng(seed)
    k=min(n_pairs,total)
    idx=np.sort(rng.choice(len(z),size=k,replace=False))
    chosen=z.iloc[idx].copy().reset_index(drop=True)
    chosen['pair_id']=chosen.Variant_ID+'||'+chosen.Ensembl_ID
    chosen['neglogp_ppmi']=-np.log10(np.maximum(chosen.Pvalue.to_numpy(float),1e-300))
    # Tertiles based only on PPMI; ranks avoid qcut failures from tied values.
    for col,out in [('rho','rho_bin'),('neglogp_ppmi','signal_bin')]:
        q=min(3,len(chosen))
        chosen[out]=pd.qcut(chosen[col].rank(method='first'),q=q,labels=False).astype(int)
    if chosen.pair_id.duplicated().any(): raise RuntimeError('Repeated selected SNP-gene pair')
    return chosen,total,testable


def without_replacement_subjects(meta: pd.DataFrame, expr: pd.DataFrame,
                                 geno: pd.DataFrame, fraction: float, seed: int):
    """Sample unique donors without replacement, preserving every real visit."""
    if not meta.index.equals(expr.index):raise RuntimeError('meta/expr misaligned')
    ids=pd.Index(pd.unique(meta['participant_id'].astype(str)))
    if ids.empty:raise RuntimeError('No participants')
    if geno.index.astype(str).has_duplicates:raise RuntimeError('Duplicated genotype participant ids')
    if not (0<fraction<=1):raise ValueError('fraction must be in (0,1]')
    if fraction==1:
        # EXACT full cohort: no bootstrap, no synthetic donor IDs.
        return meta.copy(),expr.copy(),geno.copy(),list(map(str,ids)),True
    n=max(1,int(round(len(ids)*fraction)))
    rng=np.random.default_rng(seed)
    selected=list(map(str,rng.choice(ids.to_numpy(str),size=n,replace=False)))
    mask=meta.participant_id.astype(str).isin(selected).to_numpy()
    m=meta.iloc[np.flatnonzero(mask)].copy()
    e=expr.iloc[np.flatnonzero(mask)].copy()
    g=geno.reindex(selected).copy()
    if not m.index.equals(e.index):raise RuntimeError('Subsample meta/expr misaligned')
    if m.participant_id.nunique()!=n or g.isna().all(axis=1).any():
        raise RuntimeError('Subsample lost requested genotype participants')
    return m,e,g,selected,False


def prep_pdbp(paths, chosen: pd.DataFrame):
    from gengps.amp_pd.real_qtl import (
        load_cohort_metadata,load_expression_for_meta,_augment_fixed_genotype_pcs,
        _prepare_irregular_followup_rows,method_covariates,_drop_missing_covariates,
        bfile_for_cohort)
    from gengps.amp_pd.genotype import make_genotype_source
    print('[PDBP] Loading original current_v89 metadata and expression',flush=True)
    meta=load_cohort_metadata(paths,'PD')
    meta=_augment_fixed_genotype_pcs(paths,'PD',meta,n_pcs=8)
    genes=chosen.Ensembl_ID.drop_duplicates().tolist()
    expr=load_expression_for_meta(paths,meta,genes)
    meta,expr,dups=_prepare_irregular_followup_rows(meta,expr)
    covars=method_covariates(METHOD,'PD',meta,cell_adjust=True)
    meta,expr=_drop_missing_covariates(meta,expr,covars)
    missing=[g for g in genes if g not in expr.columns or not np.isfinite(expr[g].to_numpy(float)).all()]
    if missing:
        raise RuntimeError(f'PDBP expression lacks or has non-finite values for selected PP genes: {missing[:20]}. '
                           'Do not silently alter PPMI candidate selection. Choose a new SEED if needed.')
    pids=pd.unique(meta.participant_id.astype(str)).tolist()
    snps=chosen.Variant_ID.drop_duplicates().tolist()
    gs=make_genotype_source(str(bfile_for_cohort(paths,'PD')),paths.plink19,prefer_bed_reader=True)
    gt=gs.read(pids,snps,mean_impute=False,dtype=np.float64)
    gt=gt.fillna(-1.0)  # Same original Stage-I binary missing-call category; NOT new IV QC.
    if gt.isna().any().any() or not np.isin(gt.to_numpy(float),[-1,0,1,2]).all():
        raise RuntimeError('PDBP SNPs do not match frozen Stage-I binary hardcall encoding')
    # At least two genotype classes are needed to make the IV model identifiable;
    # this is a preflight data-availability check, not a new MAF/HWE filter.
    constant=[s for s in snps if len(np.unique(gt[s].to_numpy(float)))<2]
    if constant:raise RuntimeError(f'Uninformative PDBP genotype columns for selected PPMI pairs: {constant[:12]}. '
                                   'No new QC applied; choose another random PPMI subset.')
    print(f'[PDBP] n_people={len(pids)}, n_observations={len(meta)}, n_genes={len(genes)}, '
          f'n_snps={len(snps)}, duplicate_visits_resolved={dups}, n_covariates={len(covars)}',flush=True)
    return meta,expr,gt,covars


def one_scan(scanner,meta,expr,geno,covars,cis,fraction,seed):
    m,e,g,ids,is_full=without_replacement_subjects(meta,expr,geno,fraction,seed)
    fit=scanner.scan(m,e,g,cis,covariate_cols=covars,id_col='participant_id',time_col='followup_time')
    if len(fit)!=len(cis):
        raise RuntimeError(f'CTAR scan returned {len(fit)} pairs, expected {len(cis)}; missing models cannot be silently skipped')
    fit=fit[['Variant_ID','Ensembl_ID','Pvalue','rho']].copy()
    fit['Pvalue']=pd.to_numeric(fit['Pvalue'],errors='coerce')
    if not np.isfinite(fit.Pvalue.to_numpy(float)).all():raise RuntimeError('Non-finite CTAR P-values in refit')
    fit['pair_id']=fit.Variant_ID.astype(str)+'||'+fit.Ensembl_ID.astype(str)
    if fit.pair_id.duplicated().any() or set(fit.pair_id)!=set(cis.snpid.astype(str)+'||'+cis.geneid.astype(str)):
        raise RuntimeError('Refit pairs differ from frozen PPMI-discovery selection')
    fit['detected']=(fit.Pvalue<.05).astype(int)
    fit['fraction']=fraction
    fit['n_unique_participants']=len(ids)
    fit['n_obs']=len(m)
    fit['is_full_dataset']=is_full
    return fit,ids


def write_parquet_atomic(z:pd.DataFrame,path:Path):
    tmp=path.with_suffix(path.suffix+'.tmp')
    z.to_parquet(tmp,index=False)
    os.replace(tmp,path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    p.add_argument('--out-root',default='')
    p.add_argument('--n-pairs',type=int,default=100)
    p.add_argument('--resamples',type=int,default=24)
    p.add_argument('--fractions',default='0.50,0.75,0.90,1.00')
    p.add_argument('--seed',type=int,default=20261029)
    p.add_argument('--device',default='cpu')
    p.add_argument('--rho-points',type=int,default=100)
    p.add_argument('--rho-max',type=float,default=.99)
    p.add_argument('--reference-interval',type=float,default=1.)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.n_pairs<1 or a.resamples<2:raise ValueError('n-pairs>=1, resamples>=2')
    fractions=sorted({round(float(x),5) for x in a.fractions.split(',') if x.strip()})
    if not fractions or fractions[-1]!=1.0 or any(not 0<x<=1 for x in fractions):
        raise ValueError('fractions must be in (0,1], including exactly 1.0')
    root=Path(a.root)
    out=Path(a.out_root) if a.out_root else root/'manuscript_figures'/'tqtl_benchmark_v4'
    analysis=out/'power_analysis_independent_v2'
    parts=analysis/'checkpoints';parts.mkdir(parents=True,exist_ok=True)
    pp_file=root/'qtl'/'PP'/METHOD/'final'/'significant_tqtl.parquet'
    if not pp_file.is_file():raise FileNotFoundError(pp_file)
    manifest_file=analysis/'analysis_manifest.json'
    select_file=analysis/'PPMI_selected_pairs_NO_PDBP_PREFILTER.csv'
    config={
        'analysis':'PPMI-discovered CTAR tQTLs; independent PDBP sample-size sensitivity',
        'source_pp':str(pp_file),'pp_file_fingerprint':fingerprint(pp_file),
        'n_pairs':a.n_pairs,'resamples':a.resamples,'fractions':fractions,'seed':a.seed,
        'device':a.device,'rho_points':a.rho_points,'rho_max':a.rho_max,
        'reference_interval':a.reference_interval,'cohort':'PD',
        'candidate_selection':'PPMI global BH FDR<0.05, PDBP SNP genotyped/available (assay coverage only); PDBP association p-values NOT used',
        'subsampling':'person-level without replacement; complete donor visits retained',
        'full_fraction':'100% is original full PDBP cohort fitted ONCE, not a bootstrap',
        'pdbp_test_threshold':'nominal P<0.05 (fixed, no subsample BH recalculation)',
        'model':'unmodified gengps.methods.ctar_mptga.ContinuousTimeMPTGA',
        'interpretation':'sample-size sensitivity of independent replication of known PPMI discoveries; not an unbiased power estimate',
    }
    if manifest_file.exists():
        old=json.loads(manifest_file.read_text())
        for k in ['pp_file_fingerprint','n_pairs','resamples','fractions','seed','device','rho_points','rho_max','reference_interval','cohort','candidate_selection']:
            if old.get(k)!=config[k]:
                raise RuntimeError(f'Existing checkpoints belong to another configuration: {k}. '
                                   'Set FIGROOT to a new directory or use an empty power_analysis_independent_v2 directory.')
        selected=pd.read_csv(select_file,dtype={'Variant_ID':str,'Ensembl_ID':str,'pair_id':str})
    else:
        if list(parts.glob('*.parquet')):
            raise RuntimeError('Checkpoints found without manifest; cannot safely resume')
        import pyarrow.parquet as pq
        cols=['Variant_ID','Ensembl_ID','Pvalue','FDR','rho']
        schema=set(pq.ParquetFile(pp_file).schema.names)
        missing=set(cols)-schema
        if missing:raise RuntimeError(f'Stage-I PPMI significant parquet missing: {sorted(missing)}')
        pp=pd.read_parquet(pp_file,columns=cols)
        # Physical assay availability is allowed; never condition on PDBP P-values.
        bim=root/'data'/'genotype'/'REP'/'amp_pd_REP_qc.bim'
        pp_snps=set(pp.Variant_ID.astype(str))
        have=scan_pdbp_bim_for_snps(bim,pp_snps)
        print(f'[PDBP TESTABILITY] PPMI unique SNPs={len(pp_snps)}; SNP IDs present in PDBP BIM={len(have)}. '
              'No PDBP significance check, no new SNP QC.',flush=True)
        selected,total,testable=select_ppmi_pairs(pp,a.n_pairs,a.seed,have)
        selected.to_csv(select_file,index=False)
        config['n_total_discovery_candidates']=total
        config['n_ppmi_pairs_with_PDBP_SNP_present']=testable
        config['n_selected']=len(selected)
        manifest_file.write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(f'[SELECT] PPMI BH q<0.05: selected {len(selected)} distinct SNP-gene pairs with PDBP SNP assay coverage. '
          'NO PDBP significance / trajectory-direction preselection',flush=True)
    if a.dry_run:
        print('[DRY RUN] No genotype read / CTAR refit',flush=True);return
    # Depend on production CTAR exactly; do not approximate the model.
    code=Path(__file__).resolve().parents[2]
    if not (code/'gengps'/'methods'/'ctar_mptga.py').is_file():
        raise FileNotFoundError(f'Expected project core at {code}. Install under $CODE/scripts_figures/tqtl_power')
    sys.path.insert(0,str(code))
    from gengps.amp_pd.config import AmpPdPaths
    from gengps.methods.ctar_mptga import ContinuousTimeMPTGA
    paths=AmpPdPaths(root=str(root))
    cis=selected[['Variant_ID','Ensembl_ID']].rename(columns={'Variant_ID':'snpid','Ensembl_ID':'geneid'})
    tasks=[(frac, r) for frac in fractions for r in ([-1] if frac==1 else range(a.resamples))]
    outstanding=[(frac,r) for frac,r in tasks if not (parts/f'PD_fraction_{frac:.2f}_rep_{r:04d}.parquet').is_file()]
    print(f'[TASKS] total={len(tasks)} outstanding={len(outstanding)}, '
          '100% PDBP is one deterministic full-dataset refit',flush=True)
    if not outstanding:return
    meta,expr,geno,covars=prep_pdbp(paths,selected)
    scanner=ContinuousTimeMPTGA(basis_kind='cubic',genotype_mode='binary',
        rho_points=a.rho_points,batch_size=256,device=a.device,dtype='float64',
        reference_interval=a.reference_interval,rho_max=a.rho_max)
    n_total=meta.participant_id.nunique()
    for frac,rep in outstanding:
        seed=int(np.random.SeedSequence([a.seed,int(round(frac*10000)),max(0,rep)]).generate_state(1)[0])
        fit,ids=one_scan(scanner,meta,expr,geno,covars,cis,frac,seed)
        fit['replicate_id']=rep
        fit['cohort']='PD'
        fit['n_full_cohort']=n_total
        fname=parts/f'PD_fraction_{frac:.2f}_rep_{rep:04d}.parquet'
        write_parquet_atomic(fit,fname)
        print(f'[PROGRESS] fraction={frac:.2f} rep={rep} subjects={len(ids)}/{n_total} '
              f'replicated={int(fit.detected.sum())}/{len(fit)}',flush=True)
    print('[DONE] Independent PDBP CTAR refits complete',flush=True)

if __name__=='__main__':main()
