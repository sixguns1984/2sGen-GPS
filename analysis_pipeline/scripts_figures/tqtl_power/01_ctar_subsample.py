#!/usr/bin/env python3
"""Subject-bootstrap recovery of *selected and independently replicated* CTAR-tQTLs.

Uses the project's unmodified CTAR scanner; the resulting recovery is conditional
on selection and is not an unbiased frequentist power estimate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METHOD = 'submitted_followup_ctar_mptga'


def fingerprint(path: Path) -> str:
    """Lightweight metadata fingerprint; avoid hashing multi-GB full PD result parquet."""
    s=path.stat()
    return f"{s.st_size}:{s.st_mtime_ns}"


def read_validated_trajectories(path: Path) -> pd.DataFrame:
    """Use the existing allele-harmonized PPMI->PDBP trajectory-consistency results."""
    required=['Variant_ID','Ensembl_ID','Pvalue_rep',
              'trajectory_direction_concordant','pdbp_temporal_rep_p_lt_0_05',
              'allele_alignment']
    import pyarrow.parquet as pq
    found=set(pq.ParquetFile(path).schema.names)
    missing=[x for x in required if x not in found]
    if missing:raise RuntimeError(f'Trajectory validation file lacks columns: {missing}')
    z=pd.read_parquet(path, columns=required)
    z['Variant_ID']=z['Variant_ID'].astype(str)
    z['Ensembl_ID']=z['Ensembl_ID'].astype(str)
    z['Pvalue_rep']=pd.to_numeric(z['Pvalue_rep'],errors='coerce')
    valid=z['trajectory_direction_concordant'].fillna(False).astype(bool)
    replicated=z['pdbp_temporal_rep_p_lt_0_05'].fillna(False).astype(bool)
    allowed={'same','same_complement','flip','flip_complement'}
    z=z.loc[valid & replicated & z.allele_alignment.isin(allowed) & (z.Pvalue_rep<.05)].copy()
    if z.duplicated(['Variant_ID','Ensembl_ID']).any():
        raise RuntimeError('Duplicate pairs in trajectory_direction_pairs.parquet')
    print(f'[TRAJECTORY] allele-harmonized, direction-concordant, nominally replicated pairs={len(z)}',flush=True)
    return z[['Variant_ID','Ensembl_ID','Pvalue_rep']].rename(columns={'Pvalue_rep':'Pvalue'})


def read_pairs(path: Path, columns: list[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq
    available = set(pq.ParquetFile(path).schema.names)
    needed = [c for c in columns if c in available]
    if not {'Variant_ID','Ensembl_ID','Pvalue'}.issubset(needed):
        raise RuntimeError(f'{path} missing required columns: Variant_ID, Ensembl_ID, Pvalue; found={sorted(available)[:40]}')
    z = pd.read_parquet(path, columns=needed)
    for c in ('Variant_ID','Ensembl_ID'):
        z[c] = z[c].astype(str)
    for c in ['Pvalue','FDR','rho']:
        if c in z: z[c] = pd.to_numeric(z[c], errors='coerce')
    if z.duplicated(['Variant_ID','Ensembl_ID']).any():
        raise RuntimeError(f'Duplicated Variant_ID/Ensembl_ID rows in {path}')
    return z


def select_pairs(pp: pd.DataFrame, pdz: pd.DataFrame, n_pairs: int, seed: int):
    if 'FDR' not in pp:
        raise RuntimeError('PPMI significant_tqtl.parquet must include globally adjusted FDR. Cannot use chunk-level FDR.')
    if 'rho' not in pp:
        raise RuntimeError('PPMI result does not include CTAR rho; cannot stratify by autocorrelation strength.')
    pp = pp[(pp['FDR'] < 0.05) & np.isfinite(pp['rho'])].copy()
    if pp.empty: raise RuntimeError('No PP significant CTAR-tQTLs with rho')
    pcut = float(pp['Pvalue'].max())
    merged = pp.merge(pdz[['Variant_ID','Ensembl_ID','Pvalue']].rename(columns={'Pvalue':'PD_Pvalue'}),
                      on=['Variant_ID','Ensembl_ID'], how='inner', validate='one_to_one')
    merged = merged[merged['PD_Pvalue'] < 0.05].copy()
    if merged.empty: raise RuntimeError('No nominally replicated CTAR-tQTL pairs after PP/PD join')
    merged['neglogp'] = -np.log10(np.maximum(merged['Pvalue'].to_numpy(float),1e-300))
    # stratification on rho and discovery strength; replacement-free representative sample
    merged['rho_bin'] = pd.qcut(merged['rho'], q=3, duplicates='drop', labels=False)
    merged['signal_bin'] = pd.qcut(merged['neglogp'].rank(method='first'), q=3, duplicates='drop', labels=False)
    merged['stratum'] = merged['rho_bin'].astype(str) + '_' + merged['signal_bin'].astype(str)
    rng = np.random.default_rng(seed)
    groups = [g for _,g in merged.groupby('stratum',sort=True)]
    if n_pairs >= len(merged): chosen = merged.copy()
    else:
        # round-robin allocation to avoid a high-signal stratum dominating selection
        group_orders = [rng.permutation(len(g)).tolist() for g in groups]
        picks=[]
        while len(picks)<n_pairs:
            changed=False
            for g, order in zip(groups,group_orders):
                if order and len(picks)<n_pairs:
                    picks.append(g.iloc[order.pop()])
                    changed=True
            if not changed:break
        chosen=pd.DataFrame(picks)
    chosen=chosen.sort_values(['rho','Pvalue']).reset_index(drop=True)
    chosen['pair_id'] = chosen['Variant_ID'].astype(str)+'||'+chosen['Ensembl_ID'].astype(str)
    if chosen['pair_id'].duplicated().any():raise RuntimeError('Duplicate selected pair identifiers')
    return chosen, pcut, len(merged)


def replicate_subjects(meta: pd.DataFrame, expr: pd.DataFrame, genotype: pd.DataFrame,
                       fraction: float, seed: int):
    """Whole-subject bootstrap, including duplicate draws as independent synthetic subjects."""
    rng = np.random.default_rng(seed)
    ids = pd.Index(pd.unique(meta['participant_id'].astype(str)))
    n = max(1, int(round(len(ids) * fraction)))
    sampled = rng.choice(ids.to_numpy(), size=n, replace=True)
    m_chunks=[]; e_chunks=[]; gen_chunks=[]
    # integer positional row mapping avoids incorrect duplicated index reindexing
    pid_arr = meta['participant_id'].astype(str).to_numpy()
    by_pid={pid:np.flatnonzero(pid_arr==pid) for pid in ids}
    for k,pid in enumerate(sampled):
        source=by_pid[str(pid)]
        newpid=f'b{k:06d}::{pid}'
        new_ids=[f'b{k:06d}::{sid}' for sid in meta.iloc[source]['sample_id'].astype(str)]
        m=meta.iloc[source].copy(); e=expr.iloc[source].copy()
        m['participant_id']=newpid; m['sample_id']=new_ids
        m.index=pd.Index(new_ids); e.index=pd.Index(new_ids)
        m_chunks.append(m); e_chunks.append(e)
        gt=genotype.loc[[str(pid)]].copy();gt.index=[newpid];gen_chunks.append(gt)
    m=pd.concat(m_chunks,axis=0); e=pd.concat(e_chunks,axis=0); g=pd.concat(gen_chunks,axis=0)
    order=np.lexsort((m['sample_id'].astype(str).to_numpy(),
                       pd.to_numeric(m['followup_time']).to_numpy(float),
                       m['participant_id'].astype(str).to_numpy()))
    m=m.iloc[order].copy();e=e.iloc[order].copy()
    if not m.index.equals(e.index) or g.index.has_duplicates:
        raise RuntimeError('Bootstrap alignment/participant uniqueness failed')
    return m,e,g


def prepare_cohort(paths, cohort: str, chosen: pd.DataFrame):
    from gengps.amp_pd.real_qtl import (
        load_cohort_metadata,load_expression_for_meta,_augment_fixed_genotype_pcs,
        _prepare_irregular_followup_rows,method_covariates,_drop_missing_covariates,
        bfile_for_cohort)
    from gengps.amp_pd.genotype import make_genotype_source
    print(f'[COHORT {cohort}] Loading current_v89 metadata...',flush=True)
    meta=load_cohort_metadata(paths,cohort)
    meta=_augment_fixed_genotype_pcs(paths,cohort,meta,n_pcs=8)
    genes=chosen['Ensembl_ID'].drop_duplicates().tolist()
    expr=load_expression_for_meta(paths,meta,genes)
    meta,expr,dups=_prepare_irregular_followup_rows(meta,expr)
    covars=method_covariates(METHOD,cohort,meta,cell_adjust=True)
    meta,expr=_drop_missing_covariates(meta,expr,covars)
    # Never silently alter the selected panel per bootstrap or per cohort.
    missing_genes=[g for g in genes if g not in expr.columns or not np.isfinite(expr[g].to_numpy(float)).all()]
    if missing_genes:
        raise RuntimeError(f'{cohort}: selected genes missing/non-finite in current expression: {missing_genes[:20]}')
    pids=pd.unique(meta['participant_id'].astype(str)).tolist()
    snps=chosen['Variant_ID'].drop_duplicates().tolist()
    gsrc=make_genotype_source(str(bfile_for_cohort(paths,cohort)),paths.plink19,prefer_bed_reader=True)
    genotype=gsrc.read(pids,snps,mean_impute=False,dtype=np.float64)
    genotype=genotype.fillna(-1.0)
    if not np.isin(genotype.to_numpy(),[-1,0,1,2]).all():
        raise RuntimeError(f'{cohort}: genotype coding differs from frozen Stage1 binary hardcalls')
    print(f'[COHORT {cohort}] participants={len(pids)} obs={len(meta)} genes={len(genes)} snps={len(snps)} duplicate_visits_resolved={dups} covariates={len(covars)}',flush=True)
    return meta,expr,genotype,covars


def one_bootstrap(scan,meta,expr,geno,covars,cis,fraction,seed):
    m,e,g=replicate_subjects(meta,expr,geno,fraction,seed)
    result=scan.scan(m,e,g,cis,covariate_cols=covars,id_col='participant_id',time_col='followup_time')
    if len(result)!=len(cis):
        raise RuntimeError(f'Expected {len(cis)} pairs, got {len(result)} in CTAR refit')
    return result[['Variant_ID','Ensembl_ID','Pvalue','rho']].copy(),len(g),len(m)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    ap.add_argument('--out-root',default='')
    ap.add_argument('--n-pairs',type=int,default=120)
    ap.add_argument('--bootstraps',type=int,default=40)
    ap.add_argument('--fractions',default='0.5,0.75,1.0')
    ap.add_argument('--cohorts',default='PP,PD')
    ap.add_argument('--seed',type=int,default=20261029)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--rho-points',type=int,default=100)
    ap.add_argument('--rho-max',type=float,default=0.99)
    ap.add_argument('--reference-interval',type=float,default=1.0)
    ap.add_argument('--force-new-selection',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args()
    if a.n_pairs<1 or a.bootstraps<1:raise ValueError('n-pairs and bootstraps must be >=1')
    fractions=[float(x) for x in a.fractions.split(',') if x]
    if not fractions or any(not 0<x<=1 for x in fractions):raise ValueError('fractions must lie in (0,1]')
    cohorts=[x for x in a.cohorts.split(',') if x]
    if any(x not in ('PP','PD') for x in cohorts):raise ValueError('cohorts must be PP or PD')
    root=Path(a.root)
    out=Path(a.out_root) if a.out_root else root/'manuscript_figures'/'tqtl_benchmark_v4'
    analysis=out/'power_analysis';parts=analysis/'checkpoints';parts.mkdir(parents=True,exist_ok=True)
    pp_path=root/'qtl'/'PP'/METHOD/'final'/'significant_tqtl.parquet'
    pd_path=root/'qtl'/'trajectory_direction_consistency'/METHOD/'trajectory_direction_pairs.parquet'
    for src in (pp_path,pd_path):
        if not src.exists():raise FileNotFoundError(src)
    selection=analysis/'selected_replicated_CTAR_pairs.csv'
    manifest_path=analysis/'analysis_manifest.json'
    config={
        'source_pp':str(pp_path),'source_pd':str(pd_path),
        'fingerprint_pp':fingerprint(pp_path),'fingerprint_pd':fingerprint(pd_path),
        'method':METHOD, 'n_pairs':a.n_pairs,'seed':a.seed,'fractions':fractions,
        'bootstraps':a.bootstraps,'cohorts':cohorts,'rho_points':a.rho_points,
        'rho_max':a.rho_max,'reference_interval':a.reference_interval,
        'device':a.device,'data_mode':'current_v89','genotype_mode':'binary',
        'meaning':'conditional bootstrap discovery/replication recovery in matched, allele-harmonized, trajectory-direction-concordant CTAR pairs; NOT unbiased power',
    }
    if manifest_path.exists() and not a.force_new_selection:
        old=json.loads(manifest_path.read_text())
        for k in ['fingerprint_pp','fingerprint_pd','method','n_pairs','seed','fractions','bootstraps','cohorts','rho_points','rho_max','reference_interval','device']:
            if old.get(k)!=config[k]:raise RuntimeError(f'Cannot resume changed configuration at {out}: {k}. Use new OUT_ROOT or --force-new-selection to start clean.')
        chosen=pd.read_csv(selection,dtype={'Variant_ID':str,'Ensembl_ID':str,'pair_id':str})
        cutoff=float(old['pp_global_discovery_p_threshold'])
    else:
        if list(parts.glob('*.parquet')):
            raise RuntimeError('Existing per-bootstrap checkpoints present. Select a clean output directory, or preserve matching manifest.')
        pp=read_pairs(pp_path,['Variant_ID','Ensembl_ID','Pvalue','FDR','rho'])
        pdz=read_validated_trajectories(pd_path)
        chosen,cutoff,nrep=select_pairs(pp,pdz,a.n_pairs,a.seed)
        chosen.to_csv(selection,index=False)
        config['pp_global_discovery_p_threshold']=cutoff
        config['n_independently_nominal_replicated_candidates']=nrep
        config['actual_selected_n']=len(chosen)
        manifest_path.write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'[SELECT] n={len(chosen)} PP global threshold={cutoff:.3g} from observed PP FDR<0.05; PD nominal P<0.05 + allele-harmonized trajectory_r>0',flush=True)
    if a.dry_run:
        print('[DRY RUN] no genotype/expression refits',flush=True)
        return
    # import the ACTUAL installed project implementation, not a simplified approximate CTAR.
    code=Path(__file__).resolve().parents[2]
    if not (code/'gengps'/'methods'/'ctar_mptga.py').exists():
        raise FileNotFoundError(f'Project core not found under {code}; install scripts under CODE/scripts_figures/tqtl_power')
    sys.path.insert(0,str(code))
    from gengps.amp_pd.config import AmpPdPaths
    from gengps.methods.ctar_mptga import ContinuousTimeMPTGA
    paths=AmpPdPaths(root=str(root))
    cis=chosen[['Variant_ID','Ensembl_ID']].rename(columns={'Variant_ID':'snpid','Ensembl_ID':'geneid'}).copy()
    for cohort in cohorts:
        all_files=[parts/f'{cohort}_fraction_{frac:.2f}_rep_{rep:04d}.parquet' for frac in fractions for rep in range(a.bootstraps)]
        if all(p.exists() for p in all_files):
            print(f'[SKIP] {cohort}: all checkpoints exist',flush=True)
            continue
        meta,expr,geno,covars=prepare_cohort(paths,cohort,chosen)
        scanner=ContinuousTimeMPTGA(basis_kind='cubic', genotype_mode='binary',
                 rho_points=a.rho_points,batch_size=256,device=a.device,dtype='float64',
                 reference_interval=a.reference_interval,rho_max=a.rho_max)
        for frac in fractions:
            for rep in range(a.bootstraps):
                target=parts/f'{cohort}_fraction_{frac:.2f}_rep_{rep:04d}.parquet'
                if target.exists():continue
                # reproducible draws independent of execution order / resume
                seed=int(np.random.SeedSequence([a.seed, 1 if cohort=='PP' else 2,
                                                 int(round(frac*10000)),rep]).generate_state(1)[0])
                fit,npid,nobs=one_bootstrap(scanner,meta,expr,geno,covars,cis,frac,seed)
                fit['cohort']=cohort;fit['fraction']=frac;fit['bootstrap_id']=rep
                fit['n_resampled_subjects']=npid;fit['n_observations']=nobs
                fit['p_threshold']=cutoff if cohort=='PP' else 0.05
                fit['detected']=(pd.to_numeric(fit['Pvalue'],errors='coerce')<fit['p_threshold']).astype(int)
                if not np.isfinite(fit['Pvalue'].to_numpy(float)).all():
                    raise RuntimeError(f'Non-finite bootstrap P-value for {cohort} fraction {frac} rep {rep}')
                tmp=target.with_suffix('.parquet.tmp')
                fit.to_parquet(tmp,index=False);os.replace(tmp,target)
                print(f'[PROGRESS] {cohort} fraction={frac:.2f} bootstrap={rep+1}/{a.bootstraps} matched={len(fit)} subjects={npid}',flush=True)
    print('[DONE] checkpoint refits completed. Run 02_plot_sfigure6.py.',flush=True)

if __name__=='__main__':main()
