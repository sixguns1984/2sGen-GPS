#!/usr/bin/env python3
"""Safe, restartable parallel CTAR power runner and deterministic merge.

Workers have separate output files. No two processes append to the same CSV.
Original Stage-I/Stage-II data are read-only. Checkpoint resume is per shard.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd

HERE=Path(__file__).resolve().parent

def atomic_write(df,path):
    tmp=path.with_name(path.name+'.tmp')
    df.to_csv(tmp,index=False)
    tmp.replace(path)

def merge(out,jobs,required=True):
    out=Path(out)
    paths=[out/'_shards'/f'job_{i:02d}' for i in range(jobs)]
    results=[];fails=[];sense=[];names=[]
    for i,p in enumerate(paths):
        file=p/'locus_bootstrap_results.csv'
        if not file.exists() or not file.stat().st_size:
            if required:raise RuntimeError(f'Shard {i} missing result {file}; do not merge incomplete run')
            continue
        t=pd.read_csv(file)
        if t.empty:
            if required:raise RuntimeError(f'Shard {i} has zero valid fits')
            continue
        results.append(t);names.append(i)
        for fname,collection in [('locus_failures.csv',fails),('donor_sample_size_sensitivity.csv',sense)]:
            f=p/fname
            if f.exists() and f.stat().st_size:
                try:
                    z=pd.read_csv(f)
                    if len(z):collection.append(z)
                except pd.errors.EmptyDataError:pass
    if not results:raise RuntimeError('No valid cohort fits to merge')
    tab=pd.concat(results,ignore_index=True)
    if tab.duplicated(['gene','variant','cohort']).any():raise RuntimeError('Duplicate gene+variant+cohort results between shards')
    tab=tab.sort_values(['gene','variant','cohort'],kind='stable').reset_index(drop=True)
    atomic_write(tab,out/'locus_bootstrap_results.csv')
    for coll,f in [(fails,'locus_failures.csv'),(sense,'donor_sample_size_sensitivity.csv')]:
        if coll: atomic_write(pd.concat(coll,ignore_index=True),out/f)
    examples=out/'bootstrap_LRT_examples';examples.mkdir(exist_ok=True)
    for p in paths:
        src=p/'bootstrap_LRT_examples'
        if not src.exists():continue
        for file in src.glob('*.csv'):
            target=examples/file.name
            # Separate shard indices use original global pair index, hence unique.
            target.write_bytes(file.read_bytes())
    selected=pd.read_csv(out/'selected_loci.csv')
    total=len(selected);pdmatched=int(selected.rep_match.eq('both').sum())
    pc=int((tab.cohort=='PP').sum());dc=int((tab.cohort=='PD').sum())
    report={'jobs_requested':jobs,'shards_merged':names,'selected_ppmi_pairs':total,
            'matched_pdbp_pairs':pdmatched,'ppmi_fitted':pc,'pdbp_fitted':dc,
            'ppmi_success_fraction':pc/total,'pdbp_success_fraction':dc/max(pdmatched,1),
            'failure_count':int(sum(map(len,fails))),'target':'fixed-rho real-donor bootstrap 8df overall CTAR'}
    (out/'parallel_merge_audit.json').write_text(json.dumps(report,indent=2))
    print('[MERGED]',json.dumps(report,indent=2),flush=True)

def launch(args):
    root=Path(args.root);out=Path(args.out)
    selected=out/'selected_loci.csv'
    if not selected.exists():raise FileNotFoundError(f'Missing {selected}; run prepare first')
    audit=json.loads((out/'selection_audit.json').read_text())
    alpha=float(audit['observed_max_raw_p_of_all_hits'])
    fit_audit=out/'_preflight'/'fit_audit.json'
    if not fit_audit.exists():
        raise RuntimeError(f'Missing preflight LRT audit {fit_audit}; run preflight before full bootstrap')
    if not json.loads(fit_audit.read_text()).get('pass',False):
        raise RuntimeError('Stage-I LRT reproduction audit failed; full bootstrap blocked')
    cached=out/'expression_selected.parquet'
    if not cached.exists():raise FileNotFoundError(f'Missing cache: {cached}; run cache-expression first')
    logs=out/'_shards';logs.mkdir(exist_ok=True)
    env=os.environ.copy()
    for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[k]='1'
    processes=[];handles=[]
    print(f'[PARALLEL] jobs={args.jobs}, reps={args.reps}, batch={args.batch_size}, backend={args.backend}',flush=True)
    try:
        for i in range(args.jobs):
            directory=logs/f'job_{i:02d}';directory.mkdir(exist_ok=True)
            cmd=[sys.executable,str(HERE/'02_run_power.py'),'--root',str(root),'--outdir',str(directory),
                '--selected-file',str(selected),'--expression',str(cached),
                '--alpha-disc',str(alpha),'--bootstrap-reps',str(args.reps),
                '--sensitivity-reps',str(args.sensitivity_reps),'--sensitivity-pairs',str(args.sensitivity_pairs),
                '--audit-pairs',str(args.audit_pairs),'--batch-size',str(args.batch_size),
                '--backend',args.backend,'--num-shards',str(args.jobs),'--shard-id',str(i),
                '--seed',str(args.seed),'--resume']
            logfile=(directory/'worker.log').open('a',buffering=1);handles.append(logfile)
            logfile.write('[START] '+ ' '.join(cmd)+'\n')
            proc=subprocess.Popen(cmd,stdout=logfile,stderr=subprocess.STDOUT,env=env)
            processes.append((i,proc))
        errors=[]
        for i,proc in processes:
            rc=proc.wait()
            print(f'[WORKER {i}] returncode={rc}',flush=True)
            if rc:errors.append(i)
        if errors:raise RuntimeError(f'Workers failed: {errors}. Inspect _shards/job_XX/worker.log. Resume with SAME N_JOBS after fixing.')
        merge(out,args.jobs)
    except KeyboardInterrupt:
        for _,p in processes:
            if p.poll() is None:p.terminate()
        raise
    finally:
        for f in handles:f.close()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=('run','merge'))
    p.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    p.add_argument('--out')
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--reps',type=int,default=1000)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--backend',choices=['fast','legacy'],default='fast')
    p.add_argument('--sensitivity-reps',type=int,default=100)
    p.add_argument('--sensitivity-pairs',type=int,default=400)
    p.add_argument('--audit-pairs',type=int,default=10)
    p.add_argument('--seed',type=int,default=1173)
    x=p.parse_args()
    if x.jobs<1 or x.reps<1:raise ValueError('--jobs/--reps must be positive')
    if x.out is None:x.out=str(Path(x.root)/'manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_refit_v3')
    if x.command=='merge':merge(x.out,x.jobs)
    else:launch(x)
if __name__=='__main__':main()
