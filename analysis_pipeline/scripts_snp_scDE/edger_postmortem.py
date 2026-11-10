#!/usr/bin/env python3
"""Independent edgeR QLF add-on for 2sGenGPS postmortem scDE v6.3.

Input: v6.2+ prepared pseudobulk only; NO VCF/H5AD re-processing.
Output: OUT/edger; never writes OUT/de, OUT/pseudobulk, or OUT/fit_status.csv.
One SNP x region x cell type x stratum per R process. True edgeR 4 QLF, not Dreamlet.
"""
import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, io

# Reuse actual v6.3 design, input schema and sample validation unchanged.
try:
    from postmortem_scde import (ROOT_DEFAULT, REGIONS, CELLTYPES, TARGETS,
                                prepared_region_state, design_for)
except ImportError as exc:
    raise SystemExit('Missing postmortem_scde.py v6.3 in same scripts_snp_scDE directory: '+str(exc))

MODULE_DIR = Path(__file__).resolve().parent
VERSION = 'edger-qlf-addon-v1.0'
REQUIRED_RESULT = {'gene_id','log2FC','stat','pvalue','padj'}


def parse_csv_choices(text, allowed, label):
    values = [x.strip() for x in text.split(',') if x.strip()]
    if not values or len(values) != len(set(values)) or any(x not in allowed for x in values):
        raise ValueError(f'Invalid {label}: {text}. Allowed: {list(allowed)}')
    return values


def file_stamp(path):
    path = Path(path)
    stat = path.stat()
    return {'path': str(path.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def fingerprint(sources, metadata, gene_ids, settings):
    payload = {
        'addon_version': VERSION,
        'sources': [file_stamp(x) for x in sources],
        'metadata': json.loads(metadata.to_json(orient='records', double_precision=15)),
        'gene_ids_sha256': hashlib.sha256(('\n'.join(map(str,gene_ids))).encode()).hexdigest(),
        'settings': settings,
        'r_script_sha256': hashlib.sha256((MODULE_DIR/'edger_qlf_fit.R').read_bytes()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload,sort_keys=True,default=str).encode()).hexdigest()


def persist_status(root, entries):
    root.mkdir(parents=True, exist_ok=True)
    path = root/'fit_status.csv'
    tmp = root/'fit_status.csv.tmp'
    old=[]
    if path.is_file():
        try:
            old=pd.read_csv(path).to_dict(orient='records')
        except (OSError,ValueError,pd.errors.ParserError):
            pass
    # Each job is identified by the four comparison keys; retain other historical jobs.
    key=lambda row: tuple(str(row.get(s,'')) for s in ['snp','region','cell_type','stratum'])
    combined={key(x):x for x in old}
    combined.update({key(x):x for x in entries})
    pd.DataFrame(list(combined.values())).to_csv(tmp,index=False)
    tmp.replace(path)


def result_path(root, stratum, snp, region, ct):
    return root/'de'/stratum/snp/region/ct/'edger_qlf.csv'


def is_resume_valid(path, stamp_path, fp, ids):
    if not path.is_file() or not stamp_path.is_file(): return False
    try:
        manifest=json.loads(stamp_path.read_text())
        if manifest.get('fingerprint')!=fp: return False
        z=pd.read_csv(path)
        if z.empty or not REQUIRED_RESULT.issubset(z.columns):return False
        observed=z.gene_id.astype(str)
        if observed.duplicated().any() or not set(observed).issubset(set(ids)):
            return False
        if not z.method.astype(str).eq('edger_qlf').all():return False
        if not z.contrast.astype(str).eq('VCF_ALT_carrier_vs_noncarrier').all():return False
        if 'stat_type' not in z.columns or not z.stat_type.astype(str).eq('QL_F').all():return False
        return True
    except (OSError, KeyError, ValueError, json.JSONDecodeError, pd.errors.ParserError):
        return False


def r_check(executable):
    """Check the actual installed edgeR; do not install or modify R packages."""
    if not executable or not shutil.which(executable) and not (Path(executable).is_file() and os.access(executable,os.X_OK)):
        raise RuntimeError(f'Rscript not found: {executable!r}; pass --rscript /absolute/path/bin/Rscript')
    script = """suppressPackageStartupMessages({library(edgeR); library(Matrix)})
stopifnot(packageVersion('edgeR') >= '4.0.0')
print(paste('R=',as.character(getRversion())))
print(paste('edgeR=',as.character(packageVersion('edgeR'))))
print(paste('Matrix=',as.character(packageVersion('Matrix'))))"""
    subprocess.run([executable, '-e', script], check=True)


def export_job(work, raw, gene_ids, md, X, cell_type):
    """Persist strictly aligned genes x donors MatrixMarket and donor model metadata."""
    work.mkdir(parents=True,exist_ok=True)
    assert sparse.issparse(raw)
    gene_ids=np.asarray(gene_ids,dtype=str)
    assert len(gene_ids)==raw.shape[1] and len(md)==raw.shape[0]
    if len(np.unique(gene_ids)) != len(gene_ids):
        raise ValueError('Duplicate gene IDs after gene_name collapse')
    if not np.issubdtype(raw.dtype,np.integer) or (len(raw.data) and (raw.data<0).any()):
        raise ValueError('Pseudobulk must contain nonnegative integer raw counts')
    if (np.asarray(raw.sum(axis=1)).ravel()<=0).any():
        raise ValueError('Zero-library pseudobulk samples')
    ids=md.donor.astype(str).to_numpy()
    if len(set(ids))!=len(ids):raise ValueError('Duplicated donor IDs')
    if (md.n_cells.astype(int)<1).any():raise ValueError('Invalid n_cells')
    m=pd.DataFrame({
        'donor':ids,
        'carrier':md.carrier.astype(int).to_numpy(),
        'ageScaled':X.age_z.astype(float).to_numpy(),
        'sex':md.sex.astype(str).to_numpy(),
        'status':md['case'].astype(str).to_numpy(),
        'n_cells':md.n_cells.astype(int).to_numpy(),
    })
    if set(m.carrier)!={0,1}:raise ValueError('Carrier contrast must include 0 and 1')
    if not np.isfinite(m.ageScaled).all():raise ValueError('Nonfinite age')
    if not m.sex.isin(['M','F']).all() or not m.status.isin(['Case','Control']).all():
        raise ValueError('Unrecognized sex/case fields')
    gene_csv=work/'genes.csv'
    donor_csv=work/'donors.csv'
    mtx=work/'counts.mtx'
    pd.DataFrame({'gene_id':gene_ids}).to_csv(gene_csv,index=False)
    m.to_csv(donor_csv,index=False)
    # R Matrix::readMM yields genes x samples, the edgeR input orientation.
    io.mmwrite(str(mtx),raw.T.tocoo(),field='integer')
    return mtx,gene_csv,donor_csv


def fit_one(a,root,region,ct,snp,stratum,genes,m0,counts,g,prep_marker):
    input_dir=Path(a.out)/'pseudobulk'/region
    donor_file=input_dir/ct/'donors.csv'
    cnt_file=input_dir/ct/'counts.npz'
    gene_file=input_dir/'gene_mapping.csv'
    geno_file=Path(a.out)/'vcf_genotypes.csv'
    output=result_path(root,stratum,snp,region,ct)
    prefix={'snp':snp,'region':region,'cell_type':ct,'stratum':stratum}
    gg=g.loc[g.snp.eq(snp),['donor','carrier','alt_dose']]
    m=m0.merge(gg,on='donor',how='left',validate='one_to_one')
    m.index=pd.Index(np.arange(len(m)))
    designed,why=design_for(m,stratum,a.min_group)
    if designed is None:
        return dict(**prefix,status='SKIPPED',reason=why,n_genes=0)
    md,X=designed
    raw=counts[md.index,:].tocsr()
    total=np.asarray(raw.sum(axis=0)).ravel()
    detected=np.asarray((raw>=a.min_per_donor_count).sum(axis=0)).ravel()
    good=(total>=a.min_total_count)&(detected>=a.min_detected_donors)
    if not np.any(good):return dict(**prefix,status='SKIPPED',reason='no_expressed_genes',n_genes=0)
    ids=genes.gene_id.astype(str).to_numpy()[good]
    raw=raw[:,good].tocsr()
    settings={
        'min_group':a.min_group,
        'min_cells':a.min_cells,
        'min_total_count':a.min_total_count,
        'min_per_donor_count':a.min_per_donor_count,
        'min_detected_donors':a.min_detected_donors,
        'edger_min_count':a.edger_min_count,
        'edger_min_prop':a.edger_min_prop,
        'normalize_method':'TMM',
        'edger_min_total':a.edger_min_total,
        'robust':not a.no_robust,
        'model':'glmQLFit legacy=FALSE, glmQLFTest carrier1',
        'snp':snp,'region':region,'cell_type':ct,'stratum':stratum,
    }
    fp=fingerprint([donor_file,cnt_file,gene_file,geno_file,prep_marker],md.assign(ageScaled=X.age_z.to_numpy()),ids,settings)
    stamp=output.with_suffix('.manifest.json')
    n1=int((md.carrier==1).sum());n0=int((md.carrier==0).sum())
    if a.resume and is_resume_valid(output,stamp,fp,ids):
        z=pd.read_csv(output,usecols=['padj'])
        return dict(**prefix,status='PASS',reason='RESUMED',n_carrier=n1,n_noncarrier=n0,
                    n_genes=len(z),n_input_genes=len(ids),n_fdr05=int((z.padj<.05).sum()))
    if a.dry_run:
        return dict(**prefix,status='DRY_RUN',reason='Not executed',n_carrier=n1,
                    n_noncarrier=n0,n_input_genes=len(ids),n_genes=0)
    work=root/'_work'/stratum/snp/region/ct
    mtx,genes_csv,donors_csv=export_job(work,raw,ids,md,X,ct)
    output.parent.mkdir(parents=True,exist_ok=True)
    r_temp=work/'edger_result.csv'
    cmd=[a.rscript,str(MODULE_DIR/'edger_qlf_fit.R'),str(mtx),str(genes_csv),
         str(donors_csv),str(r_temp),str(a.edger_min_count),
         str(a.edger_min_total),str(a.edger_min_prop),str(int(not a.no_robust))]
    env=dict(os.environ)
    env.update({key:str(a.n_cpus) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS')})
    logfile=output.parent/'edger_qlf.log'
    try:
        with logfile.open('w') as log:
            log.write('COMMAND: '+' '.join(cmd)+'\n')
            log.flush()
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not r_temp.is_file():
            raise RuntimeError(f'R exited 0 but missing expected result; see {logfile}')
        z=pd.read_csv(r_temp)
        if z.empty or not REQUIRED_RESULT.issubset(z.columns):
            raise RuntimeError('R edgeR output missing required columns or empty')
        if z.gene_id.astype(str).duplicated().any() or not set(z.gene_id.astype(str)).issubset(set(ids)):
            raise RuntimeError('R edgeR output has invalid gene IDs')
        z['gene_id']=z.gene_id.astype(str)
        z=z.merge(genes[['gene_id','gene_symbol']],on='gene_id',how='left',validate='one_to_one')
        z.insert(0,'method','edger_qlf')
        for key,value in reversed(list(prefix.items())):
            z.insert(0,key,value)
        norm_src=work/'edger_result_norm_factors.csv'
        if norm_src.exists():
            shutil.copyfile(norm_src,output.parent/'edgeR_TMM_norm_factors.csv')
        z['n_carrier']=n1
        z['n_noncarrier']=n0
        z['n_genes_tested']=len(z)
        z['n_genes_input']=len(ids)
        z['contrast']='VCF_ALT_carrier_vs_noncarrier'
        csv_tmp=output.with_suffix('.csv.tmp')
        z.to_csv(csv_tmp,index=False)
        csv_tmp.replace(output)
        manifest={
            'fingerprint':fp,'settings':settings,'n_carrier':n1,'n_noncarrier':n0,
            'n_genes_input':len(ids),'n_genes_tested':len(z),
            'timestamp_utc':datetime.now(timezone.utc).isoformat(),
            'source_prepared_marker':str(prep_marker),
            'method':'edger_qlf',
            'description':'single-donor pseudobulk, edgeR v4 NB quasi-likelihood F-test; stat is unsigned F, sign in log2FC'
        }
        stamp_tmp=stamp.with_suffix('.manifest.json.tmp')
        stamp_tmp.write_text(json.dumps(manifest,indent=2))
        stamp_tmp.replace(stamp)
        return dict(**prefix,status='PASS',reason='',n_carrier=n1,n_noncarrier=n0,
                    n_genes=len(z),n_input_genes=len(ids),n_fdr05=int((z.padj<.05).sum()))
    except Exception as ex:
        raise RuntimeError(f'Failure at {snp} {region} {ct} {stratum}: {ex}. '
                           f'See {logfile}; work files retained at {work}') from ex
    finally:
        if not a.keep_work and output.is_file() and stamp.is_file():
            shutil.rmtree(work,ignore_errors=True)


def run_fit(a):
    root=Path(a.out)/'edger'
    root.mkdir(parents=True,exist_ok=True)
    regions=parse_csv_choices(a.regions,REGIONS,'regions')
    celltypes=parse_csv_choices(a.celltypes,CELLTYPES,'celltypes')
    snps=parse_csv_choices(a.snps,TARGETS,'snps')
    strata=parse_csv_choices(a.strata,['all','case_only'],'strata')
    if not a.dry_run:r_check(a.rscript)
    geno=Path(a.out)/'vcf_genotypes.csv'
    if not geno.is_file():raise RuntimeError(f'Missing prepared genotype table: {geno}')
    g=pd.read_csv(geno)
    if not {'snp','donor','carrier','alt_dose'}.issubset(g.columns):
        raise RuntimeError('vcf_genotypes.csv missing expected columns')
    if g.duplicated(['snp','donor']).any():raise RuntimeError('Duplicate donor-genotype rows')
    all_status=[]
    attempted=0
    for region in regions:
        regdir=Path(a.out)/'pseudobulk'/region
        valid,missing=prepared_region_state(regdir)
        if not valid:raise RuntimeError(f'{region}: incomplete v6.2+ pseudobulk: {missing}. '
                                        'Do not re-run prepare while v6.3 is still running.')
        prep_marker=regdir/'.prepared_complete.json'
        prep=json.loads(prep_marker.read_text())
        if prep.get('min_cells')!=a.min_cells:
            raise RuntimeError(f'{region}: prepare min_cells={prep.get("min_cells")}, '
                               f'but edgeR --min-cells={a.min_cells}. Match existing prepare.')
        genes=pd.read_csv(regdir/'gene_mapping.csv')
        if not {'gene_id','gene_symbol'}.issubset(genes.columns):raise RuntimeError('Invalid gene_mapping.csv')
        if genes.gene_id.astype(str).duplicated().any():raise RuntimeError(f'{region}: duplicate prepared gene IDs')
        for ct in celltypes:
            dest=regdir/ct
            m0=pd.read_csv(dest/'donors.csv')
            if m0.empty:
                logging.warning('%s/%s: zero eligible donors',region,ct)
                continue
            if not {'donor','age','sex','case','n_cells'}.issubset(m0.columns):
                raise RuntimeError(f'{dest}/donors.csv has missing covariates')
            counts=sparse.load_npz(dest/'counts.npz').tocsr()
            if counts.shape!=(len(m0),len(genes)):
                raise RuntimeError(f'{region}/{ct}: metadata/counts/gene shapes mismatched')
            if m0.donor.astype(str).duplicated().any():
                raise RuntimeError(f'{region}/{ct}: duplicate donor IDs')
            for snp in snps:
                for stratum in strata:
                    try:
                        item=fit_one(a,root,region,ct,snp,stratum,genes,m0,counts,g,prep_marker)
                        all_status.append(item)
                        persist_status(root,all_status)
                        logging.info('%s: %s/%s/%s/%s input_genes=%s tested_genes=%s group=%s/%s FDR05=%s',
                                     item['status'],snp,region,ct,stratum,item.get('n_input_genes'),item.get('n_genes'),
                                     item.get('n_carrier'),item.get('n_noncarrier'),item.get('n_fdr05'))
                    except Exception as ex:
                        all_status.append(dict(region=region,cell_type=ct,snp=snp,stratum=stratum,status='ERROR',reason=str(ex)))
                        persist_status(root,all_status)
                        logging.exception('edgeR job failed')
                        if a.fail_fast:raise
                    attempted+=1
                    if a.max_jobs and attempted>=a.max_jobs:
                        logging.info('Stopped intentionally after %s attempted jobs (--max-jobs)',attempted)
                        return
    logging.info('Completed edgeR jobs; output = %s',root)


def run_summary(a):
    root=Path(a.out)/'edger'
    files=sorted((root/'de').glob('*/*/*/*/edger_qlf.csv'))
    if not files:raise RuntimeError('No edger_qlf.csv results found; run fit')
    frames=[]
    for file in files:
        x=pd.read_csv(file)
        if not REQUIRED_RESULT.issubset(x.columns):
            raise RuntimeError(f'edgeR result file malformed: {file}')
        frames.append(x)
    z=pd.concat(frames,ignore_index=True)
    z.to_csv(root/'ALL_genes_edgeR_QLF_results.csv',index=False)
    z['fdr05']=pd.to_numeric(z.padj,errors='coerce')<.05
    z[z.fdr05].sort_values('padj').to_csv(root/'FDR05_significant_edgeR_QLF_genes.csv',index=False)
    z.sort_values(['snp','region','cell_type','stratum','pvalue']).groupby(
        ['snp','region','cell_type','stratum'],sort=False).head(30).to_csv(
            root/'Top30_edgeR_QLF_genes_by_group.csv',index=False)
    z.groupby(['snp','region','cell_type','stratum']).agg(
        n_genes=('gene_id','size'),n_fdr05=('fdr05','sum'),min_p=('pvalue','min'),
        min_fdr=('padj','min'),n_carrier=('n_carrier','first'),
        n_noncarrier=('n_noncarrier','first')).reset_index().to_csv(
            root/'edgeR_QLF_group_summary.csv',index=False)
    print(f'edgeR QLF summary: {len(z)} gene tests; FDR<0.05: {int(z.fdr05.sum())}; {len(files)} comparisons')
    print('Output:',root)



def run_comparison(a):
    """Compare overlapping genes only; FDR values remain from each original method."""
    from scipy.stats import spearmanr
    root=Path(a.out)/'edger'
    rows=[]
    comparisons={'hc3':Path(a.out)/'de','deseq2':Path(a.out)/'de',
                 'dreamlet':Path(a.out)/'dreamlet'/'de'}
    for edgefile in sorted((root/'de').glob('*/*/*/*/edger_qlf.csv')):
        ct=edgefile.parent.name
        region=edgefile.parent.parent.name
        snp=edgefile.parent.parent.parent.name
        stratum=edgefile.parent.parent.parent.parent.name
        ref=pd.read_csv(edgefile,usecols=['gene_id','log2FC','pvalue','padj']).drop_duplicates('gene_id')
        for method,dirbase in comparisons.items():
            filename='dreamlet.csv' if method=='dreamlet' else method+'.csv'
            path=dirbase/stratum/snp/region/ct/filename
            if not path.is_file():continue
            other=pd.read_csv(path,usecols=['gene_id','log2FC','pvalue','padj']).drop_duplicates('gene_id')
            merged=ref.merge(other,on='gene_id',suffixes=('_edger','_'+method),validate='one_to_one')
            valid=merged[['log2FC_edger','log2FC_'+method]].apply(pd.to_numeric,errors='coerce').notna().all(axis=1)
            x=merged.loc[valid,'log2FC_edger'].to_numpy(dtype=float)
            y=merged.loc[valid,'log2FC_'+method].to_numpy(dtype=float)
            rho=float(spearmanr(x,y).statistic) if len(x)>2 and np.ptp(x)>1e-12 and np.ptp(y)>1e-12 else np.nan
            common=(merged.padj_edger.lt(.05)&merged['padj_'+method].lt(.05))
            rows.append(dict(stratum=stratum,snp=snp,region=region,cell_type=ct,
                             compared_to=method,n_edger=len(ref),n_other=len(other),
                             n_shared=len(merged),n_valid_effects=len(x),log2fc_spearman=rho,
                             directional_agreement=float(np.mean(np.sign(x)==np.sign(y))) if len(x) else np.nan,
                             n_fdr05_edger_in_shared=int(merged.padj_edger.lt(.05).sum()),
                             n_fdr05_other_in_shared=int(merged['padj_'+method].lt(.05).sum()),
                             n_fdr05_both=int(common.sum())))
    if not rows:
        print('No comparable edgeR/other method files found. No comparison output written.')
        return
    dest=root/'edgeR_QLF_vs_other_methods.csv'
    pd.DataFrame(rows).to_csv(dest,index=False)
    print('Method comparison:',len(rows),'rows ->',dest)

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['check','fit','summarize','compare'])
    p.add_argument('--out',default=str(Path(os.getenv('ROOT',ROOT_DEFAULT))/'single_iv_causal/postmortem_snp_scDE/carrier_vs_noncarrier_v6_vcf_verified'))
    p.add_argument('--regions',default=','.join(REGIONS))
    p.add_argument('--celltypes',default=','.join(CELLTYPES))
    p.add_argument('--snps',default=','.join(TARGETS))
    p.add_argument('--strata',default='all,case_only')
    p.add_argument('--min-group',type=int,default=5)
    p.add_argument('--min-cells',type=int,default=30)
    p.add_argument('--min-total-count',type=int,default=10)
    p.add_argument('--min-per-donor-count',type=int,default=2)
    p.add_argument('--min-detected-donors',type=int,default=3)
    p.add_argument('--edger-min-count',type=int,default=5,help='edgeR filterByExpr min.count')
    p.add_argument('--edger-min-total',type=int,default=15,help='edgeR filterByExpr min.total.count')
    p.add_argument('--no-robust',action='store_true',help='Disable robust EB quasi-dispersion moderation')
    p.add_argument('--edger-min-prop',type=float,default=0.7,help='edgeR filterByExpr min.prop')
    p.add_argument('--n-cpus',type=int,default=1)
    p.add_argument('--rscript',default=os.getenv('EDGE_RSCRIPT',str(Path.home()/'.conda/envs/2sgps_dreamlet_fix/bin/Rscript')))
    p.add_argument('--max-jobs',type=int,default=0,help='limit attempted jobs (for smoke testing)')
    p.add_argument('--dry-run',action='store_true',help='inspect planned analyses; do not run R or write count bridges')
    p.add_argument('--resume',dest='resume',action='store_true',default=True)
    p.add_argument('--overwrite',dest='resume',action='store_false')
    p.add_argument('--keep-work',action='store_true')
    p.add_argument('--fail-fast',action='store_true')
    return p


def main():
    a=parser().parse_args()
    if a.min_group<2 or a.min_cells<1 or a.n_cpus<1 or a.edger_min_count<1 or a.edger_min_total<1 or not 0<a.edger_min_prop<=1:
        raise ValueError('Invalid filtering / CPU options')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if a.command=='check':r_check(a.rscript)
    elif a.command=='fit':run_fit(a)
    elif a.command=='summarize':run_summary(a)
    elif a.command=='compare':run_comparison(a)

if __name__=='__main__':main()
