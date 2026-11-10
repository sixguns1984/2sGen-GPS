#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse,json,multiprocessing as mp,os,sys,hashlib,traceback
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from gengps.amp_pd.config import AmpPdPaths
from gengps.amp_pd.genotype import make_genotype_source
from gengps.amp_pd.real_qtl import bfile_for_cohort,load_cohort_metadata,load_expression_for_meta,method_covariates,_augment_fixed_genotype_pcs,_drop_missing_covariates,_prepare_irregular_followup_rows,resolve_expression_gene_columns
from gengps.methods.ctar_mptga import _prepare_layout,_ctar_whiten_numpy
from gengps.utils import bh_fdr,genotype_channels
from gengps.amp_pd.vqtl import make_balanced_fold_map,participant_fluctuation,adjusted_scale_test,matrix_rank
KEYS=['Variant_ID','Ensembl_ID'];MODEL='submitted_followup_ctar_mptga';_WORKER=None


def _prepare_worker(root,n_folds,fold_seed,min_visits,min_group_n):
    paths=AmpPdPaths(root=root);meta=load_cohort_metadata(paths,'PP');meta=_augment_fixed_genotype_pcs(paths,'PP',meta,n_pcs=8)
    dummy=pd.DataFrame({'__dummy':np.zeros(len(meta),float)},index=meta.index);meta,dummy,ndups=_prepare_irregular_followup_rows(meta,dummy)
    covars=method_covariates(MODEL,'PP',meta,cell_adjust=True);meta,dummy=_drop_missing_covariates(meta,dummy,covars)
    t=pd.to_numeric(meta['followup_time'],errors='raise').to_numpy(float);B=np.column_stack([np.ones_like(t),t,t*t,t*t*t])
    C=meta[covars].apply(pd.to_numeric,errors='raise').to_numpy(float) if covars else np.empty((len(meta),0),float);X0=np.column_stack([B,C])
    ids=meta['participant_id'].astype(str).to_numpy();pid_order=pd.Index(pd.unique(ids),dtype=str);pid_to_row={p:i for i,p in enumerate(pid_order)};obs_to_part=np.fromiter((pid_to_row[x] for x in ids),dtype=int,count=len(ids))
    fold_map=make_balanced_fold_map(pid_order,n_folds=n_folds,seed=fold_seed);obs_fold=np.fromiter((fold_map[x] for x in ids),dtype=int,count=len(ids));part_fold=np.asarray([fold_map[x] for x in pid_order],int)
    pmean=meta.assign(__pid=meta['participant_id'].astype(str)).groupby('__pid',sort=False)[covars].mean(numeric_only=True).reindex(pid_order) if covars else pd.DataFrame(index=pid_order)
    nvis=pd.Series(obs_to_part).value_counts().reindex(range(len(pid_order)),fill_value=0).sort_index().to_numpy(float)
    Wp=np.column_stack([np.ones(len(pid_order),float),pmean.to_numpy(float) if pmean.shape[1] else np.empty((len(pid_order),0),float),nvis[:,None]])
    bfile=bfile_for_cohort(paths,'PP');gsrc=make_genotype_source(str(bfile),paths.plink19,prefer_bed_reader=True)
    return {'paths':paths,'meta':meta,'covars':covars,'t':t,'B':B,'X0':X0,'ids':ids,'pid_order':pid_order,'obs_to_part':obs_to_part,'obs_fold':obs_fold,'part_fold':part_fold,'Wp':Wp,'gsrc':gsrc,'rho_cache':{},'n_folds':int(n_folds),'min_visits':int(min_visits),'min_group_n':int(min_group_n),'n_duplicate_time_rows_removed':int(ndups)}


def _init_worker(root,n_folds,fold_seed,min_visits,min_group_n):
    global _WORKER;_WORKER=_prepare_worker(root,n_folds,fold_seed,min_visits,min_group_n)


def _fold_whiten_cache(fold,rho,reference_interval):
    global _WORKER
    key=(int(fold),round(float(rho),8),round(float(reference_interval),8))
    if key not in _WORKER['rho_cache']:
        tr=_WORKER['obs_fold']!=int(fold);m=_WORKER['meta'].loc[tr].copy();layout=_prepare_layout(m,'participant_id','followup_time')
        Xw,_=_ctar_whiten_numpy(_WORKER['X0'][tr],layout,float(rho),float(reference_interval));Bw,_=_ctar_whiten_numpy(_WORKER['B'][tr],layout,float(rho),float(reference_interval))
        _WORKER['rho_cache'][key]=(tr,layout,Xw,Bw)
    return _WORKER['rho_cache'][key]


def _crossfit_full_mean_residual(y,gpart,rho,reference_interval):
    global _WORKER
    g=np.asarray(gpart,float);finite=np.isfinite(g);hard=finite&np.isclose(g,np.rint(g),atol=1e-8,rtol=0)&(g>=0)&(g<=2)
    if np.any(finite&~hard):raise ValueError(f'non-missing genotype values are not hard calls: {np.unique(g[finite&~hard])[:10].tolist()}')
    if hard.sum()<30:raise ValueError(f'too few participants with non-missing hard-call genotype: {int(hard.sum())}')
    channels_part=np.full((len(g),2),np.nan,float);channels_part[hard]=genotype_channels(g[hard][None,:],'binary')[0];channels_obs=channels_part[_WORKER['obs_to_part']]
    valid_obs=hard[_WORKER['obs_to_part']];Zraw=np.full((len(y),8),np.nan,float);Zraw[valid_obs]=(channels_obs[valid_obs,:,None]*_WORKER['B'][valid_obs,None,:]).reshape(valid_obs.sum(),8);oof=np.full(len(y),np.nan,float)
    all_complete=bool(hard.all())
    for fold in range(_WORKER['n_folds']):
        te=valid_obs&(_WORKER['obs_fold']==fold);tr=valid_obs&(_WORKER['obs_fold']!=fold)
        if te.sum()==0:continue
        if all_complete:
            tr_cached,layout,Xw,Bw=_fold_whiten_cache(fold,rho,reference_interval)
            if not np.array_equal(tr,tr_cached):raise RuntimeError('internal fold-cache mismatch')
        else:
            layout=_prepare_layout(_WORKER['meta'].loc[tr].copy(),'participant_id','followup_time');Xw,_=_ctar_whiten_numpy(_WORKER['X0'][tr],layout,float(rho),float(reference_interval));Bw,_=_ctar_whiten_numpy(_WORKER['B'][tr],layout,float(rho),float(reference_interval))
        Zw=(channels_obs[tr,:,None]*Bw[:,None,:]).reshape(tr.sum(),8);yw,_=_ctar_whiten_numpy(np.asarray(y,float)[tr],layout,float(rho),float(reference_interval));Dw=np.column_stack([Xw,Zw]);beta=np.linalg.pinv(Dw,rcond=1e-10)@yw;Dte=np.column_stack([_WORKER['X0'][te],Zraw[te]]);oof[te]=np.asarray(y,float)[te]-Dte@beta
    return oof,channels_part,hard


def _group_summary(var,g):
    gg=np.asarray(g,float);valid=np.isfinite(gg)&np.isclose(gg,np.rint(gg),atol=1e-8,rtol=0);hard=np.full(len(gg),-1,int);hard[valid]=np.rint(gg[valid]).astype(int);out={}
    for k in [0,1,2]:
        vv=np.asarray(var,float)[hard==k];vv=vv[np.isfinite(vv)];out[f'n_g{k}']=int(len(vv));out[f'median_var_g{k}']=float(np.median(vv)) if len(vv) else np.nan
    return out


def _analyze_pair(row,y,gpart):
    global _WORKER
    rho=float(row['rho']);ref=float(row.get('rho_reference_interval',1.0));resid,channels_part,geno_hard_mask=_crossfit_full_mean_residual(y,gpart,rho,ref)
    var,mad,nvis=participant_fluctuation(resid,_WORKER['obs_to_part'],len(_WORKER['pid_order']),min_visits=_WORKER['min_visits']);pos=np.isfinite(var)&(var>0);eps=max(float(np.nanmedian(var[pos]))*1e-8 if pos.any() else 1e-12,1e-12);logvar=np.log(var+eps)
    wp=_WORKER['Wp'];add=adjusted_scale_test(logvar,gpart,wp,primary='additive',min_group_n=_WORKER['min_group_n']);grp=adjusted_scale_test(logvar,gpart,wp,primary='genotypic',min_group_n=_WORKER['min_group_n'])
    mad_pos=np.isfinite(mad)&(mad>0);meps=max(float(np.nanmedian(mad[mad_pos]))*1e-8 if mad_pos.any() else 1e-12,1e-12);logmad=np.log(mad+meps);madtest=adjusted_scale_test(logmad,gpart,wp,primary='additive',min_group_n=_WORKER['min_group_n'])
    return {'Variant_ID':str(row['Variant_ID']),'Ensembl_ID':str(row['Ensembl_ID']),'rho_ctar':rho,'rho_reference_interval':ref,'n_obs':int(len(y)),'n_participants':int(len(_WORKER['pid_order'])),'n_genotype_nonmissing':int(geno_hard_mask.sum()),'n_genotype_missing':int((~geno_hard_mask).sum()),'n_participants_variance':int(np.isfinite(logvar).sum()),'median_visits':float(np.median(nvis[nvis>=_WORKER['min_visits']])) if np.any(nvis>=_WORKER['min_visits']) else np.nan,'crossfit_resid_sd':float(np.nanstd(resid,ddof=1)),'fluct_add_wald':add['stat'],'fluct_add_p':add['p'],'fluct_add_df':add['df'],'fluct_add_beta':add['beta'],'fluct_genotypic_wald':grp['stat'],'fluct_genotypic_p':grp['p'],'fluct_genotypic_df':grp['df'],'fluct_genotypic_min_group_n':grp['min_group_n'],'mad_add_wald':madtest['stat'],'mad_add_p':madtest['p'],'mad_add_df':madtest['df'],'mad_add_beta':madtest['beta'],**_group_summary(var,gpart)}


def _analyze_batch(batch):
    global _WORKER
    rows=pd.DataFrame(batch);genes=rows['Ensembl_ID'].astype(str).drop_duplicates().tolist();gene_map=resolve_expression_gene_columns(_WORKER['paths'],genes,allow_unresolved=True);usable=[g for g in genes if g in gene_map]
    if not usable:return [],[{'Ensembl_ID':str(r['Ensembl_ID']),'Variant_ID':str(r['Variant_ID']),'error':'expression_gene_unresolved'} for r in batch]
    expr=load_expression_for_meta(_WORKER['paths'],_WORKER['meta'],usable);union_snps=rows.loc[rows['Ensembl_ID'].astype(str).isin(usable),'Variant_ID'].astype(str).drop_duplicates().tolist();gmat=_WORKER['gsrc'].read(_WORKER['pid_order'].tolist(),union_snps,mean_impute=False,dtype=np.float64)
    results=[];errors=[]
    for gene in genes:
        if gene not in usable:
            errors.extend([{'Ensembl_ID':gene,'Variant_ID':str(x),'error':'expression_gene_unresolved'} for x in rows.loc[rows['Ensembl_ID'].astype(str).eq(gene),'Variant_ID']]);continue
        y=pd.to_numeric(expr[gene],errors='coerce').to_numpy(float)
        if not np.isfinite(y).all():errors.extend([{'Ensembl_ID':gene,'Variant_ID':str(x),'error':'nonfinite_expression'} for x in rows.loc[rows['Ensembl_ID'].astype(str).eq(gene),'Variant_ID']]);continue
        for _,rr in rows[rows['Ensembl_ID'].astype(str).eq(gene)].iterrows():
            snp=str(rr['Variant_ID'])
            try:gpart=pd.to_numeric(gmat[snp],errors='coerce').to_numpy(float);results.append(_analyze_pair(rr.to_dict(),y,gpart))
            except Exception as e:errors.append({'Ensembl_ID':gene,'Variant_ID':snp,'error':repr(e)})
    return results,errors




def _read_pairs(a):
    root=Path(a.root)
    eligible=Path(a.eligible_csv) if a.eligible_csv else root/'single_iv_causal'/'iv_validation'/'single_iv_primary_eligible.csv'
    ctarp=Path(a.ctar_parquet) if a.ctar_parquet else root/'qtl'/'PP'/MODEL/'final'/'significant_tqtl.parquet'
    if not eligible.is_file():raise FileNotFoundError(f'Final Stage-II eligible pairs not found: {eligible}')
    if not ctarp.is_file():raise FileNotFoundError(f'Final PPMI Stage-I CTAR table not found: {ctarp}')
    d=pd.read_csv(eligible,low_memory=False,dtype={'lead_snp':str,'gene':str})
    if not {'gene','lead_snp'}.issubset(d.columns):raise ValueError(f'{eligible} missing gene/lead_snp; columns={list(d.columns)}')
    if 'iv_primary_eligible' in d.columns:
        flag=d.iv_primary_eligible.astype(str).str.lower().isin(['true','1','yes'])
        if not flag.all():raise ValueError(f'Eligible CSV contains {int((~flag).sum())} unapproved IV rows; expected final primary eligible file')
    d=d.rename(columns={'gene':'Ensembl_ID','lead_snp':'Variant_ID'})
    d=d[KEYS].copy()
    if d.duplicated(KEYS).any():raise ValueError(f'Final eligible IV file has {int(d.duplicated(KEYS).sum())} duplicate SNP–gene keys; refusing silent de-duplication')
    if not len(d):raise ValueError('No eligible IV-gene pairs')
    if not a.gene_list and not a.max_genes and a.expected_pairs and len(d)!=a.expected_pairs:
        raise ValueError(f'Expected {a.expected_pairs:,} final IV-gene pairs, but found {len(d):,} in {eligible}. Check the source/version, or explicitly set --expected-pairs 0 to accept a different frozen final table.')
    import pyarrow.parquet as pq
    available=set(pq.ParquetFile(ctarp).schema.names)
    cols=KEYS+['rho']+(['rho_reference_interval'] if 'rho_reference_interval' in available else [])
    if not set(KEYS+['rho']).issubset(available):raise ValueError(f'CTAR table {ctarp} lacks required fields: {set(KEYS+["rho"])-available}')
    c=pd.read_parquet(ctarp,columns=cols)
    c['Ensembl_ID']=c['Ensembl_ID'].astype(str);c['Variant_ID']=c['Variant_ID'].astype(str)
    if c.duplicated(KEYS).any():
        repeated=c.loc[c.duplicated(KEYS,keep=False),KEYS].head(10).to_dict('records')
        raise ValueError(f'CTAR table has duplicate SNP-gene keys: {repeated}')
    pair=d.merge(c,on=KEYS,how='left',validate='one_to_one',indicator=True)
    missing=pair['_merge'].ne('both')|~np.isfinite(pd.to_numeric(pair['rho'],errors='coerce'))
    if missing.any():
        raise ValueError(f'{int(missing.sum()):,} eligible IV-gene pairs missing fitted CTAR rho in {ctarp}. Examples: {pair.loc[missing,KEYS].head(10).to_dict("records")}. Refusing to silently drop pairs.')
    pair=pair.drop(columns=['_merge'])
    if ((pd.to_numeric(pair['rho'],errors='coerce')<0)|(pd.to_numeric(pair['rho'],errors='coerce')>=1)).any():raise ValueError('CTAR rho must be in [0,1) for whitening')
    if 'rho_reference_interval' not in pair.columns:pair['rho_reference_interval']=1.0
    pair['rho_reference_interval']=pd.to_numeric(pair['rho_reference_interval'],errors='coerce').fillna(1.0)
    if ((pair['rho_reference_interval']<=0)|(~np.isfinite(pair['rho_reference_interval']))).any():raise ValueError('Bad reference interval')
    subset=False
    if a.gene_list:
        filt={s.strip() for s in Path(a.gene_list).read_text().splitlines() if s.strip()}
        pair=pair[pair.Ensembl_ID.isin(filt)].copy();subset=True
    if a.max_genes>0:
        keep=set(pair.Ensembl_ID.drop_duplicates().iloc[:a.max_genes]);pair=pair[pair.Ensembl_ID.isin(keep)].copy();subset=True
    if pair.empty:raise ValueError('No SNP-gene pairs remain in selected subset')
    pair=pair.sort_values(['Ensembl_ID','Variant_ID'],kind='mergesort').reset_index(drop=True)
    return pair,eligible,ctarp,subset


def _hash_run(pair,eligible,ctar,a):
    h=hashlib.sha256()
    # Contains full exact pair list, covariances, source metadata and settings.
    h.update(pair.to_csv(index=False).encode())
    for path in [eligible,ctar]:
        st=path.stat();h.update(f'{path.resolve()}|{st.st_size}|{st.st_mtime_ns}'.encode())
    h.update(json.dumps({'folds':a.n_folds,'seed':a.fold_seed,'visits':a.min_fluctuation_visits,'group_n':a.min_genotype_group_n},sort_keys=True).encode())
    return h.hexdigest()


def _write_batch(path,res,err):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps({'results':res,'errors':err},allow_nan=True),encoding='utf-8')
    os.replace(tmp,path)


def _fdr_with_missing_as_one(s):
    x=pd.to_numeric(s,errors='coerce').to_numpy(float)
    p=np.where(np.isfinite(x)&(x>=0)&(x<=1),x,1.0)
    # Every frozen eligible pair enters the family; untestable comparisons count as P=1.
    return bh_fdr(p)


def main():
    ap=argparse.ArgumentParser(description='Residual-variance QTL testing of ALL final CTAR Stage-I/Stage-II eligible temporal cis-eQTL SNP-gene pairs (no dynamic-QTL requirement)')
    ap.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    ap.add_argument('--eligible-csv',default='',help='Default: ROOT/single_iv_causal/iv_validation/single_iv_primary_eligible.csv')
    ap.add_argument('--ctar-parquet',default='',help='Default: ROOT/qtl/PP/submitted_followup_ctar_mptga/final/significant_tqtl.parquet')
    ap.add_argument('--expected-pairs',type=int,default=22842,help='Safety assertion on final input, 0 disables')
    ap.add_argument('--out-dir',default='',help='Default: ROOT/qtl/PP/submitted_followup_ctar_mptga/final/vqtl_final_22842')
    ap.add_argument('--n-jobs',type=int,default=8);ap.add_argument('--genes-per-task',type=int,default=10)
    ap.add_argument('--alpha',type=float,default=.05);ap.add_argument('--n-folds',type=int,default=5);ap.add_argument('--fold-seed',type=int,default=1)
    ap.add_argument('--min-fluctuation-visits',type=int,default=3);ap.add_argument('--min-genotype-group-n',type=int,default=10)
    ap.add_argument('--gene-list',default='');ap.add_argument('--max-genes',type=int,default=0)
    ap.add_argument('--restart',action='store_true',help='Clear previous batch results and begin afresh')
    ap.add_argument('--preflight-only',action='store_true',help='Validate 22,842 pair matching and CTAR rho without running vQTL fits')
    a=ap.parse_args()
    if a.n_jobs<1 or a.genes_per_task<1 or a.n_folds<2:raise ValueError('n-jobs>=1, genes-per-task>=1 and n-folds>=2 required')
    pair,eligible,ctar,subset=_read_pairs(a)
    if a.preflight_only:
        print(f'[PREFLIGHT PASS] matched_pairs={len(pair):,}, unique_genes={pair.Ensembl_ID.nunique():,}, cohort=PPMI, subset={subset}')
        print(f'[IV] {eligible}\n[CTAR] {ctar}')
        return
    out=Path(a.out_dir) if a.out_dir else Path(a.root)/'qtl'/'PP'/MODEL/'final'/'vqtl_final_22842'
    out.mkdir(parents=True,exist_ok=True)
    batches_dir=out/'batches';batches_dir.mkdir(exist_ok=True)
    manifest=out/'run_fingerprint.json';fingerprint=_hash_run(pair,eligible,ctar,a)
    if a.restart:
        for f in batches_dir.glob('batch_*.json'):f.unlink()
        if manifest.exists():manifest.unlink()
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old.get('fingerprint')!=fingerprint:raise RuntimeError(f'Existing checkpoint belongs to a DIFFERENT input/parameter set: {out}. Use --restart to intentionally replace it, or use a new --out-dir.')
    else:
        if list(batches_dir.glob('batch_*.json')):raise RuntimeError('Checkpoint files present without fingerprint; use --restart or another out-dir')
        manifest.write_text(json.dumps({'fingerprint':fingerprint,'eligible_csv':str(eligible),'ctar_parquet':str(ctar),'expected_pairs':a.expected_pairs,'actual_pairs':len(pair),'genes':int(pair.Ensembl_ID.nunique()),'subset':subset,'parameters':vars(a)},indent=2),encoding='utf-8')
    genes=pair.Ensembl_ID.drop_duplicates().tolist();batches=[]
    for start in range(0,len(genes),a.genes_per_task):
        chunk=set(genes[start:start+a.genes_per_task]);batches.append(pair[pair.Ensembl_ID.isin(chunk)].to_dict('records'))
    pending=[(i,b) for i,b in enumerate(batches) if not (batches_dir/f'batch_{i:06d}.json').exists()]
    print(f'[FINAL TQTL vQTL] eligible={len(pair):,}, genes={len(genes):,}, jobs={a.n_jobs}, batches={len(batches)}, pending={len(pending)}, subset={subset}',flush=True)
    print(f'[INPUT] {eligible}\n[CTAR] {ctar}\n[OUT] {out}',flush=True)
    if subset:print('[SMOKE] Subset-level FDR is diagnostic only, not a manuscript-wide result.',flush=True)
    if pending:
        if a.n_jobs==1:
            _init_worker(a.root,a.n_folds,a.fold_seed,a.min_fluctuation_visits,a.min_genotype_group_n)
            for j,(i,batch) in enumerate(pending,1):
                res,err=_analyze_batch(batch)
                _write_batch(batches_dir/f'batch_{i:06d}.json',res,err)
                if j%10==0 or j==len(pending):print(f'[CHECKPOINT] {j}/{len(pending)} new batches finished',flush=True)
        else:
            ctx=mp.get_context('spawn')
            with ctx.Pool(a.n_jobs,initializer=_init_worker,initargs=(a.root,a.n_folds,a.fold_seed,a.min_fluctuation_visits,a.min_genotype_group_n)) as pool:
                # imap preserves batch positions, allowing correct resume keys.
                for j,(i,result) in enumerate(zip([x[0] for x in pending],pool.imap(_analyze_batch,[x[1] for x in pending],chunksize=1)),1):
                    res,err=result;_write_batch(batches_dir/f'batch_{i:06d}.json',res,err)
                    if j%10==0 or j==len(pending):print(f'[CHECKPOINT] {j}/{len(pending)} new batches finished',flush=True)
    results=[];errors=[]
    for i in range(len(batches)):
        path=batches_dir/f'batch_{i:06d}.json'
        if not path.is_file():raise RuntimeError(f'Missing checkpoint {path}; do not compute FDR from a partially completed run')
        x=json.loads(path.read_text());results.extend(x.get('results',[]));errors.extend(x.get('errors',[]))
    res=pd.DataFrame(results)
    if res.empty:raise RuntimeError(f'No results. Sample errors: {errors[:5]}')
    if res.duplicated(KEYS).any():raise RuntimeError('Duplicate result keys')
    joined=pair.merge(res,on=KEYS,how='left',validate='one_to_one')
    if len(joined)!=len(pair):raise RuntimeError('Lost eligible pairs during merge')
    pcols=['fluct_add_p','fluct_genotypic_p','mad_add_p']
    for pcol in pcols:
        if pcol not in joined.columns:joined[pcol]=np.nan
        qcol=pcol.replace('_p','_fdr')
        joined[qcol]=_fdr_with_missing_as_one(joined[pcol])
    alpha=a.alpha
    tests=[('crossfit_log_variance_additive_PRIMARY','fluct_add_p','fluct_add_fdr'),('crossfit_log_variance_genotypic_SENSITIVITY','fluct_genotypic_p','fluct_genotypic_fdr'),('crossfit_log_MAD_additive_SENSITIVITY','mad_add_p','mad_add_fdr')]
    summ=[]
    for label,pcol,qcol in tests:
        testable=np.isfinite(pd.to_numeric(joined[pcol],errors='coerce'))
        sig=testable&(pd.to_numeric(joined[qcol],errors='coerce')<alpha)
        summ.append({'test':label,'n_eligible_pairs':len(joined),'n_tested_pairs':int(testable.sum()),'n_untestable_pairs':int((~testable).sum()),'n_vqtl_bh_fdr_lt_0_05':int(sig.sum()),'fraction_of_all_eligible':float(sig.sum()/len(joined)),'fraction_of_tested':float(sig.sum()/testable.sum()) if testable.sum() else np.nan,'n_genes':int(joined.loc[sig,'Ensembl_ID'].nunique()),'fdr_family':'all_frozen_eligible_pairs_with_untestable_p_as_1'})
    joined['primary_vqtl_fdr05']=np.isfinite(joined.fluct_add_p)&(joined.fluct_add_fdr<alpha)
    joined['genotypic_vqtl_fdr05']=np.isfinite(joined.fluct_genotypic_p)&(joined.fluct_genotypic_fdr<alpha)
    joined['mad_vqtl_fdr05']=np.isfinite(joined.mad_add_p)&(joined.mad_add_fdr<alpha)
    joined['tested_primary']=np.isfinite(joined.fluct_add_p)
    joined.to_parquet(out/'final_22842_tqtl_vqtl_results.parquet',index=False,compression='zstd')
    joined.to_csv(out/'final_22842_tqtl_vqtl_results.csv.gz',index=False,compression='gzip')
    pd.DataFrame(summ).to_csv(out/'vqtl_overlap_summary.csv',index=False)
    pd.DataFrame(errors,columns=KEYS+['error']).to_csv(out/'errors.csv',index=False)
    failed=joined[~joined.tested_primary][KEYS].merge(pd.DataFrame(errors,columns=KEYS+['error']),on=KEYS,how='left')
    failed.to_csv(out/'untestable_primary_pairs.csv',index=False)
    overview={'scope':'frozen final eligible temporal cis-eQTL SNP-gene pairs; NOT the historical dynamic subset','cohort':'PPMI','candidate_csv':str(eligible),'ctar_parquet':str(ctar),'n_pairs':int(len(pair)),'n_genes':int(pair.Ensembl_ID.nunique()),'subset_mode':subset,'n_primary_tested':int(joined.tested_primary.sum()),'n_primary_untestable':int((~joined.tested_primary).sum()),'n_primary_significant':int(joined.primary_vqtl_fdr05.sum()),'primary_fraction_all':float(joined.primary_vqtl_fdr05.mean()),'primary_test':'genotype dosage association with log donor-level sample variance of out-of-fold CTAR mean residuals; HC3 variance adjusted for donor-mean nuisance covariates and visit count','mean_residuals':'5-fold donor-disjoint full CTAR mean refit; original Stage-I full-model rho held fixed across folds','multiple_testing':'BH across all frozen eligible pairs; untestable P values set to 1, not classified as negative vQTL evidence','fdr_reporting':'subset-only adjusted values MUST NOT be quoted as full-panel evidence','note':'No downstream IV candidate or Stage-II master table modified.'}
    (out/'vqtl_sensitivity_metadata.json').write_text(json.dumps(overview,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\n=== FINAL 22,842 tQTL vs vQTL summary ===');print(pd.DataFrame(summ).to_string(index=False))
    print(f'\nOutput directory: {out}')
    if (~joined.tested_primary).any():print(f'[WARNING] {int((~joined.tested_primary).sum())} pair(s) untestable: inspect untestable_primary_pairs.csv',flush=True)
    if subset:print('[SMOKE] Never quote subset FDR percentages in response letter.')


if __name__=='__main__':main()
