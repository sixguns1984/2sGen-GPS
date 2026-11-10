#!/usr/bin/env python3
"""Donor-level pseudobulk genotype-expression associations, PyDESeq2 + robust OLS sensitivity."""
import argparse,json,sys, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from statsmodels.stats.multitest import multipletests
from carrier_utils import allele_definition, carrier_from_dosage

TARGET_SNPS=('rs2253588','rs35162690')
CELL_TYPES=('Oli','Immune','Inh','OPC','Ast','Vast','Exc')
REGIONS=('GPI','DMNX','PFC','PMC','PVC')

def fdr(p):
    arr=np.asarray(p,dtype=float)
    out=np.full(arr.shape,np.nan)
    good=np.isfinite(arr)
    if good.any(): out[good]=multipletests(arr[good],method='fdr_bh')[1]
    return out

def design_matrix(samples, case_only):
    age=pd.to_numeric(samples['age'],errors='coerce').to_numpy(float)
    sex=(samples['sex'].astype(str)=='Male').astype(float).to_numpy()
    group=samples['carrier'].to_numpy(float)
    cols=['Intercept','age_centered','sex_Male']
    pieces=[np.ones(len(age)),age-age.mean(),sex]
    if not case_only:
        pieces.append((samples['status'].astype(str)=='Case').astype(float).to_numpy())
        cols.append('status_Case')
    pieces.append(group);cols.append('carrier')
    X=np.column_stack(pieces)
    if not np.isfinite(X).all() or np.linalg.matrix_rank(X)<X.shape[1]:
        raise ValueError(f'Non-identifiable design: {cols} (check genotype/status/sex confounding)')
    if len(samples)<=len(cols)+5:
        raise ValueError(f'Insufficient independent donors: {len(samples)} with {len(cols)} design terms')
    return X,cols

def robust_logcpm(counts, libsize, x, genes):
    libsize=np.asarray(libsize,dtype=float)
    cpm=counts.astype(float)/libsize[:,None]*1e6
    y=np.log2(cpm+0.5)
    pinv=np.linalg.pinv(x)
    beta=pinv@y
    residual=y-x@beta
    leverage=np.sum(x*(x@np.linalg.inv(x.T@x)),axis=1)
    leverage=np.clip(leverage,0,0.9999999)
    adjusted=residual/(1-leverage[:,None])
    variance=(pinv[-1,:]**2)@(adjusted**2)
    se=np.sqrt(np.maximum(variance,0))
    effect=beta[-1,:]
    t=np.divide(effect,se,out=np.full_like(effect,np.nan),where=se>1e-14)
    p=2*stats.t.sf(np.abs(t),df=len(x)-x.shape[1])
    res=pd.DataFrame({'gene':genes,'HC3_log2CPM_beta_carrier_vs_noncarrier':effect,
                      'HC3_se':se,'HC3_t':t,'HC3_pvalue':p})
    res['HC3_padj']=fdr(res.HC3_pvalue)
    return res

def run_pydeseq2(counts,samples,genes,cpus,case_only):
    import pydeseq2
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    from packaging.version import Version
    ver=Version(pydeseq2.__version__)
    if ver < Version('0.4.0') or ver >= Version('0.5.0'):
        raise RuntimeError(f'This patch is validated for PyDESeq2 0.4.x, installed={ver}. Do not silently change APIs.')
    meta=samples[['age','sex','status','carrier']].copy()
    meta['age_centered']=pd.to_numeric(meta['age']).astype(float)-pd.to_numeric(meta['age']).mean()
    meta['carrier_group']=pd.Categorical(np.where(meta['carrier'].to_numpy(float)>0.5,'Carrier','NonCarrier'),
                                         categories=['NonCarrier','Carrier'])
    meta['sex']=pd.Categorical(meta['sex'].astype(str),categories=['Female','Male'])
    factors=['age_centered','sex']
    if not case_only:
        meta['status']=pd.Categorical(meta['status'].astype(str),categories=['Control','Case'])
        factors.append('status')
    factors.append('carrier_group')
    input_counts=pd.DataFrame(np.asarray(counts,dtype=np.int64),index=meta.index,columns=pd.Index(genes,dtype=str))
    print(f'[PYDESEQ2 {ver}] design_factors={factors} continuous_factors=[age_centered] '
          f'contrast=Carrier/NonCarrier samples={len(meta)} genes={len(genes)}',flush=True)
    dds=DeseqDataSet(counts=input_counts,metadata=meta,design_factors=factors,
                     continuous_factors=['age_centered'],ref_level=['carrier_group','NonCarrier'],
                     n_cpus=cpus,quiet=True)
    dds.deseq2()
    ds=DeseqStats(dds,contrast=['carrier_group','Carrier','NonCarrier'],quiet=True)
    ds.summary()
    res=ds.results_df.reset_index()
    res=res.rename(columns={res.columns[0]:'gene','baseMean':'DESeq2_baseMean',
        'log2FoldChange':'DESeq2_log2FC_carrier_vs_noncarrier','lfcSE':'DESeq2_lfcSE',
        'stat':'DESeq2_stat','pvalue':'DESeq2_pvalue','padj':'DESeq2_padj'})
    return res

def load_pseudobulk(source,region):
    path=source/region
    mat=sp.load_npz(path/'counts.npz').tocsr()
    samples=pd.read_csv(path/'samples.csv',low_memory=False)
    genes=pd.read_csv(path/'genes.csv').gene.astype(str).to_numpy()
    if mat.shape!=(len(samples),len(genes)):
        raise ValueError(f'{region}: pseudobulk shape mismatch {mat.shape} vs {len(samples)} x {len(genes)}')
    return mat,samples,genes

def write_no_fit(out,info,reason):
    out.mkdir(parents=True,exist_ok=True)
    (out/'skip_reason.txt').write_text(str(reason)+'\n')
    info.update(status='SKIPPED',reason=str(reason))
    print('[SKIP]',info.get('region'),info.get('cell_type'),info.get('snp'),reason,flush=True)
    return info

def run_stratum(mat,region_samples,genes,region,cell,snp,args,root,alleles,case_only=False):
    tag='case_only' if case_only else 'adjusted_case_control'
    out=root/'DE_results'/tag/region/cell/snp
    info={'region':region,'cell_type':cell,'snp':snp,'model':tag,'comparison':'ALT_Carrier_vs_NonCarrier'}
    needed={'pydeseq2':'DESeq2_all_genes.csv.gz','hc3':'HC3_all_genes.csv.gz'}
    requested=['hc3','pydeseq2'] if args.methods=='both' else [args.methods]
    if args.resume and all((out/needed[m]).is_file() for m in requested) and (out/'DE_all_genes.csv.gz').is_file():
        info.update(status='ALREADY_DONE')
        print('[RESUME]',region,cell,snp,tag,flush=True)
        return info
    ix=(region_samples['cell_type']==cell) & region_samples[snp].notna()
    if case_only: ix &= region_samples['status'].eq('Case')
    rows=region_samples.loc[ix].copy()
    numeric=mat[ix.to_numpy(),:].toarray().astype(np.int64)
    cfg=alleles[snp]
    alt,carrier=carrier_from_dosage(rows[snp],cfg)
    valid=np.isfinite(carrier)
    rows=rows.iloc[np.flatnonzero(valid)].copy()
    numeric=numeric[valid]
    rows['ALT_dosage']=alt[valid]
    rows['carrier']=carrier[valid].astype(int)
    rows['carrier_group']=np.where(rows.carrier==1,'Carrier','NonCarrier')
    rows.index=pd.Index([f'donor_{i}' for i in range(len(rows))],name='sample')
    n0=int((rows.ALT_dosage==0).sum());n1=int((rows.ALT_dosage==1).sum());n2=int((rows.ALT_dosage==2).sum())
    info.update(n_donors=len(rows),n_case=int(rows.status.eq('Case').sum()),
                n_control=int(rows.status.eq('Control').sum()),n_noncarrier=n0,n_carrier=n1+n2,
                n_alt_het=n1,n_alt_hom=n2,n_excluded_nonhardcall=int((~valid).sum()),
                ALT=cfg['alt'],REF=cfg['ref'],PLINK_counted_allele=cfg['plink_counted_allele'])
    if len(rows)<args.min_donors:return write_no_fit(out,info,f'<{args.min_donors} donors')
    if n0 < args.min_minor_donors or n1+n2 < args.min_minor_donors:
        return write_no_fit(out,info,f'Carrier={n1+n2}, NonCarrier={n0}; require >= {args.min_minor_donors} each')
    if rows.sex.nunique()<2:return write_no_fit(out,info,'Only one sex')
    if not case_only and rows.status.nunique()<2:
        return write_no_fit(out,info,'Only one clinical status (Case/Control required)')
    try:design,cols=design_matrix(rows,case_only)
    except ValueError as exc:return write_no_fit(out,info,exc)
    detected=(numeric>=args.min_count).sum(axis=0)>=args.min_samples
    detected&=numeric.sum(axis=0)>0
    if detected.sum()<2:return write_no_fit(out,info,'No genes passed count filter')
    cc=numeric[:,detected];gg=genes[detected]
    info['n_genes_tested']=len(gg);info['design_terms']=cols
    out.mkdir(parents=True,exist_ok=True)
    rows[['donor','age','sex','status',snp,'ALT_dosage','carrier_group','n_cells','library_size']].to_csv(out/'samples_used.csv')
    tables=[]; failures={}
    for method in requested:
        path=out/needed[method]
        if args.resume and path.exists():
            print('[RESUME METHOD]',region,cell,snp,tag,method,flush=True)
            tables.append(pd.read_csv(path,low_memory=False))
            continue
        try:
            if method=='hc3':
                r=robust_logcpm(cc,rows.library_size.to_numpy(float),design,gg)
            else:
                r=run_pydeseq2(cc,rows,gg,args.cpus,case_only)
            r.to_csv(path,index=False,compression='gzip')
            tables.append(r)
        except Exception as exc:
            failures[method]=traceback.format_exc()
            print('[FAILED]',region,cell,snp,tag,method,repr(exc),flush=True)
    if not tables:
        (out/'errors.txt').write_text('\n'.join(failures.values()))
        info.update(status='FAILED',reason='; '.join(failures));return info
    # Include an existing second method if one is available, even when only running a single method now.
    for method in ('hc3','pydeseq2'):
        path=out/needed[method]
        if path.is_file() and not any(('HC3_pvalue' if method=='hc3' else 'DESeq2_pvalue') in r for r in tables):
            tables.append(pd.read_csv(path,low_memory=False))
    merged=tables[0]
    for other in tables[1:]:merged=merged.merge(other,on='gene',how='outer',validate='one_to_one')
    merged.insert(0,'snp',snp);merged.insert(0,'cell_type',cell);merged.insert(0,'region',region)
    merged.insert(3,'REF',cfg['ref']);merged.insert(4,'ALT',cfg['alt'])
    merged.insert(5,'plink_counted_allele',cfg['plink_counted_allele'])
    merged.insert(6,'comparison','Carrier_vs_NonCarrier')
    merged['both_methods_direction_concordant']=False
    c1='DESeq2_log2FC_carrier_vs_noncarrier';c2='HC3_log2CPM_beta_carrier_vs_noncarrier'
    if {c1,c2}.issubset(merged):
        merged['both_methods_direction_concordant']=(merged[c1]*merged[c2])>0
    order='DESeq2_pvalue' if 'DESeq2_pvalue' in merged else 'HC3_pvalue'
    merged.sort_values(order,na_position='last').to_csv(out/'DE_all_genes.csv.gz',index=False,compression='gzip')
    pd.DataFrame({'gene':['FLOT1','C7orf25']}).merge(merged,on='gene',how='left').to_csv(out/'MRI_candidate_gene_expression.csv',index=False)
    if failures:(out/'errors.txt').write_text('\n'.join(failures.values()))
    else:(out/'errors.txt').unlink(missing_ok=True)
    info.update(status='COMPLETED' if not failures else 'PARTIAL',n_methods=len(tables))
    for q in ('DESeq2_padj','HC3_padj'):
        if q in merged:info['n_'+q+'_lt005']=int((pd.to_numeric(merged[q],errors='coerce')<.05).sum())
    (out/'run_info.json').write_text(json.dumps(info,indent=2,default=str))
    print('[DONE]',region,cell,snp,tag,'n=',len(rows),'n_noncarrier=',n0,
          'n_carrier=',n1+n2,'genes=',len(gg),'methods=',len(tables),flush=True)
    return info

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    p.add_argument('--outdir',default='')
    p.add_argument('--regions',nargs='+',choices=REGIONS,default=list(REGIONS))
    p.add_argument('--cell-types',nargs='+',choices=CELL_TYPES,default=list(CELL_TYPES))
    p.add_argument('--snps',nargs='+',choices=TARGET_SNPS,default=list(TARGET_SNPS))
    p.add_argument('--methods',choices=['both','pydeseq2','hc3'],default='both')
    p.add_argument('--pseudobulk-source',required=True,help='Existing prepared pseudobulk directory')
    p.add_argument('--dosage-alleles',required=True,help='Existing dosage_alleles.json')
    p.add_argument('--cpus',type=int,default=4)
    p.add_argument('--min-donors',type=int,default=15)
    p.add_argument('--min-minor-donors',type=int,default=5)
    p.add_argument('--min-count',type=int,default=10)
    p.add_argument('--min-samples',type=int,default=5)
    p.add_argument('--case-only',action='store_true',help='Run additional Case-only sensitivity model')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    root=Path(a.outdir) if a.outdir else Path(a.root)/'single_iv_causal'/'postmortem_snp_scDE'
    root.mkdir(exist_ok=True,parents=True)
    allelemap={snp:allele_definition(a.dosage_alleles,snp) for snp in a.snps}
    audits=[]
    for region in a.regions:
        mat,meta,genes=load_pseudobulk(Path(a.pseudobulk_source),region)
        for cell in a.cell_types:
            for snp in a.snps:
                for only in ([False,True] if a.case_only else [False]):
                    result=run_stratum(mat,meta,genes,region,cell,snp,a,root,allelemap,only)
                    audits.append(result)
                    pd.DataFrame(audits).to_csv(root/'analysis_progress.csv',index=False)
    print('[DONE] all requested strata; see',root/'analysis_progress.csv',flush=True)

if __name__=='__main__':main()
