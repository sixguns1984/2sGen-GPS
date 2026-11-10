#!/usr/bin/env python3
"""Recreate the original real-data CTAR AR coefficient + SNP-gene expression figure.

Restored from the original Stage-I Figure v4 plotting code. Uses only frozen Stage-I
PPMI and PDBP association results plus observed expression and genotype data.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figure_common import set_style,tidy,tag,save,write_json

METHOD='submitted_followup_ctar_mptga'
KEYS=['Variant_ID','Ensembl_ID']
COLORS={0:'#828E9D',1:'#287C90',2:'#BF4A59'}
# SNPs and symbols transcribed from the user's surviving published-style figure.
DEFAULT_FIXED=[('Low','XRRA1','rs10899051'),
               ('Intermediate','MTCO2P2','rs10445485'),
               ('High','PSPHP1','rs10260993')]


def read_fields(path,required,optional=()):
    import pyarrow.parquet as pq
    if not path.exists(): raise FileNotFoundError(f'Missing real Stage1 result: {path}')
    cols=pq.ParquetFile(path).schema.names
    absent=set(required)-set(cols)
    if absent:raise KeyError(f'{path}: missing required columns: {sorted(absent)}')
    use=list(dict.fromkeys(list(required)+[c for c in optional if c in cols]))
    return pd.read_parquet(path,columns=use)


def validated_pairs(root,rep_alpha=.05):
    base=Path(root)/'qtl'
    p=base/'PP'/METHOD/'final'/'significant_tqtl.parquet'
    if not p.exists():p=base/'PP'/METHOD/'final'/'significant_temporal.parquet'
    r=base/'PD'/METHOD/'final'/'all_results.parquet'
    dis=read_fields(p,KEYS+['Pvalue','rho'],['FDR','phi','rho_reference_interval'])
    rep=read_fields(r,KEYS+['Pvalue'],['rho','phi'])
    if dis.duplicated(KEYS).any() or rep.duplicated(KEYS).any():
        raise RuntimeError('Stage1 pair keys duplicated; abort instead of silently merging')
    m=dis.merge(rep,on=KEYS,how='inner',validate='one_to_one',suffixes=('_PP','_PD'))
    m['Pvalue_PD']=pd.to_numeric(m['Pvalue_PD'],errors='coerce')
    m=m.loc[m['Pvalue_PD'].lt(rep_alpha)].copy()
    m['rho_PP']=pd.to_numeric(m['rho_PP'],errors='coerce')
    m=m.loc[np.isfinite(m.rho_PP)].copy()
    if m.empty:raise RuntimeError('No PPMI FDR significant / PDBP P<0.05 CTAR SNP-gene pairs with finite rho')
    # rho in CTAR is a residual covariance parameter.  Do not interpret as SNP effect coefficient.
    if not m.rho_PP.between(0,1,inclusive='both').all():
        raise RuntimeError('Found rho outside [0,1]; inspect Stage1 CTAR parameterization')
    return m.sort_values(KEYS).reset_index(drop=True),p,r


def full_stage1_meta(paths):
    from gengps.amp_pd.real_qtl import (load_cohort_metadata,_augment_fixed_genotype_pcs,
        _prepare_irregular_followup_rows,method_covariates,_drop_missing_covariates)
    meta=load_cohort_metadata(paths,'PP')
    meta=_augment_fixed_genotype_pcs(paths,'PP',meta,n_pcs=8)
    empty=pd.DataFrame(index=meta.index)
    meta,empty,_=_prepare_irregular_followup_rows(meta,empty)
    covars=method_covariates(METHOD,'PP',meta,cell_adjust=True)
    meta,empty=_drop_missing_covariates(meta,empty,covars)
    return meta


def candidates_by_quantile(m, n_each=40):
    out=[]
    for qlabel,q in [('Low',.15),('Intermediate',.50),('High',.85)]:
        target=float(m.rho_PP.quantile(q))
        d=m.assign(_dist=(m.rho_PP-target).abs())
        d=d.sort_values(['_dist','Pvalue_PD','Variant_ID','Ensembl_ID']).head(n_each)
        out.append((qlabel,target,d))
    return out


def find_fixed_pairs(paths, pairs, spec=DEFAULT_FIXED):
    """Resolve the three documented figure examples by rsID + HGNC gene symbol.

    Stage-I association tables use Ensembl IDs. Do not try to match gene symbols
    directly to the Ensembl column; instead reuse the official v8.9 mapping.
    """
    from gengps.amp_pd.real_qtl import resolve_expression_gene_columns
    snps={str(v) for _,_,v in spec}
    cand=pairs.loc[pairs['Variant_ID'].astype(str).isin(snps)].copy()
    if cand.empty:
        raise RuntimeError('Requested example SNPs absent from PPMI significant / PDBP replicated results. '
                           'If outputs or criteria changed, use --selection quantile to reselect real examples.')
    mapping=resolve_expression_gene_columns(paths,cand.Ensembl_ID.astype(str).unique().tolist(),
                                            allow_unresolved=True)
    records=[]
    for label,gene,snp in spec:
        one=cand[cand.Variant_ID.astype(str).eq(str(snp))].copy()
        one=one[one.Ensembl_ID.astype(str).map(lambda s: mapping.get(s,'').upper()==str(gene).upper())]
        if len(one)!=1:
            choices=[f"{g}: {mapping.get(g,'?')}" for g in cand.loc[cand.Variant_ID.astype(str).eq(str(snp)),'Ensembl_ID'].astype(str).head(12)]
            raise RuntimeError(f'Fixed {label} example {gene}/{snp} not uniquely resolved among replicated pairs; '
                               f'found={len(one)}, candidate Ensembl->symbol={choices}. '
                               'Use --selection quantile if Stage-I tables have changed.')
        records.append((label,one.iloc[0].copy()))
    return records


def annotate_variant_info(z,row,geno,allele_df):
    snp=str(row['Variant_ID'])
    g=pd.to_numeric(geno[snp],errors='coerce') if snp in geno.columns else pd.Series(dtype=float)
    g=g[np.isfinite(g)]
    maf=float('nan')
    if len(g):
        p=float(g.mean()/2.0)
        maf=min(p,1-p)
    aa=allele_df.loc[allele_df['Variant_ID'].astype(str).eq(snp)]
    if len(aa):
        a1=str(aa.iloc[0]['A1'])
        a2=str(aa.iloc[0]['A2'])
    else:
        a1='A1'; a2='A2'
    meta={'A1':a1,'A2':a2,'MAF':maf,
          'genotype_labels':{0:f'{a2}/{a2}',1:f'{a1}/{a2}',2:f'{a1}/{a1}'}}
    if z is not None:
        for k,v in meta.items():
            z.attrs[k]=v
    return meta


def select_fixed(records,meta,expr,geno,genemap,allele_df,min_genotype_donors=5):
    selected=[]; traces=[]; audit=[]
    for label,row in records:
        z=sample_pair(meta,expr,geno,genemap,row,min_genotype_donors=min_genotype_donors)
        if z is None:
            raise RuntimeError(f'{label} example {row.Variant_ID}/{row.Ensembl_ID} lacks two genotype groups '
                               f'with at least {min_genotype_donors} donors or enough valid expression visits. '
                               'To choose alternatives from the current data, run --selection quantile.')
        target=float(row.rho_PP)
        vinfo=annotate_variant_info(z,row,geno,allele_df)
        selected.append((label,target,row,z,vinfo))
        traces.append(z.assign(Variant_ID=str(row.Variant_ID),Ensembl_ID=str(row.Ensembl_ID),strength_group=label))
        audit.append({'strength_group':label,'quantile_target_rho':float('nan'),
                      'PPMI_rho':float(row.rho_PP),
                      'PDBP_rho':float(row.rho_PD) if 'rho_PD' in row.index and pd.notna(row.rho_PD) else float('nan'),
                      'PDBP_P':float(row.Pvalue_PD),'Variant_ID':str(row.Variant_ID),
                      'Ensembl_ID':str(row.Ensembl_ID),
                      'gene_symbol':genemap.get(str(row.Ensembl_ID)),
                      'n_donors':int(z.participant_id.nunique()),'n_observations':len(z),
                      'timepoints':sorted(z.time.unique().tolist()),
                      'genotype_donor_counts':z.groupby('genotype').participant_id.nunique().to_dict()})
    return selected,pd.concat(traces,ignore_index=True),pd.DataFrame(audit)


def read_bim_alleles(bfile):
    b=Path(str(bfile)+'.bim')
    if not b.exists():
        raise FileNotFoundError(f'Missing BIM file for allele annotation: {b}')
    bim=pd.read_csv(b,sep='\t',header=None,names=['chr','Variant_ID','cm','pos','A1','A2'])
    bim['Variant_ID']=bim['Variant_ID'].astype(str)
    return bim[['Variant_ID','A1','A2']].drop_duplicates('Variant_ID')


def get_observed_data(paths,meta,pairs):
    from gengps.amp_pd.real_qtl import resolve_expression_gene_columns,load_expression_columns,bfile_for_cohort
    from gengps.amp_pd.genotype import make_genotype_source
    genes=sorted(set(pairs.Ensembl_ID.astype(str)))
    snps=sorted(set(pairs.Variant_ID.astype(str)))
    gene_map=resolve_expression_gene_columns(paths,genes,allow_unresolved=True)
    if not gene_map:raise RuntimeError('No replicated tQTL Ensembl IDs map to the existing expression matrix')
    expr=load_expression_columns(paths,list(gene_map))
    pids=sorted(meta.participant_id.astype(str).unique())
    bfile=bfile_for_cohort(paths,'PP')
    src=make_genotype_source(str(bfile),plink19=paths.plink19,prefer_bed_reader=True)
    # No MAF/HWE/missing-rate/LD QC; only read previously Stage1-validated SNPs.
    geno=src.read(pids,snps,mean_impute=False,dtype=np.float64)
    geno.index=geno.index.astype(str)
    allele_df=read_bim_alleles(str(bfile))
    return expr,geno,gene_map,str(bfile),allele_df


def sample_pair(meta,expr,geno,genemap,row,min_genotype_donors=5):
    gene=str(row['Ensembl_ID']);snp=str(row['Variant_ID'])
    if gene not in genemap or gene not in expr.columns or snp not in geno.columns:return None
    z=meta[['sample_id','participant_id','followup_time']].copy()
    z['participant_id']=z.participant_id.astype(str)
    z['sample_id']=z.sample_id.astype(str)
    z['time']=pd.to_numeric(z.followup_time,errors='coerce')
    z['expression']=pd.to_numeric(expr[gene].reindex(z.sample_id).to_numpy(),errors='coerce')
    g=geno[snp]
    z['genotype']=pd.to_numeric(z.participant_id.map(g),errors='coerce')
    z=z[np.isfinite(z.time)&np.isfinite(z.expression)&np.isfinite(z.genotype)].copy()
    if z.empty:return None
    z=z[(z.genotype-z.genotype.round()).abs()<1e-5].copy()
    z['genotype']=z.genotype.round().astype(int)
    z=z[z.genotype.isin([0,1,2])].copy()
    if z.empty:return None
    groups=z.groupby('genotype').participant_id.nunique()
    accepted=groups[groups>=min_genotype_donors]
    if len(accepted)<2 or z.time.nunique()<3: return None
    z=z[z.genotype.isin(accepted.index)].copy()
    # Check informative means across >2 timepoints, do not impute missing visits.
    if any(v.time.nunique()<2 for _,v in z.groupby('genotype')):return None
    return z


def select_three(pairs,meta,expr,geno,genemap,allele_df,n_per_group=40):
    selected=[];traces=[];used_pairs=set();used_genes=set();audit=[]
    for label,target,group in candidates_by_quantile(pairs,n_per_group):
        options=[]
        for _,row in group.iterrows():
            pair=(str(row.Variant_ID),str(row.Ensembl_ID))
            if pair in used_pairs:continue
            z=sample_pair(meta,expr,geno,genemap,row)
            if z is None:continue
            options.append((0 if pair[1] not in used_genes else 1,row,z))
        if not options:
            raise RuntimeError(f'No verifiable {label} rho example with >=2 genotype classes, >=5 donors each, and >=3 follow-up times. '
                               'Try --search-per-quantile 100; do not substitute synthetic data.')
        options.sort(key=lambda v:v[0])
        _,row,z=options[0]
        vinfo=annotate_variant_info(z,row,geno,allele_df)
        selected.append((label,target,row,z,vinfo))
        traces.append(z.assign(Variant_ID=str(row.Variant_ID),Ensembl_ID=str(row.Ensembl_ID),strength_group=label))
        used_pairs.add((str(row.Variant_ID),str(row.Ensembl_ID)));used_genes.add(str(row.Ensembl_ID))
        audit.append({'strength_group':label,'quantile_target_rho':target,
                      'PPMI_rho':float(row.rho_PP),
                      'PDBP_rho':float(row.rho_PD) if 'rho_PD' in row.index and pd.notna(row.rho_PD) else float('nan'),
                      'PDBP_P':float(row.Pvalue_PD),
                      'Variant_ID':str(row.Variant_ID),'Ensembl_ID':str(row.Ensembl_ID),
                      'n_donors':int(z.participant_id.nunique()),'n_observations':len(z),
                      'timepoints':sorted(z.time.unique().tolist()),
                      'genotype_donor_counts':z.groupby('genotype').participant_id.nunique().to_dict()})
    return selected,pd.concat(traces,ignore_index=True),pd.DataFrame(audit)


def histogram(ax,pairs,chosen):
    rho=pairs.rho_PP.to_numpy(float)
    ax.hist(rho,bins=np.linspace(0,1,31),density=False,color='#92A9BE',
            edgecolor='white',lw=.5)
    for (label,_,row,_,_),color in zip(chosen,['#767F89','#267F90','#BF4A59']):
        ax.axvline(float(row.rho_PP),color=color,lw=1.6,ls='--',label=f'{label}: {row.rho_PP:.2f}')
    ax.set_xlim(0,1);ax.set_xlabel(r'PPMI CTAR residual $\rho$ (1-year reference)')
    ax.set_ylabel('Replicated SNP–gene pairs')
    ax.set_title('Residual AR coefficient distribution',fontsize=8.6,fontweight='semibold',loc='left',pad=8)
    ax.legend(frameon=False,fontsize=7,loc='upper left')
    tidy(ax);tag(ax,'a')


def expression_panel(ax, record, genemap, letter):
    label,target,row,z,vinfo=record
    grp=z.groupby(['genotype','time'],observed=True).expression.agg(['mean','sem','size']).reset_index()
    for g, zz in z.groupby('genotype'):
        color=COLORS[int(g)]
        # A fixed small set of true individual donor trajectories is shown faintly.
        for pid, sub in list(zz.groupby('participant_id'))[:12]:
            sub=sub.sort_values('time')
            if len(sub)>=2:
                ax.plot(sub.time,sub.expression,color=color,alpha=.105,lw=.58,zorder=1)
        v=grp[grp.genotype.eq(g)].sort_values('time')
        x=v.time.to_numpy(float);y=v['mean'].to_numpy(float)
        se=v['sem'].fillna(0).to_numpy(float)
        ax.plot(x,y,marker='o',markersize=4.2,lw=1.6,color=color,label=f'A1 copies = {int(g)}',zorder=4)
        ax.fill_between(x,y-se,y+se,color=color,alpha=.14,lw=0,zorder=3)
    gene=str(row.Ensembl_ID)
    gene_symbol=genemap.get(gene,gene)
    rep_rho=(float(row.rho_PD) if 'rho_PD' in row.index and pd.notna(row.rho_PD) else None)
    maf=vinfo.get('MAF', float('nan'))
    allele_text=f"{vinfo.get('A2','A2')} > {vinfo.get('A1','A1')}"
    maf_text=(f"MAF={maf:.2f}" if np.isfinite(maf) else "MAF=NA")
    rho_text=(rf'$\rho_{{PPMI}}$={float(row.rho_PP):.2f}; $\rho_{{PDBP}}$={rep_rho:.2g}'
              if rep_rho is not None else
              rf'$\rho_{{PPMI}}$={float(row.rho_PP):.2f}; $P_{{PDBP}}$={float(row.Pvalue_PD):.2g}')
    ax.set_title(f'{label} AR: {str(gene_symbol)[:18]} / {str(row.Variant_ID)[:18]} ({allele_text})\n'
                 + f'{maf_text}; ' + rho_text, fontsize=8.0,fontweight='semibold',loc='left',pad=8)
    ax.set_xlabel('Follow-up time (years)')
    ax.set_ylabel('Expression (z-scored log2 TPM)')
    tidy(ax);tag(ax,letter)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    a.add_argument('--code',default=None)
    a.add_argument('--out-root',default=None)
    a.add_argument('--replication-p',type=float,default=.05)
    a.add_argument('--search-per-quantile',type=int,default=40)
    a.add_argument('--selection',choices=['fixed','quantile'],default='fixed',
                   help='fixed: reproduce XRRA1/MTCO2P2/PSPHP1 from screenshot; quantile: original automatic selection')
    a.add_argument('--min-genotype-donors',type=int,default=5,
                   help='minimum donors per displayed genotype class (default 5)')
    x=a.parse_args();set_style()
    root=Path(x.root);out=Path(x.out_root) if x.out_root else root/'manuscript_figures'/'tqtl_benchmark_v4'
    code=Path(x.code) if x.code else Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(code))
    from gengps.amp_pd.config import AmpPdPaths
    paths=AmpPdPaths(root=str(root))
    pairs, pp_path, pd_path=validated_pairs(root,x.replication_p)
    print(f'[VALIDATED] PPMI FDR-significant + PDBP P<{x.replication_p}: {len(pairs):,} unique SNP–gene pairs',flush=True)
    meta=full_stage1_meta(paths)
    if x.selection=='fixed':
        fixed=find_fixed_pairs(paths,pairs)
        selected_candidates=pd.DataFrame([row for _,row in fixed]).reset_index(drop=True)
    else:
        selected_candidates=pd.concat([sub for _,_,sub in candidates_by_quantile(pairs,x.search_per_quantile)],ignore_index=True)
    expr,geno,genemap,bfile,allele_df=get_observed_data(paths,meta,selected_candidates)
    if x.selection=='fixed':
        selected,long,examples=select_fixed(fixed,meta,expr,geno,genemap,allele_df,x.min_genotype_donors)
    else:
        selected,long,examples=select_three(pairs,meta,expr,geno,genemap,allele_df,x.search_per_quantile)
    fig,aa=plt.subplots(2,2,figsize=(11.2,7.7))
    histogram(aa[0,0],pairs,selected)
    for ax,rec,ltr in zip([aa[0,1],aa[1,0],aa[1,1]],selected,'bcd'):
        expression_panel(ax,rec,genemap,ltr)
    handles=[Line2D([0],[0],color=color,lw=1.6,marker='o',markersize=4,
                    label=lab) for g,color,lab in [
                        (0,COLORS[0],'0 = A2/A2'),
                        (1,COLORS[1],'1 = A1/A2'),
                        (2,COLORS[2],'2 = A1/A1')
                    ]]
    fig.legend(handles=handles,frameon=False,ncol=3,loc='lower center',bbox_to_anchor=(.5,.025),fontsize=8)
    fig.subplots_adjust(left=.09,right=.985,top=.945,bottom=.135,wspace=.42,hspace=.80)
    save(fig,out/'supplementary'/'FigS_tQTL_verified_AR_distribution_and_3_expression_trajectories')
    out.joinpath('tables').mkdir(parents=True,exist_ok=True)
    pairs.to_csv(out/'tables'/'FigS_validated_pair_rho.csv',index=False)
    examples.to_csv(out/'tables'/'FigS_three_real_examples.csv',index=False)
    gr=long.groupby(['strength_group','Variant_ID','Ensembl_ID','genotype','time'],observed=True).expression.agg(['mean','sem','size']).reset_index()
    gr.to_csv(out/'tables'/'FigS_example_expression_group_means.csv',index=False)
    write_json(out/'tables'/'FigS_AR_manifest.json',{
        'method':METHOD,'discovery':str(pp_path),'replication':str(pd_path),
        'expression':str(root/'data/expression/amp_pd_expression_z.parquet'),
        'metadata':str(root/'data/metadata/amp_pd_samples.parquet'),
        'genotype':bfile,'selection':'PPMI FDR significant, PDBP nominal P<0.05 exact same SNP–gene; 15/50/85th percentile rho',
        'note':'rho is residual CTAR autocorrelation, not genotype interaction effect; plotted means and SEM at observed times only',
        'n_replicated_pairs':len(pairs),'n_candidate_genes_resolved':len(genemap),'examples':examples.to_dict('records')})
    print('[FIGS DONE] real replicated tQTL AR distribution + examples',flush=True)
    print('[OUTPUT]',out/'supplementary'/'FigS_tQTL_verified_AR_distribution_and_3_expression_trajectories.pdf',flush=True)
if __name__=='__main__':main()
