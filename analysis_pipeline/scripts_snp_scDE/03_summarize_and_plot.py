#!/usr/bin/env python3
"""Publication-friendly summary/volcano from independent region x cell-type SNP associations."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REGIONS=('GPI','DMNX','PFC','PMC','PVC')
CELL_TYPES=('Oli','Immune','Inh','OPC','Ast','Vast','Exc')
SNPS=('rs2253588','rs35162690')

def plot_volcano(d, out, method, snp, cell, region, top=10):
    if method=='DESeq2':
        effect='DESeq2_log2FC_carrier_vs_noncarrier';q='DESeq2_padj';p='DESeq2_pvalue'
    else:
        effect='HC3_log2CPM_beta_carrier_vs_noncarrier';q='HC3_padj';p='HC3_pvalue'
    if not {effect,q,p}.issubset(d.columns):return
    x=d[["gene",effect,p,q]].copy().dropna(subset=[effect,p])
    if x.empty:return
    x['l10']=-np.log10(np.maximum(x[p].astype(float),1e-300))
    x['is_sig']=np.where(x[q].fillna(1)<0.05,'BH-FDR < 0.05','Not BH-significant')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':9,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(5.6,4.4))
    color={'Not BH-significant':'#AAB3BF','BH-FDR < 0.05':'#C96D58'}
    for k in ['Not BH-significant','BH-FDR < 0.05']:
        t=x[x.is_sig==k]
        ax.scatter(t[effect],t.l10,s=9 if k=='Not BH-significant' else 12,c=color[k],edgecolors='none',alpha=.6 if k=='Not BH-significant' else .9,rasterized=True)
    x=x.sort_values(p)
    interest=x[x.gene.isin(['FLOT1','C7orf25'])]
    topgenes=pd.concat([x.head(top),interest],ignore_index=True).drop_duplicates('gene')
    for r in topgenes.itertuples(index=False):
        ax.text(float(getattr(r,effect)),float(r.l10)+.09,str(r.gene),fontsize=6.8,ha='center',va='bottom')
    ax.axvline(0,color='#536274',lw=.8)
    ax.set(xlabel='log2FC: ALT carrier vs non-carrier' if method=='DESeq2' else 'log2(CPM) difference: carrier vs non-carrier',ylabel=r'$-\log_{10}(P)$')
    ax.set_title(f'{region} | {cell} | {snp}\n{method} pseudobulk',loc='left',fontweight='bold',fontsize=10)
    ax.grid(axis='y',color='#E5EAF0',lw=.5)
    ax.legend(handles=[Line2D([0],[0],marker='o',linestyle='',label=k,markerfacecolor=v,markeredgecolor='none',markersize=5) for k,v in color.items()],frameon=False,loc='upper right',fontsize=7)
    fig.tight_layout();fig.savefig(out,dpi=250,bbox_inches='tight');plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    p.add_argument('--outdir',default='')
    p.add_argument('--top-labels',type=int,default=6)
    p.add_argument('--all-volcano',action='store_true',help='Make volcano for all strata, not just ones with q<.05')
    a=p.parse_args()
    base=Path(a.outdir) if a.outdir else Path(a.root)/'single_iv_causal'/'postmortem_snp_scDE'
    report=base/'summary';report.mkdir(parents=True,exist_ok=True)
    progress=base/'analysis_progress.csv'
    if not progress.is_file(): raise FileNotFoundError(progress)
    aud=pd.read_csv(progress)
    aud.to_csv(report/'all_stratum_models.csv',index=False)
    rows=[];hits=[];target=[]
    for f in sorted((base/'DE_results').glob('*/*/*/*/DE_all_genes.csv.gz')):
        ds=f.parts[-5];region=f.parts[-4];cell=f.parts[-3];snp=f.parts[-2]
        d=pd.read_csv(f,low_memory=False)
        d['model']=ds
        for m,q in [('DESeq2','DESeq2_padj'),('HC3','HC3_padj')]:
            if q not in d:continue
            yes=d[d[q]<.05].copy()
            hits.append(yes.assign(method=m))
            rows.append(dict(model=ds,region=region,cell_type=cell,snp=snp,method=m,
                             n_genes=len(d),n_q005=len(yes),n_nominal_p005=int((d[m+'_pvalue']<.05).sum())))
            if len(yes) or a.all_volcano:
                plot_volcano(d,report/f'volcano_{ds}_{region}_{cell}_{snp}_{m}.pdf',m,snp,cell,region,a.top_labels)
        target.append(d[d.gene.isin(['FLOT1','C7orf25'])].copy())
    if rows:pd.DataFrame(rows).to_csv(report/'number_of_differentially_expressed_genes.csv',index=False)
    if hits:pd.concat(hits,ignore_index=True).to_csv(report/'DE_all_FDR_significant.csv.gz',index=False,compression='gzip')
    if target:pd.concat(target,ignore_index=True).to_csv(report/'FLOT1_C7orf25_direct_expression_results.csv',index=False)
    print('[DONE] report folder:',report)
    if rows:print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
