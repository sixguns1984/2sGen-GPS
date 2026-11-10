#!/usr/bin/env python3
"""PPMI->PDBP tQTL reproducibility figures from replication_summary.csv."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from figure_common import (CURRENT,OLD,MAIN_METHODS,REVIEW_METHODS,LABELS,COLORS,
                           clean_methods,set_style,tidy,tag,save,audit_methods,write_json)

REQUIRED=['method','n_tqtl_discovery','n_tqtl_matched','n_tqtl_rep_nominal',
          'tqtl_rep_nominal_rate','pi1_tqtl','tqtl_logp_spearman']


def read_real(raw,main):
    if raw.method.duplicated().any():
        raise ValueError(f'Duplicated method rows in real data: {raw[raw.method.duplicated()].method.to_list()}')
    missing=[c for c in REQUIRED if c not in raw]
    if missing:raise ValueError(f'Missing essential columns in replication_summary.csv: {missing}')
    z=clean_methods(raw,main)
    for col in REQUIRED[1:]:z[col]=pd.to_numeric(z[col],errors='coerce')
    z['computed_replication_rate']=z.n_tqtl_rep_nominal/z.n_tqtl_matched.replace({0:np.nan})
    bad=(z['computed_replication_rate']-z.tqtl_rep_nominal_rate).abs()>1e-6
    if bad.any():
        raise ValueError(f'Internal replication rate mismatch for methods {z.loc[bad,"method"].tolist()}')
    z['tqtl_match_fraction']=z.n_tqtl_matched/z.n_tqtl_discovery.replace({0:np.nan})
    z['order']=z.method.map({m:i for i,m in enumerate(REVIEW_METHODS if not main else MAIN_METHODS)})
    z=z.sort_values('order').reset_index(drop=True)
    return z


def metric_axis(ax,z,col,letter,title,xlabel,lim=None,fmt=None,highlight=True):
    y=np.arange(len(z))
    values=z[col].to_numpy(float)
    colors=[COLORS[m] if m in (CURRENT,OLD,'lin2018_regression_cov') else '#8995A0' for m in z.method]
    if highlight:
        ax.barh(y,values,height=.60,color=colors,zorder=3,alpha=.92)
    else:
        ax.scatter(values,y,s=32,color=colors,edgecolors='white',linewidths=.5,zorder=4)
        ax.hlines(y,0,np.nan_to_num(values,nan=0),color='#D2D8DC',lw=1.15,zorder=2)
    ax.invert_yaxis()
    ax.set_yticks(y,labels=[LABELS.get(m,m) for m in z.method])
    ax.set_xlabel(xlabel)
    ax.set_title(title,pad=8,weight='semibold')
    if lim:ax.set_xlim(*lim)
    elif np.isfinite(values).any():ax.set_xlim(0,max(.015,np.nanmax(values)*1.18))
    tidy(ax);ax.grid(axis='x',color='#ECEFF2',lw=.6);ax.grid(axis='y',visible=False)
    tag(ax,letter)
    for i,v in enumerate(values):
        if not np.isfinite(v):continue
        if fmt=='percent':label=f'{v*100:.1f}%'
        elif fmt=='decimal':label=f'{v:.2f}'
        else:label=f'{int(round(v)):,}'
        pad=.007 if lim else (np.nanmax(values)*.012 if np.nanmax(values)>0 else .1)
        ax.text(v+pad,i,label,ha='left',va='center',fontsize=6.7,color='#4F535A')


def real_main_plot(z,out,titlelabel):
    fig,axes=plt.subplots(2,2,figsize=(10.0,max(5.4,3.9+len(z)*.2)))
    y=np.arange(len(z)); ax=axes[0,0]
    back=z.n_tqtl_matched.to_numpy(float); front=z.n_tqtl_rep_nominal.to_numpy(float)
    ax.barh(y,back,height=.7,color='#DFE4E8',label='Tested in PDBP',zorder=2)
    ax.barh(y,front,height=.45,color=[COLORS[m] for m in z.method],
            label='Replicated (P < 0.05)',zorder=3)
    ax.set_yticks(y,labels=[LABELS.get(m,m) for m in z.method]);ax.invert_yaxis()
    ax.set_title('Independent PDBP replication',weight='semibold',pad=8)
    ax.set_xlabel('Number of SNP–gene pairs')
    xmax=max(np.nanmax(back)*1.16,1)
    ax.set_xlim(0,xmax)
    tidy(ax);ax.grid(axis='x',color='#ECEFF2',lw=.6);ax.grid(axis='y',visible=False)
    for i,(t,r) in enumerate(zip(back,front)):
        if np.isfinite(t) and np.isfinite(r):
            ax.text(t+xmax*.012,i,f'{int(r):,}/{int(t):,}',va='center',fontsize=6.8,color='#4C5156')
    tag(ax,'a')
    metric_axis(axes[0,1],z,'tqtl_rep_nominal_rate','b','Replication fraction',
                'Replicated / tested in PDBP',lim=(0,max(1.075,float(z.tqtl_rep_nominal_rate.max())*1.2)),fmt='percent')
    metric_axis(axes[1,0],z,'pi1_tqtl','c','P-value enrichment in PDBP',
                r'$\pi_1$ estimate',lim=(0,1.075),fmt='decimal',highlight=False)
    # Spearman can be negative, use a signed horizontal plot with real zero axis.
    ax=axes[1,1];y=np.arange(len(z));vals=z.tqtl_logp_spearman.to_numpy(float)
    ax.axvline(0,color='#737B84',lw=.65,zorder=1)
    for i,(m,v) in enumerate(zip(z.method,vals)):
        if not np.isfinite(v):continue
        ax.plot([0,v],[i,i],color='#CDD3D8',lw=1.25,zorder=2)
        ax.scatter(v,i,s=32,color=COLORS[m],edgecolors='white',linewidths=.4,zorder=3)
        ax.text(v+(.025 if v>=0 else -.025),i,f'{v:.2f}',fontsize=6.8,
                ha='left' if v>=0 else 'right',va='center',color='#4C5156')
    ax.set_yticks(y,labels=[LABELS.get(m,m) for m in z.method]);ax.invert_yaxis()
    ax.set_xlim(-1.08,1.08);ax.set_xticks(np.arange(-1,1.01,.5))
    ax.set_title('Cross-cohort association concordance',weight='semibold',pad=8)
    ax.set_xlabel(r"Spearman $r$ of $-\log_{10}(P)$")
    tidy(ax);ax.grid(axis='x',color='#ECEFF2',lw=.6);ax.grid(axis='y',visible=False)
    tag(ax,'d')
    fig.text(.5,.005,'PPMI discovery: BH FDR < 0.05; PDBP replication: nominal P < 0.05. '
             'Light bars: tested; colored bars: replicated. Effect direction not assessed.',
             ha='center',va='bottom',fontsize=7,color='#5B626B')
    fig.subplots_adjust(left=.23,right=.982,top=.925,bottom=.11,hspace=.48,wspace=.62)
    save(fig,out)


def real_sup_plot(z,out):
    fig,axes=plt.subplots(1,3,figsize=(10.2,max(3.0,1.9+len(z)*.25)),sharey=True)
    for i,(col,title) in enumerate([
        ('n_tqtl_discovery','PPMI BH discoveries'),
        ('n_tqtl_matched','With PDBP P-value'),
        ('n_tqtl_rep_nominal','PDBP nominal replication')]):
        metric_axis(axes[i],z,col,'abc'[i],title,'SNP–gene pairs')
    fig.subplots_adjust(left=.20,right=.99,top=.86,bottom=.18,wspace=.42)
    save(fig,out)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out-root',required=True)
    a=p.parse_args();set_style();src=Path(a.input);out=Path(a.out_root)
    raw=pd.read_csv(src,low_memory=False)
    audit=audit_methods(raw,src,out/'tables'/'realdata_input_audit.json','PPMI_to_PDBP')
    m=read_real(raw,True)
    m.to_csv(out/'tables'/'realdata_main_methods.csv',index=False)
    real_main_plot(m,out/'main'/'Fig_REAL01_PPMI_PDBP_replication','main')
    real_sup_plot(m,out/'supplementary'/'Fig_REAL_S01_discovery_matched_replication')
    fair=m[m.method.isin([CURRENT,'lin2018_regression_cov'])].copy()
    if len(fair)==2 and ('data_mode' not in fair or fair.data_mode.nunique()==1):
        real_main_plot(fair,out/'supplementary'/'Fig_REAL_S02_current_v89_matched_preprocessing','matched')
    else:
        print('[WARN] Cannot produce matched-current_v89 comparison: current pipeline comparator missing or data_mode differs')
    if OLD in set(raw.method):
        r=read_real(raw,False)
        r.to_csv(out/'tables'/'realdata_review_methods.csv',index=False)
        real_main_plot(r,out/'reviewer'/'Fig_REAL_REVIEW_original_MPTGA_comparison','reviewer')
    else:print('[WARN] Original MPTGA missing, reviewer real-data comparison skipped')
    if 'data_mode' in m.columns and m.data_mode.nunique()>1:
        audit['important_warning']=('Different methods use different data_mode/implementation: '
                                    'current_v89 vs stage1_manuscript. Results are descriptive; '
                                    'do not attribute differences purely to method without harmonized inputs.')
        print('[IMPORTANT]',audit['important_warning'])
    if 'tqtl_match_fraction' in m and m.tqtl_match_fraction.lt(.8).any():
        audit['low_matching_warning']=m.loc[m.tqtl_match_fraction.lt(.8),
                                            ['method','tqtl_match_fraction']].to_dict('records')
        print('[WARN] Low discovery->replication match fractions; denominator must be reported.')
    write_json(out/'tables'/'realdata_input_audit.json',audit)
    write_json(out/'tables'/'realdata_figure_manifest.json',{
        'source':str(src),'main_methods':m.method.tolist(),
        'replication_definition':'discovery BH FDR<0.05, PDBP nominal P<0.05 among matched pairs',
        'effect_direction_concordance':'not available from this summary; do not interpret as validated',
        'scope':'tQTL only; dynamic/additive excluded'})
    print('[REAL DONE]',out)
if __name__=='__main__':main()
