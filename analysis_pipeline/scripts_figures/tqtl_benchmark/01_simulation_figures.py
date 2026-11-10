#!/usr/bin/env python3
"""Nature Genetics-style simulation benchmarking; reads real aggregated metrics only."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from figure_common import (CURRENT,OLD,MAIN_METHODS,REVIEW_METHODS,LABELS,COLORS,MARKERS,
                           clean_methods,parse_simulation,ci,set_style,tidy,tag,save,audit_methods,write_json)

METRICS={'power_FDR05':('Detection power','BH FDR < 0.05',None),
         'type1_p05':('False-positive rate','Null P < 0.05',0.05),
         'empirical_fdr_FDR05':('Empirical FDR','Among BH FDR discoveries',0.05),
         'auprc':('Area under PR curve','Across all simulated pairs',None),
         'auc':('AUROC','Across all simulated pairs',None),
         'runtime_sec':('Runtime per simulation (s)','Original benchmarking configuration',None)}


def draw_series(ax,d,methods,metric,state):
    if metric+'_mean' not in d.columns:return False
    found=False
    for m in methods:
        v=d[(d.method==m)&(d.confounding_status==state)].sort_values('rho')
        if v.empty:continue
        y,l,h=ci(v,metric)
        x=v.rho.to_numpy(float)
        keep=np.isfinite(x)&np.isfinite(y)
        if not keep.any():continue
        x=x[keep];y=y[keep];l=l[keep];h=h[keep]
        ax.plot(x,y,marker=MARKERS[m],markersize=4.3,color=COLORS[m],
                markeredgecolor='white',markeredgewidth=.35,
                linewidth=2.15 if m==CURRENT else 1.2,
                alpha=1 if m==CURRENT else .88,zorder=4 if m==CURRENT else 3)
        ax.fill_between(x,l,h,color=COLORS[m],alpha=.105 if m==CURRENT else .055,lw=0)
        found=True
    ref=METRICS[metric][2]
    if ref is not None:
        ax.axhline(ref,color='#676B71',lw=.8,ls=(0,(3,3)),zorder=1)
    tidy(ax)
    ax.set_xlabel(r'Residual autocorrelation ($\rho$)')
    ax.set_ylabel(METRICS[metric][0])
    xticks=sorted(d.rho.dropna().unique())
    ax.set_xticks(xticks)
    if len(xticks)>6:ax.tick_params(axis='x',rotation=45)
    if metric!='runtime_sec':
        ax.set_ylim(bottom=-.006)
        if metric in ('power_FDR05','auprc','auc'):ax.set_ylim(-.018,1.025)
        elif found:
            max_high=[]
            for m in methods:
                v=d[(d.method==m)&(d.confounding_status==state)]
                col=metric+'_ci95_high'
                if col in v:max_high+=v[col].dropna().astype(float).tolist()
            top=max(.07,max(max_high)*1.19) if max_high else .12
            ax.set_ylim(-.003,min(1.0,top))
    return found


def legend(fig,methods,y):
    items=[Line2D([0],[0],color=COLORS[m],marker=MARKERS[m],lw=2.1 if m==CURRENT else 1.25,
                  markersize=5,label=LABELS[m]) for m in methods]
    fig.legend(handles=items,loc='lower center',ncol=min(4,len(items)),
               bbox_to_anchor=(.5,y),frameon=False,columnspacing=1.3,handlelength=2.1)


def six_panel(df,methods,metrics,outpath,subtitle=None):
    states=('unconfounded','confounded')
    fig,axes=plt.subplots(2,3,figsize=(10.0,5.45))
    for j,state in enumerate(states):
        for k,metric in enumerate(metrics):
            ax=axes[j,k]
            valid=draw_series(ax,df,methods,metric,state)
            title=METRICS[metric][0]
            ax.set_title(title+('' if valid else ' (unavailable)'),weight='semibold',pad=8)
            tag(ax,'abcdef'[3*j+k])
            if k==0:
                ax.text(-.31,.5,'No confounding' if j==0 else 'With confounding',
                        rotation=90,transform=ax.transAxes,ha='center',va='center',
                        fontsize=8.5,color='#494C50',weight='bold')
    if subtitle:fig.suptitle(subtitle,x=.50,y=.997,fontsize=10)
    legend(fig,methods,-.013)
    fig.subplots_adjust(left=.093,right=.985,top=.915 if not subtitle else .895,
                        bottom=.165,wspace=.39,hspace=.47)
    save(fig,outpath)


def four_panel(df,methods,metrics,outpath):
    fig,axes=plt.subplots(2,2,figsize=(7.25,5.35))
    for j,state in enumerate(('unconfounded','confounded')):
        for k,metric in enumerate(metrics):
            ax=axes[j,k];draw_series(ax,df,methods,metric,state)
            ax.set_title(METRICS[metric][0],weight='semibold',pad=7)
            tag(ax,'abcd'[j*2+k])
            if k==0:ax.text(-.34,.5,'No confounding' if j==0 else 'With confounding',
                            transform=ax.transAxes,rotation=90,ha='center',va='center',weight='bold',fontsize=8)
    legend(fig,methods,-.02)
    fig.subplots_adjust(left=.12,right=.98,top=.94,bottom=.18,wspace=.39,hspace=.46)
    save(fig,outpath)


def runtime_plot(df,methods,outpath):
    if 'runtime_sec_mean' not in df:return False
    sub=df[df.confounding_status=='confounded']
    if sub['runtime_sec_mean'].notna().sum()==0:return False
    fig,ax=plt.subplots(figsize=(6.6,3.25))
    draw_series(ax,sub,methods,'runtime_sec','confounded')
    ax.set_ylim(bottom=0);ax.set_ylabel('Runtime per simulation (s)')
    legend(fig,methods,-.04)
    fig.subplots_adjust(bottom=.28,left=.12,right=.98,top=.94)
    save(fig,outpath)
    return True


def output_metric_table(df,methods,out,tagname):
    z=df[df.method.isin(methods)].copy()
    cols=['scenario','rho','confounding_status','method','n_replicates']
    cols+=[k for m in METRICS for k in (m+'_mean',m+'_ci95_low',m+'_ci95_high') if k in z]
    z[[c for c in cols if c in z.columns]].to_csv(out/tagname,index=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True)
    p.add_argument('--out-root',required=True)
    a=p.parse_args();set_style()
    src=Path(a.input);out=Path(a.out_root)
    raw=pd.read_csv(src,low_memory=False)
    audit_methods(raw,src,out/'tables'/'simulation_input_audit.json','simulation')
    df=parse_simulation(raw)
    main=clean_methods(df,True);review=clean_methods(df,False)
    for m in ('power_FDR05','type1_p05','empirical_fdr_FDR05'):
        if m+'_mean' not in main:raise RuntimeError(f'Required main simulation metric missing: {m}_mean')
    methods=[m for m in MAIN_METHODS if m in set(main.method)]
    six_panel(main,methods,['power_FDR05','type1_p05','empirical_fdr_FDR05'],
              out/'main'/'Fig_SIM01_power_type1_FDR')
    if {'auprc_mean','auc_mean'}.issubset(main.columns):
        four_panel(main,methods,['auprc','auc'],out/'supplementary'/'Fig_SIM_S01_AUPRC_AUROC')
    else:print('[WARN] AUPRC or AUC absent: skip supplementary AUPRC/AUROC figure')
    runtime_plot(main,methods,out/'supplementary'/'Fig_SIM_S02_runtime')
    output_metric_table(main,methods,out/'tables','simulation_main_methods.csv')
    output_metric_table(review,[m for m in REVIEW_METHODS if m in set(review.method)],out/'tables',
                        'simulation_review_methods.csv')
    if OLD in set(review.method):
        rm=[m for m in (CURRENT,OLD,'lin2018_regression_cov','lin2018_mptga') if m in set(review.method)]
        six_panel(review,rm,['power_FDR05','type1_p05','empirical_fdr_FDR05'],
                  out/'reviewer'/'Fig_SIM_REVIEW_original_MPTGA_comparison')
    else:print('[WARN] Original MPTGA unavailable; reviewer-specific simulation figure skipped')
    data={'source':str(src),'main_methods':methods,
          'review_methods':[m for m in REVIEW_METHODS if m in set(review.method)],
          'simulation_scenarios':sorted(df.scenario.astype(str).unique().tolist()),
          'note':'No dynamic/additive panels; unadjusted linear regression and __nocov aliases excluded.'}
    write_json(out/'tables'/'simulation_figure_manifest.json',data)
    print('[SIM DONE]',out)
if __name__=='__main__':main()
