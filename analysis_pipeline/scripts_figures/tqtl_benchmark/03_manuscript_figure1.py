#!/usr/bin/env python3
"""Figure 1: a–d original real replication benchmark, e–g covariate simulation."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from figure_common import (
    CURRENT, MAIN_METHODS, LABELS, COLORS, MARKERS, REGRESSION_METHODS,
    COMBINATION_METHODS, GROUP_TEXT_COLORS, clean_methods, parse_simulation,
    ci, set_style, tidy, tag, save, write_json, audit_methods,
)

REQUIRED = ['method', 'n_tqtl_discovery', 'n_tqtl_matched', 'n_tqtl_rep_nominal',
            'tqtl_rep_nominal_rate', 'pi1_tqtl', 'tqtl_logp_spearman']
METRICS = [('auc', 'AUROC', None), ('type1_p05', 'False-positive rate', .05),
           ('empirical_fdr_FDR05', 'Empirical FDR', .05)]


def get_real(csv):
    raw = pd.read_csv(csv, low_memory=False)
    missing = sorted(set(REQUIRED) - set(raw.columns))
    if missing: raise ValueError(f'Real replication summary missing {missing}')
    z = clean_methods(raw, only_main=True)
    if z.method.duplicated().any(): raise ValueError('Duplicated method in real summary')
    for c in REQUIRED[1:]: z[c] = pd.to_numeric(z[c], errors='coerce')
    rate = z.n_tqtl_rep_nominal / z.n_tqtl_matched.replace(0, np.nan)
    if ((rate-z.tqtl_rep_nominal_rate).abs() > 1e-6).any():
        raise ValueError('Replicated/matched counts disagree with reported replication fraction')
    z['_sort'] = z.method.map({m: i for i,m in enumerate(MAIN_METHODS)})
    return z.sort_values('_sort').reset_index(drop=True)


def labels(ax, z):
    ax.set_yticks(np.arange(len(z)), [LABELS.get(m,m) for m in z.method])
    for lab, m in zip(ax.get_yticklabels(), z.method):
        if m in REGRESSION_METHODS:
            lab.set_color(GROUP_TEXT_COLORS['regression']); lab.set_fontweight('semibold')
        elif m in COMBINATION_METHODS:
            lab.set_color(GROUP_TEXT_COLORS['combination'])
    ax.invert_yaxis()


def horiz(ax, z, column, title, xlab, marker='bar', fmt='decimal', bounds=None):
    vals = z[column].to_numpy(float); yy=np.arange(len(z))
    if marker == 'bar':
        ax.barh(yy, vals, height=.61, color=[COLORS[m] for m in z.method], zorder=3)
    else:
        ax.hlines(yy, 0, vals, lw=1.2, color='#CDD3D9', zorder=2)
        ax.scatter(vals,yy,color=[COLORS[m] for m in z.method],s=29,zorder=4,
                   edgecolor='white',linewidth=.45)
    labels(ax,z); tidy(ax)
    ax.grid(axis='x',color='#E7EBF0',lw=.6); ax.grid(axis='y',visible=False)
    ax.set_title(title,loc='left',fontweight='semibold',fontsize=8.8,pad=9)
    ax.set_xlabel(xlab)
    if bounds is not None: ax.set_xlim(*bounds)
    elif np.isfinite(vals).any(): ax.set_xlim(0,max(float(np.nanmax(vals))*1.19,.05))
    span=ax.get_xlim()[1]-ax.get_xlim()[0]
    for y,v in enumerate(vals):
        if not np.isfinite(v): continue
        t=f'{v*100:.1f}%' if fmt=='pct' else f'{v:.2f}'
        ax.text(v+.017*span,y,t,ha='left',va='center',fontsize=7,color='#4F5560')


def real_panels(axes,z):
    a,b,c,d=axes
    yy=np.arange(len(z))
    matched=z.n_tqtl_matched.to_numpy(float)
    replicated=z.n_tqtl_rep_nominal.to_numpy(float)
    a.barh(yy,matched,height=.67,color='#DCE2E8',zorder=2)
    a.barh(yy,replicated,height=.44,color=[COLORS[m] for m in z.method],zorder=3)
    labels(a,z); tidy(a)
    a.grid(axis='x',color='#E7EBF0',lw=.6);a.grid(axis='y',visible=False)
    a.set_xlim(0,max(1,float(np.nanmax(matched))*1.27))
    a.set_title('Independent PDBP replication',loc='left',fontweight='semibold',fontsize=8.8,pad=9)
    a.set_xlabel('Number of SNP–gene pairs')
    for i,(m,r) in enumerate(zip(matched,replicated)):
        if np.isfinite(m) and np.isfinite(r):
            a.text(m+a.get_xlim()[1]*.013,i,f'{int(r):,}/{int(m):,}',fontsize=6.8,va='center',color='#4C535B')
    a.legend(handles=[Patch(color='#DCE2E8',label='Matched'),
                      Patch(color='#537C9C',label='Replicated')],
             frameon=False,fontsize=6.7,loc='lower right')
    horiz(b,z,'tqtl_rep_nominal_rate','Replication fraction','Replicated / tested in PDBP',fmt='pct',bounds=(0,1.13))
    horiz(c,z,'pi1_tqtl',r'P-value enrichment in PDBP',r'$\pi_1$ estimate',marker='dot',bounds=(0,1.13))
    r=z.tqtl_logp_spearman.to_numpy(float); y=np.arange(len(z))
    d.axvline(0,color='#727B83',lw=.7)
    for i,(m,v) in enumerate(zip(z.method,r)):
        if not np.isfinite(v):continue
        d.plot([0,v],[i,i],color='#C9D0D7',lw=1.25,zorder=2)
        d.scatter(v,i,s=29,color=COLORS[m],edgecolors='white',linewidths=.45,zorder=3)
        d.text(v+(.035 if v>=0 else -.035),i,f'{v:.2f}',ha='left' if v>=0 else 'right',
               va='center',fontsize=7,color='#4F5560')
    labels(d,z); tidy(d)
    d.grid(axis='x',color='#E7EBF0',lw=.6);d.grid(axis='y',visible=False)
    d.set_xlim(-1.13,1.13);d.set_xticks([-1,-.5,0,.5,1])
    d.set_title('Cross-cohort association concordance',loc='left',fontweight='semibold',fontsize=8.8,pad=9)
    d.set_xlabel(r"Spearman $r$ of $-\log_{10}(P)$")
    for letter,ax in zip('abcd',axes):tag(ax,letter)


def sim_panel(ax, sim, methods, metric, title, ref, letter):
    # STRICTLY keep the WITH-COVARIATE (confounded) data scenario.
    d=sim[sim.confounding_status.eq('confounded')]
    if d.empty: raise ValueError('No simulation rows with confounded/with-covariate scenario')
    if metric+'_mean' not in d:raise ValueError(f'Simulation missing {metric}_mean')
    plotted=0
    for m in methods:
        v=d[d.method.eq(m)].sort_values('rho')
        if v.empty:continue
        y,lo,hi=ci(v,metric);x=v.rho.to_numpy(float)
        good=np.isfinite(x)&np.isfinite(y)
        if not good.any():continue
        x=x[good];y=y[good];lo=lo[good];hi=hi[good]
        ax.plot(x,y,color=COLORS[m],marker=MARKERS[m],markersize=3.7,
                markeredgecolor='white',markeredgewidth=.3,
                lw=2.0 if m==CURRENT else 1.14,zorder=4 if m==CURRENT else 3)
        ax.fill_between(x,lo,hi,color=COLORS[m],alpha=.09 if m==CURRENT else .045,lw=0)
        plotted+=1
    if not plotted:raise RuntimeError(f'No finite {metric} values in covariate simulation rows')
    if ref is not None: ax.axhline(ref,lw=.8,color='#686F77',ls=(0,(3,3)),zorder=1)
    tidy(ax)
    ax.set_title(title,loc='left',fontsize=8.8,fontweight='semibold',pad=9)
    ax.set_ylabel(title);ax.set_xlabel(r'Residual autocorrelation ($\rho$)')
    ax.set_xticks(sorted(d.rho.dropna().unique()))
    if metric=='auc':ax.set_ylim(-.015,1.015)
    else:
        hi=np.nanmax(pd.to_numeric(d[metric+'_mean'],errors='coerce').to_numpy(float))
        if metric+'_ci95_high' in d:
            hi=max(hi,float(np.nanmax(pd.to_numeric(d[metric+'_ci95_high'],errors='coerce'))))
        ax.set_ylim(-.006,max(.09,hi*1.15))
    tag(ax,letter)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--real',required=True);p.add_argument('--simulation',required=True)
    p.add_argument('--out-root',required=True)
    a=p.parse_args();set_style();out=Path(a.out_root)
    z=get_real(a.real)
    raw_sim=pd.read_csv(a.simulation,low_memory=False)
    audit_methods(raw_sim,a.simulation,out/'tables'/'Fig1_simulation_input_audit.json','simulation')
    sim=clean_methods(parse_simulation(raw_sim),only_main=True)
    sim=sim[sim.confounding_status.eq('confounded')].copy()
    if not len(sim):raise RuntimeError('No confounded/covariate simulation data: refusing to draw noncov')
    methods=[m for m in MAIN_METHODS if m in set(sim.method)]
    if not methods:raise RuntimeError('No permitted methods in simulation results')
    fig=plt.figure(figsize=(12.0,10.45))
    grid=GridSpec(3,6,figure=fig,height_ratios=[1,1,.93],hspace=.52,wspace=.88,
                  left=.224,right=.980,top=.972,bottom=.145)
    axes=[fig.add_subplot(grid[0,:3]),fig.add_subplot(grid[0,3:]),
          fig.add_subplot(grid[1,:3]),fig.add_subplot(grid[1,3:])]
    real_panels(axes,z)
    for letter,seg, (metric,title,ref) in zip('efg',[(0,2),(2,4),(4,6)],METRICS):
        sim_panel(fig.add_subplot(grid[2,seg[0]:seg[1]]),sim,methods,metric,title,ref,letter)
    leg=[Line2D([0],[0],color=COLORS[m],marker=MARKERS[m],lw=2.0 if m==CURRENT else 1.1,
                markersize=5,label=LABELS[m]) for m in methods]
    fig.legend(handles=leg,loc='lower center',bbox_to_anchor=(.60,.05),frameon=False,
               ncol=4,columnspacing=1.5,handlelength=2.0,fontsize=7.5)
    fig.text(.224,.101,'Simulation: covariate-adjusted scenario only; lines show mean ± 95% CI across replicates.',
             fontsize=7.6,ha='left',va='center',color='#555E67')
    fig.text(.224,.085,'Real data: PPMI BH FDR < 0.05; independent PDBP nominal P < 0.05 (direction not assessed).',
             fontsize=7.4,ha='left',va='center',color='#555E67')
    fig.text(.224,.069,'Blue method names: regression/effect-estimating methods. Brown: combination/omnibus methods.',
             fontsize=7.3,ha='left',va='center',color='#555E67')
    save(fig,out/'main'/'Fig1_tQTL_real_replication_and_covariate_simulation')
    z.to_csv(out/'tables'/'Fig1_real_methods.csv',index=False)
    sim.to_csv(out/'tables'/'Fig1_covariate_simulation_metrics.csv',index=False)
    write_json(out/'tables'/'Fig1_manifest.json',{
        'real_source':a.real,'simulation_source':a.simulation,
        'real_panels':{'a':'replication_counts','b':'replication_fraction','c':'pi1','d':'spearman_logp'},
        'simulation_panels':{'e':'AUROC','f':'false_positive_rate','g':'empirical_FDR'},
        'simulation_condition':'confounded only (covariate scenario)',
        'real_methods':z.method.tolist(),'simulation_methods':methods,
        'excluded':'unadjusted linear regression, all __nocov, old submitted MPTGA, dynamic/additive subdivisions'})
    print('[FIG1 DONE] a–d real, e–g covariate simulation',flush=True)
if __name__=='__main__':main()
