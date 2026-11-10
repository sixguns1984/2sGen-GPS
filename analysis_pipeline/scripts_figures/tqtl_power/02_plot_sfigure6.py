#!/usr/bin/env python3
"""Nature-style Supplementary Fig. 6: simulation power + selected real-tQTL bootstrap recovery."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('cairo')  # Use Cairo backend for high-quality output
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

METHODS=[
 ('submitted_followup_ctar_mptga','CTAR-MPTGA','#B83B45','o'),
 ('lin2018_regression_cov','Linear regression + covariates','#245A81','s'),
 ('lin2018_mptga','Lin2018-MPTGA','#34989D','D'),
 ('lin2018_ar','AR(1) regression','#73568B','^'),
 ('union','Union','#AD813E','v'),
 ('fisher','Fisher','#626C78','P'),
 ('manova','MANOVA','#3F8565','h')]
COLOR_BIN={0:'#487DA8',1:'#369A93',2:'#C75A58'}


def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
        'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
        'axes.linewidth':.75,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
        'xtick.direction':'out','ytick.direction':'out','savefig.dpi':600})


def tidy(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#E6EAF0',lw=.6,zorder=0)


def letter(ax,s):
    ax.text(-.16,1.08,s,transform=ax.transAxes,fontweight='bold',fontsize=11,ha='right',va='top')


def plot_sim(ax,sim):
    if not {'method','scenario','power_FDR05_mean'}.issubset(sim):
        raise ValueError('Simulation summary lacks required method, scenario, power_FDR05_mean')
    if 'confounding_status' in sim.columns:
        z=sim[sim.confounding_status.astype(str)=='confounded'].copy()
    else:
        z=sim[~sim.scenario.astype(str).str.contains(r'__unconfounded$')].copy()
    if 'rho_true_mean' in z:
        z['rho_plot']=pd.to_numeric(z['rho_true_mean'],errors='coerce')
    elif 'rho_true' in z:
        z['rho_plot']=pd.to_numeric(z['rho_true'],errors='coerce')
    else:
        z['rho_plot']=z.scenario.astype(str).str.extract(r'rho_([m\d]+p\d+)')[0].map(
            lambda x:float(str(x).replace('m','-').replace('p','.')) if pd.notna(x) else np.nan)
    z=z[~z.method.astype(str).str.endswith('__nocov')]
    z=z[z.method.astype(str)!='lin2018_regression']
    if z.rho_plot.isna().any():raise ValueError('Cannot parse rho from simulation summary')
    for m,label,color,marker in METHODS:
        group=z[z.method==m].sort_values('rho_plot')
        if group.empty: continue
        x=group.rho_plot.to_numpy(float); y=pd.to_numeric(group.power_FDR05_mean,errors='coerce').to_numpy(float)
        lowcol='power_FDR05_ci95_low'; highcol='power_FDR05_ci95_high'
        if lowcol in group and highcol in group:
            lo=pd.to_numeric(group[lowcol],errors='coerce').to_numpy(float)
            hi=pd.to_numeric(group[highcol],errors='coerce').to_numpy(float)
            ax.fill_between(x,lo,hi,color=color,alpha=.07,lw=0)
        ax.plot(x,y,marker=marker,markersize=3.6,color=color,lw=2.2 if m==METHODS[0][0] else 1.3,label=label)
    ax.set_ylim(0,1.02)
    ax.set_xlabel(r'Residual autocorrelation ($\rho$)')
    ax.set_ylabel('Detection power (simulation)')
    ax.set_title('Simulated tQTL detection power',pad=9,fontweight='semibold')
    tidy(ax);letter(ax,'a')


def collect(checkpoint_dir,manifest,chosen):
    frames=[]
    for cohort in manifest['cohorts']:
        for frac in manifest['fractions']:
            for rep in range(manifest['bootstraps']):
                p=checkpoint_dir/f'{cohort}_fraction_{float(frac):.2f}_rep_{rep:04d}.parquet'
                if not p.exists():
                    raise RuntimeError(f'Missing {p}; refits incomplete. Do not plot partial results.')
                z=pd.read_parquet(p,columns=['Variant_ID','Ensembl_ID','Pvalue','cohort','fraction','bootstrap_id','detected'])
                if len(z)!=len(chosen):raise RuntimeError(f'{p} expected {len(chosen)} pairs, found {len(z)}')
                z['pair_id']=z.Variant_ID.astype(str)+'||'+z.Ensembl_ID.astype(str)
                if set(z.pair_id)!=set(chosen.pair_id):raise RuntimeError(f'{p} has a mismatched SNP–gene set')
                frames.append(z[['pair_id','cohort','fraction','bootstrap_id','detected','Pvalue']])
    return pd.concat(frames,ignore_index=True)


def summarize(df,chosen):
    pair=df.groupby(['cohort','fraction','pair_id'],as_index=False).agg(
        recovery=('detected','mean'),n_bootstraps=('detected','size'))
    pair=pair.merge(chosen[['pair_id','rho','rho_bin','Pvalue','PD_Pvalue']],on='pair_id',validate='many_to_one')
    pair['rho_bin']=pd.to_numeric(pair['rho_bin']).astype(int)
    # mean fraction of selected pairs detected in each repeated sample
    samples=df.groupby(['cohort','fraction','bootstrap_id'],as_index=False).detected.mean().rename(columns={'detected':'mean_recovery'})
    rows=[]
    for (cohort,frac),group in samples.groupby(['cohort','fraction']):
        vals=group.mean_recovery.to_numpy(float)
        rows.append({'cohort':cohort,'fraction':float(frac),'mean_recovery':float(np.mean(vals)),
                     'ci_low':float(np.quantile(vals,.025)),'ci_high':float(np.quantile(vals,.975)),
                     'n_bootstraps':len(vals),'n_selected_pairs':chosen.shape[0]})
    summary=pd.DataFrame(rows)
    high=(pair.assign(recovered80=pair.recovery>=.80)
          .groupby(['cohort','fraction'],as_index=False).recovered80.mean()
          .rename(columns={'recovered80':'fraction_pairs_recovery_ge80'}))
    return pair,summary,high


def plot_cohort(ax,pair,summary,cohort,panel):
    p=pair[pair.cohort==cohort]
    ss=summary[summary.cohort==cohort].sort_values('fraction')
    if p.empty:raise ValueError(f'No bootstrap data for cohort {cohort}')
    for bin_id,label in [(0,'Low residual AR'),(1,'Intermediate AR'),(2,'High residual AR')]:
        sub=(p[p.rho_bin==bin_id].groupby('fraction',as_index=False).recovery.mean().sort_values('fraction'))
        if not sub.empty:
            ax.plot(sub.fraction*100,sub.recovery,marker='o',markersize=3.2,
                    lw=1.15,color=COLOR_BIN[bin_id],alpha=.84,label=label)
    ax.plot(ss.fraction.to_numpy(float)*100,ss.mean_recovery,marker='s',lw=2.35,
            markersize=4.0,color='#202E3B',zorder=5,label='Overall')
    ax.fill_between(ss.fraction.to_numpy(float)*100,ss.ci_low.to_numpy(float),
                    ss.ci_high.to_numpy(float),color='#202E3B',alpha=.12,lw=0)
    ax.set_ylim(0,1.02);ax.set_xlim(42,105)
    ax.set_xticks(sorted(ss.fraction.unique()*100))
    ax.set_xlabel('Resampled participants (% of original)')
    ax.set_ylabel('Fraction of selected pairs detected')
    ax.set_title(('PPMI' if cohort=='PP' else 'PDBP')+' conditional recovery',fontweight='semibold',pad=9)
    tidy(ax);letter(ax,panel)


def plot_high(ax,high):
    for cohort,color,label,marker in [('PP','#B83B45','PPMI','o'),('PD','#245A81','PDBP','s')]:
        z=high[high.cohort==cohort].sort_values('fraction')
        if not z.empty:
            ax.plot(z.fraction*100,z.fraction_pairs_recovery_ge80,marker=marker,color=color,lw=1.9,markersize=4,label=label)
    ax.axhline(.8,ls=(0,(3,3)),color='#9DA6B0',lw=.8)
    ax.set_xlim(42,105);ax.set_ylim(0,1.02)
    ax.set_xlabel('Resampled participants (% of original)')
    ax.set_ylabel('Pairs with recovery ≥80%')
    ax.set_title('High-recovery tQTLs',pad=9,fontweight='semibold')
    ax.legend(frameon=False,loc='best',fontsize=7)
    tidy(ax);letter(ax,'d')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    p.add_argument('--out-root',default='')
    p.add_argument('--sim',default='')
    a=p.parse_args();style()
    root=Path(a.root)
    out=Path(a.out_root) if a.out_root else root/'manuscript_figures'/'tqtl_benchmark_v4'
    analysis=out/'power_analysis'
    manifest_path=analysis/'analysis_manifest.json';chosen_path=analysis/'selected_replicated_CTAR_pairs.csv'
    if not manifest_path.exists():raise FileNotFoundError(manifest_path)
    manifest=json.loads(manifest_path.read_text())
    chosen=pd.read_csv(chosen_path,dtype={'Variant_ID':str,'Ensembl_ID':str,'pair_id':str})
    data=collect(analysis/'checkpoints',manifest,chosen)
    pair,summary,high=summarize(data,chosen)
    analysis.mkdir(parents=True,exist_ok=True)
    pair.to_csv(analysis/'CTAR_pairwise_subsample_recovery.csv',index=False)
    summary.to_csv(analysis/'CTAR_bootstrap_recovery_summary.csv',index=False)
    high.to_csv(analysis/'CTAR_fraction_pairs_recovery_ge80.csv',index=False)
    simpath=Path(a.sim) if a.sim else root/'simulation_qtl_benchmark_rho_covariate'/'simulation_summary.csv'
    if not simpath.exists():raise FileNotFoundError(simpath)
    sim=pd.read_csv(simpath,low_memory=False)
    fig=plt.figure(figsize=(9.8,7.3))
    gs=GridSpec(2,2,figure=fig,left=.11,right=.98,top=.91,bottom=.18,hspace=.45,wspace=.36)
    axA=fig.add_subplot(gs[0,0]);axB=fig.add_subplot(gs[0,1]);axC=fig.add_subplot(gs[1,0]);axD=fig.add_subplot(gs[1,1])
    plot_sim(axA,sim)
    plot_cohort(axB,pair,summary,'PP','b')
    plot_cohort(axC,pair,summary,'PD','c')
    plot_high(axD,high)
    handles=[Line2D([0],[0],color=c,marker=m,lw=1.4,markersize=4,label=l) for _,l,c,m in METHODS if _ in set(sim.method)]
    fig.legend(handles=handles,frameon=False,loc='lower center',ncol=4,bbox_to_anchor=(.50,.065),fontsize=7.3,columnspacing=1.6)
    fig.text(.5,.014,'Panel a: simulation power with true causal labels. Panels b–d: subject-bootstrap recovery conditional on PPMI discovery and PDBP replication; not an unbiased power estimate.',
             ha='center',va='center',fontsize=7,color='#515D69')
    outdir=out/'supplementary';outdir.mkdir(parents=True,exist_ok=True)
    stem=outdir/'Supplementary_Figure_6_CTAR_power_and_sample_size'
    for suffix in ['pdf','svg','png']:
        fig.savefig(stem.with_suffix('.'+suffix),dpi=600 if suffix=='png' else None,bbox_inches='tight',pad_inches=.1, backend=(suffix if suffix in ("svg", "pdf") else None))
    plt.close(fig)
    print('[FIG DONE]',stem,flush=True)
    print('[SUM]',summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
