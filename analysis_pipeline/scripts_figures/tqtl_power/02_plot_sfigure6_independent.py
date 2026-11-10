#!/usr/bin/env python3
"""Supplementary Figure 6 (new): independent PDBP sample-size sensitivity."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('cairo')  # Use Cairo backend for high-quality output
import matplotlib.pyplot as plt

BLUE='#245A81'; RED='#B83B45'; TEAL='#34989D'; GRID='#E7EBEF'


def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
        'font.size':8.3,'axes.labelsize':9,'axes.titlesize':10,'xtick.labelsize':8,
        'ytick.labelsize':8,'axes.linewidth':.8,'pdf.fonttype':42,
        'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':600})


def tidy(ax):
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    ax.grid(axis='y',color=GRID,lw=.7,zorder=0)
    ax.tick_params(direction='out',length=3)
    ax.set_axisbelow(True)


def panel(ax,lett):
    ax.text(-.12,1.065,lett,transform=ax.transAxes,ha='right',va='top',fontsize=12,fontweight='bold')


def collect(part_dir:Path,manifest:dict,selected:pd.DataFrame):
    frames=[]
    set_pairs=set(selected.pair_id)
    for frac in manifest['fractions']:
        repids=[-1] if frac==1 else range(int(manifest['resamples']))
        for r in repids:
            path=part_dir/f'PD_fraction_{frac:.2f}_rep_{r:04d}.parquet'
            if not path.is_file():raise RuntimeError(f'Checkpoint incomplete: {path}')
            z=pd.read_parquet(path)
            if len(z)!=len(selected) or not z.pair_id.is_unique or set(z.pair_id)!=set_pairs:
                raise RuntimeError(f'CTAR refit SNP-gene pairs inconsistent in {path}')
            if (z['n_unique_participants'].nunique()!=1 or z['n_full_cohort'].nunique()!=1 or
                abs(float(z['fraction'].iloc[0])-float(frac))>1e-6):
                raise RuntimeError(f'Sample size metadata inconsistent in {path}')
            if frac==1 and (not z.is_full_dataset.all() or int(z.n_unique_participants.iloc[0])!=int(z.n_full_cohort.iloc[0])):
                raise RuntimeError('100% record is not real full PDBP cohort!')
            if frac<1 and z.is_full_dataset.any():raise RuntimeError('Partial subsample incorrectly labeled full')
            frames.append(z[['pair_id','fraction','replicate_id','detected','Pvalue',
                            'n_unique_participants','n_full_cohort','is_full_dataset']])
    return pd.concat(frames,ignore_index=True)


def summarize(z:pd.DataFrame,selected:pd.DataFrame):
    rep=z.groupby(['fraction','replicate_id'],as_index=False).agg(
        replication_fraction=('detected','mean'), n_detected=('detected','sum'),
        n_unique_participants=('n_unique_participants','first'),
        n_full_cohort=('n_full_cohort','first'))
    meanrows=[]
    for f,grp in rep.groupby('fraction',sort=True):
        values=grp.replication_fraction.to_numpy(float)
        full=(float(f)==1.)
        if full and len(values)!=1:raise ValueError('Full PDBP should be tested once')
        meanrows.append({'fraction':float(f), 'n_unique_participants':int(grp.n_unique_participants.iloc[0]),
            'n_total_PDBP':int(grp.n_full_cohort.iloc[0]),'n_selected_pairs':len(selected),
            'mean_replication_rate':float(values.mean()),
            'resampling_interval_low':float(np.quantile(values,.025)) if not full else np.nan,
            'resampling_interval_high':float(np.quantile(values,.975)) if not full else np.nan,
            'n_subsamples':len(values), 'full_cohort_not_resampled':bool(full)})
    summary=pd.DataFrame(meanrows).sort_values('fraction')
    grp=z.groupby(['fraction','pair_id'],as_index=False).agg(n_detected=('detected','sum'),
                                                              n_subsamples=('detected','size'),
                                                              mean_detection_fraction=('detected','mean'))
    grp=grp.merge(selected[['pair_id','Variant_ID','Ensembl_ID','Pvalue','FDR','rho',
                            'signal_bin','rho_bin','neglogp_ppmi']],on='pair_id',how='left',validate='many_to_one')
    grp['nominal_replication_at_full_PDBP']=grp['pair_id'].map(
        z[z.fraction==1].set_index('pair_id')['detected'].astype(int).to_dict()).astype(int)
    strata=[]
    for strat in ['rho_bin','signal_bin']:
        tmp=z.merge(selected[['pair_id',strat]],on='pair_id',validate='many_to_one')
        vals=tmp.groupby(['fraction','replicate_id',strat],as_index=False).detected.mean()
        for (f,g),t in vals.groupby(['fraction',strat]):
            x=t.detected.to_numpy(float)
            strata.append({'stratification':strat,'group':int(g),'fraction':float(f),
                'mean_replication_rate':float(x.mean()),
                'resampling_interval_low':float(np.quantile(x,.025)) if f<1 else np.nan,
                'resampling_interval_high':float(np.quantile(x,.975)) if f<1 else np.nan,
                'n_subsamples':len(x),
                'n_pairs':int((selected[strat]==g).sum())})
    return rep,summary,grp,pd.DataFrame(strata)


def overall(ax,summary):
    ss=summary[summary.fraction<1].sort_values('fraction')
    full=summary[summary.fraction==1].iloc[0]
    x=ss.fraction.to_numpy(float)*100
    y=ss.mean_replication_rate.to_numpy(float)
    ax.plot(x,y,color=RED,marker='o',lw=1.9,markersize=5,label='Without-replacement subsets')
    ax.fill_between(x,ss.resampling_interval_low.to_numpy(float),
        ss.resampling_interval_high.to_numpy(float),color=RED,alpha=.13,linewidth=0,
        label='95% resampling interval')
    ax.scatter([100],[full.mean_replication_rate],color=BLUE,marker='D',s=43,zorder=5,label='Full PDBP cohort')
    ax.plot([x[-1],100],[y[-1],full.mean_replication_rate],color='#A7AEB7',lw=1.0,linestyle='--')
    ax.legend(frameon=False,fontsize=7,loc='best')
    ax.set_title('Independent PDBP replication',fontweight='semibold',pad=8)
    ax.set_ylabel('Replicated PPMI tQTL pairs (fraction)')
    tidy(ax);panel(ax,'a')


def stratified(ax,strata,column, title, labels,lett):
    colors=[BLUE,TEAL,RED]
    xall=[]
    z=strata[strata.stratification==column]
    for i,g in enumerate(sorted(z.group.unique())):
        s=z[z.group==g].sort_values('fraction')
        x=s.fraction.to_numpy(float)*100;y=s.mean_replication_rate.to_numpy(float)
        xall.extend(x)
        ax.plot(x,y,color=colors[i%3],marker=['s','o','^'][i%3],lw=1.8,
                markersize=4.6,label=labels[i] if i<len(labels) else f'Group {g}')
    ax.set_title(title,fontweight='semibold',pad=8)
    ax.legend(frameon=False,fontsize=7,loc='best')
    tidy(ax);panel(ax,lett)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    p.add_argument('--out-root',default='')
    a=p.parse_args();style()
    root=Path(a.root)
    out=Path(a.out_root) if a.out_root else root/'manuscript_figures'/'tqtl_benchmark_v4'
    analysis=out/'power_analysis_independent_v2'
    manifest_file=analysis/'analysis_manifest.json';selected_file=analysis/'PPMI_selected_pairs_NO_PDBP_PREFILTER.csv'
    if not manifest_file.exists():raise FileNotFoundError(manifest_file)
    manifest=json.loads(manifest_file.read_text())
    chosen=pd.read_csv(selected_file,dtype={'Variant_ID':str,'Ensembl_ID':str,'pair_id':str})
    z=collect(analysis/'checkpoints',manifest,chosen)
    replicate,summary,pair,strata=summarize(z,chosen)
    analysis.mkdir(parents=True,exist_ok=True)
    replicate.to_csv(analysis/'PDBP_subsample_replicate_rates.csv',index=False)
    summary.to_csv(analysis/'PDBP_replication_vs_sample_size.csv',index=False)
    pair.to_csv(analysis/'PDBP_pairwise_replication_probabilities.csv',index=False)
    strata.to_csv(analysis/'PDBP_replication_by_PPMI_signal_and_rho.csv',index=False)
    fig,axs=plt.subplots(1,3,figsize=(10.4,3.45),sharey=True)
    overall(axs[0],summary)
    stratified(axs[1],strata,'signal_bin','By PPMI discovery significance',
               ['Lower significance','Intermediate','Higher significance'],'b')
    stratified(axs[2],strata,'rho_bin','By PPMI residual autocorrelation',
               ['Lower residual AR','Intermediate','Higher residual AR'],'c')
    ticks=[round(float(x)*100) for x in manifest['fractions']]
    for ax in axs:
        ax.set_xlim(min(ticks)-7,106)
        ax.set_xticks(ticks)
        ax.set_ylim(0,1.03)
        ax.set_xlabel('PDBP participants (% of full cohort)')
    fig.subplots_adjust(left=.08,right=.98,top=.86,bottom=.30,wspace=.20)
    fig.text(.5,.13,'Fixed PPMI BH-significant CTAR-tQTL set; PDBP nominal P < 0.05. '
             '50–90%: donor-level sampling without replacement; 100%: original PDBP cohort.',
             ha='center',fontsize=7.2,color='#5B626B')
    fig.text(.5,.075,'Sample-size sensitivity of independent replication; not prospective power. '
             'Shaded area: 2.5th–97.5th percentile across resamples.',
             ha='center',fontsize=7.2,color='#5B626B')
    dst=out/'supplementary';dst.mkdir(parents=True,exist_ok=True)
    stem=dst/'Supplementary_Figure_6_CTAR_independent_replication_sample_size'
    for ext in ('pdf','svg','png'):
        fig.savefig(stem.with_suffix('.'+ext),dpi=600 if ext=='png' else None,
                    bbox_inches='tight',pad_inches=.08, backend=(ext if ext in ("svg", "pdf") else None))
    plt.close(fig)
    print(f'[FIG COMPLETE] {stem}.pdf',flush=True)
    print('[SUMMARY]\n'+summary.to_string(index=False),flush=True)

if __name__=='__main__':main()
