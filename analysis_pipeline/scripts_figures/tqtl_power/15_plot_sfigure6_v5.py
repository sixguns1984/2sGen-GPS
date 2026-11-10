#!/usr/bin/env python3
"""Three-panel Nature-style all-vector Supplementary Figure 6.

(a) all matched PPMI discoveries tested in PDBP (existing), vs distinct
    donor subsample reference set (never connect different denominators).
(b) normalized GLS partial genetic signal strengths at REAL median IV MAF.
(c) 80%-power N grid for varying MAF, standardized real trajectory shapes.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('cairo')  # Use Cairo backend for high-quality output
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch
from matplotlib.lines import Line2D

RED='#BA3C49';BLUE='#2A648A';TEAL='#2C9798';GOLD='#CD9143';PURPLE='#6C568A';GRAY='#667481'
R2COLORS={.005:BLUE,.01:TEAL,.02:GOLD,.05:RED}

def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
     'font.size':8,'axes.titlesize':10,'axes.labelsize':9,'axes.linewidth':.8,
     'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.1,
     'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':400})

def axis(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(direction='out',length=3)
    ax.grid(axis='y',color='#DFE6EB',alpha=.85,lw=.55)
    ax.set_axisbelow(True)

def panel(ax,txt):ax.text(-.11,1.08,txt,transform=ax.transAxes,ha='right',va='top',fontsize=12,fontweight='bold')

def full_plot(ax,figroot,full):
    n_all=int(full.PDBP_matched_testable_pairs)
    p=float(full.PDBP_nominal_replication_rate)
    ax.scatter([100],[p],s=53,marker='D',color=BLUE,zorder=10,
               label=f'All matched pairs (n={n_all:,})')
    ss=figroot/'v5_subsampling'/'power_analysis_independent_v2'/'PDBP_replication_vs_sample_size.csv'
    if ss.exists():
        x=pd.read_csv(ss).sort_values('fraction')
        if not np.isclose(x.fraction.max(),1):raise RuntimeError('Missing full-100% subset checkpoint')
        if x.n_selected_pairs.nunique()!=1:raise RuntimeError('Subset denominator changed')
        x=x[x.fraction<1]; nsmall=int(pd.read_csv(ss).n_selected_pairs.iloc[0])
        ax.plot(x.fraction*100,x.mean_replication_rate,marker='o',markersize=4.5,
                color=RED,lw=1.9,label=f'Random PPMI-discovered subset (n={nsmall})')
        ax.fill_between(x.fraction.to_numpy()*100,
            x.resampling_interval_low.to_numpy(),x.resampling_interval_high.to_numpy(),
            color=RED,alpha=.14,lw=0)
        x100=pd.read_csv(ss); r=float(x100.loc[x100.fraction==1,'mean_replication_rate'].iloc[0])
        ax.scatter([100],[r],color=RED,marker='o',s=27,zorder=8)
        ax.plot([float(x.fraction.iloc[-1])*100,100],
                [float(x.mean_replication_rate.iloc[-1]),r],color=RED,alpha=.55,lw=.85,ls='--')
    else:
        ax.text(.04,.11,'Optional donor subsampling not run',transform=ax.transAxes,
                color=GRAY,fontsize=7)
    ax.set_xlim(45,106);ax.set_ylim(0,1.04)
    ax.set_xticks([50,75,90,100])
    ax.set_xlabel('PDBP participants (% of full cohort)')
    ax.set_ylabel('Nominal PDBP replication fraction')
    ax.set_title('Observed independent replication',pad=8)
    ax.legend(loc='lower right',frameon=False,fontsize=6.4)
    axis(ax);panel(ax,'a')

def power_panel(ax,power,manifest,maf):
    data=power[(power.threshold=='discovery_fixed_BH')&np.isclose(power.maf,maf)]
    levels=sorted(data.target_R2_at_reference_MAF.unique())
    for i,r2 in enumerate(levels):
        x=data[np.isclose(data.target_R2_at_reference_MAF,r2)].sort_values('n_participants')
        c=R2COLORS.get(round(float(r2),4),plt.get_cmap('viridis')(i/max(len(levels)-1,1)))
        ax.plot(x.n_participants,x.median_power,color=c,lw=1.85,marker='o',markersize=3.25,
                label=f'{r2*100:g}%')
        ax.fill_between(x.n_participants.to_numpy(float),x.p10_power.to_numpy(float),
                        x.p90_power.to_numpy(float),color=c,alpha=.12,lw=0)
    ax.axhline(.8,color='#737B84',lw=.95,ls=(0,(3,3)))
    pp=int(manifest['participants_PP']);pdN=int(manifest['participants_PD'])
    # Close cohorts: a single shaded zone avoids overlap of two vertical labels
    ax.axvspan(min(pp,pdN),max(pp,pdN),facecolor='#BAC6D1',alpha=.25,linewidth=0)
    ax.axvline(pdN,color=BLUE,ls=(0,(4,3)),lw=.85)
    ax.axvline(pp,color=RED,ls=':',lw=.85)
    ax.set_xlim(0,max(power.n_participants.max(),5000))
    ax.set_ylim(0,1.04)
    ax.set_xlabel('Independent participants, N')
    ax.set_ylabel('Approximate 8-df discovery power')
    ax.set_title(f'Effect-size sensitivity at MAF = {maf:.2f}',pad=8)
    ax.legend(title='GLS genetic signal fraction',loc='lower right',frameon=False,ncol=2)
    axis(ax);panel(ax,'b')

def heatmap(ax,table,manifest):
    q=table[table.threshold=='discovery_fixed_BH'].copy()
    mafs=sorted(q.maf.unique());r2s=sorted(q.target_R2_at_reference_MAF.unique())
    maxn=max(int(q.max_tested_N.max()),5000)
    vals=np.full((len(mafs),len(r2s)),np.nan)
    for i,p in enumerate(mafs):
        for j,r2 in enumerate(r2s):
            one=q[np.isclose(q.maf,p)&np.isclose(q.target_R2_at_reference_MAF,r2)]
            if len(one)!=1:raise RuntimeError('Missing or duplicated N80 cell')
            vals[i,j]=one.N80_median_first_grid.iloc[0]
    cmap=plt.get_cmap('YlGnBu_r')
    for i,p in enumerate(mafs):
        for j,r2 in enumerate(r2s):
            val=vals[i,j]
            score=(maxn-val)/maxn if np.isfinite(val) else 0.0
            color=cmap(.11+.77*score)
            ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=color,edgecolor='white',lw=1.4))
            txt=f'{int(val):,}' if np.isfinite(val) else f'>{maxn:,}'
            ax.text(j,i,txt,ha='center',va='center',fontsize=8,
                    color='white' if score<.54 else '#152638',fontweight='semibold')
    ax.set_xlim(-.5,len(r2s)-.5);ax.set_ylim(len(mafs)-.5,-.5)
    ax.set_yticks(range(len(mafs)),[f'{x:.2f}' for x in mafs])
    ax.set_xticks(range(len(r2s)),[f'{v*100:g}%' for v in r2s])
    ax.set_xlabel('GLS genetic signal fraction at reference MAF')
    ax.set_ylabel('Assumed MAF')
    ax.set_title('First tested N reaching 80% power',pad=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():s.set_visible(False)
    panel(ax,'c')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--figroot',required=True)
    parser.add_argument('--main-maf',default='auto',help='auto uses reference MAF; other value must exist in MAF grid')
    a=parser.parse_args();style()
    figroot=Path(a.figroot);folder=figroot/'power_analysis_CTAR_v5'
    power=pd.read_csv(folder/'power_standardized_PVE_grid_v5.csv')
    n80=pd.read_csv(folder/'N80_standardized_PVE_v5.csv')
    full=pd.read_csv(folder/'full_PPMI_PDBP_replication_summary.csv').iloc[0]
    manifest=json.loads((folder/'power_manifest_v5.json').read_text())
    refmaf=float(manifest['reference_maf']);scenarios=sorted(power.maf.unique())
    mainmaf=min(scenarios,key=lambda p:abs(float(p)-refmaf)) if a.main_maf=='auto' else float(a.main_maf)
    if not any(np.isclose(scenarios,mainmaf)):raise ValueError('Requested main MAF not in MAF scenarios')
    f,axs=plt.subplots(1,3,figsize=(12.7,3.65),gridspec_kw={'width_ratios':[1.07,1.30,1.12]})
    full_plot(axs[0],figroot,full)
    power_panel(axs[1],power,manifest,mainmaf)
    heatmap(axs[2],n80,manifest)
    f.subplots_adjust(left=.06,right=.99,top=.87,bottom=.32,wspace=.36)
    nPP=manifest['participants_PP'];nPD=manifest['participants_PD'];nsh=manifest['n_accepted_shapes']
    f.text(.5,.16,f'Stage I: PPMI {nPP:,} and PDBP {nPD:,} participants. '
          f'Planning uses {nsh} normalized real CTAR trajectory shapes and reference MAF = {refmaf:.3f}.',
          ha='center',fontsize=7,color=GRAY)
    f.text(.5,.11,'a: all-pair independent replication vs separate candidate subset. '
          'b: shaded bands = 10th–90th percentile over real effect shapes, not confidence intervals.',
          ha='center',fontsize=6.9,color=GRAY)
    f.text(.5,.06,'b–c: fixed observed PPMI BH raw-P boundary and fixed-rho 8-df GLS approximation; '
          'N80 is the first grid point, not a validated minimum.',ha='center',fontsize=6.9,color=GRAY)
    out=figroot/'supplementary';out.mkdir(parents=True,exist_ok=True)
    stem=out/'Supplementary_Figure_6_CTAR_power_standardized_PVE_v5'
    for suf in ['pdf','svg','png']:
        f.savefig(str(stem)+'.'+suf,dpi=600 if suf=='png' else None,bbox_inches='tight',pad_inches=.07, backend=(suf if suf in ("svg", "pdf") else None))
    plt.close(f)
    print('[FIGURE]',str(stem)+'.pdf',flush=True)
    # Dedicated MAF=.05 sensitivity to avoid hiding low-MAF limitations.
    if any(np.isclose(scenarios,.05)) and not np.isclose(mainmaf,.05):
        fig2,ax=plt.subplots(figsize=(5.1,3.75))
        power_panel(ax,power,manifest,.05)
        fig2.subplots_adjust(left=.16,right=.96,top=.85,bottom=.23)
        ss=out/'Supplementary_Figure_6_MAF005_sensitivity_v5'
        for ext in ['pdf','svg','png']:
            fig2.savefig(str(ss)+'.'+ext,dpi=600 if ext=='png' else None,bbox_inches='tight',pad_inches=.07, backend=(ext if ext in ("svg", "pdf") else None))
        plt.close(fig2)
    # export audit
    pd.DataFrame([{'main_maf':mainmaf,'reference_maf':refmaf,
      'n_PP':nPP,'n_PD':nPD,'n_trajectory_shapes':nsh,
      'full_PDBP_matched_n':full.PDBP_matched_testable_pairs,
      'full_nominal_rate':full.PDBP_nominal_replication_rate}]).to_csv(folder/'figure_6_readout_v5.csv',index=False)

if __name__=='__main__':main()
