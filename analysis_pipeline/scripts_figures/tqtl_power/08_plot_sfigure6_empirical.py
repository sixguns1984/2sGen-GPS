#!/usr/bin/env python3
"""Publication-ready vector supplementary Fig S6: full replication + real CTAR curve power."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('cairo')  # Use Cairo backend for high-quality output
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

RED='#B83B45';BLUE='#245A81';TEAL='#34989D';GREY='#66717E';PALETTE={'Lower':BLUE,'Median':TEAL,'Higher':RED}

def setup():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
       'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,
       'axes.linewidth':.75,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':600})

def tidy(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#E7EBEF',lw=.55,zorder=0);ax.tick_params(direction='out',length=3)
    ax.set_axisbelow(True)

def panel(ax,letter):ax.text(-.12,1.06,letter,transform=ax.transAxes,fontweight='bold',fontsize=11,ha='right',va='top')

def plot(out,figroot,maxn):
    powr=pd.read_csv(out/'power_grid_real_8parameter_effect_templates.csv')
    n80=pd.read_csv(out/'N80_grid_empirical_trajectory_maf.csv')
    templ=pd.read_csv(out/'three_real_CTAR_curve_templates.csv')
    full=pd.read_csv(out/'full_PPMI_PDBP_replication_summary.csv').iloc[0]
    import json
    manifest=json.loads((out/'power_manifest.json').read_text())
    nPP=int(manifest['n_PP_participants']);nPD=int(manifest['n_PD_participants'])
    fig,axs=plt.subplots(1,3,figsize=(11.8,3.45),gridspec_kw={'width_ratios':[.94,1.3,.98]})
    ax=axs[0];panel(ax,'a')
    # Full-cohort independent replication is a genuine all-pair statistic.
    # Do NOT connect it to a subsample curve from another set of pairs.
    full_rate=float(full['PDBP_nominal_replication_rate'])
    ax.scatter([100],[full_rate],marker='D',s=54,color=BLUE,zorder=5,label=f'All matched pairs (n={int(full["PDBP_matched_testable_pairs"]):,})')
    samp=figroot/'v4_subsampling'/'power_analysis_independent_v2'/'PDBP_replication_vs_sample_size.csv'
    if samp.is_file():
        z=pd.read_csv(samp).sort_values('fraction')
        ax.plot(z.fraction*100,z.mean_replication_rate,color=RED,marker='o',lw=1.7,markersize=4.5,label=f'Refitted subset (n={int(z.n_selected_pairs.iloc[0])})')
        s=z[z.fraction<1]
        ax.fill_between(s.fraction.to_numpy(float)*100,s.resampling_interval_low.to_numpy(float),
              s.resampling_interval_high.to_numpy(float),color=RED,alpha=.13,lw=0)
    else:
        ax.annotate('Subsampling not run',(.08,.1),xycoords='axes fraction',fontsize=7,color=GREY)
    ax.set_ylim(0,1.03);ax.set_xlim(43,106)
    ax.set_xticks([50,75,90,100]);ax.set_xlabel('PDBP participants (% of full cohort)')
    ax.set_ylabel('PDBP nominal replication fraction')
    ax.set_title('Independent replication vs. cohort size',pad=8)
    ax.legend(frameon=False,loc='lower right',fontsize=6.1)
    tidy(ax)

    ax=axs[1];panel(ax,'b')
    z=powr[(powr.threshold=='discovery')&(np.isclose(powr.maf,.05))&(powr.n_participants<=maxn)]
    for lab in ['Lower','Median','Higher']:
        s=z[z.template_label==lab].sort_values('n_participants')
        q=templ[templ.template_label==lab].iloc[0]
        val=float(q.signal_fraction)
        ax.plot(s.n_participants,s.power,color=PALETTE[lab],marker='o',markersize=3,
                lw=1.7,label=f'{lab} real tQTL (signal {val:.1%})')
    ax.axhline(.8,color=GREY,ls='--',lw=.85)
    ax.axvline(nPP,color='#D18A50',ls=':',lw=1.2)
    ax.axvline(nPD,color=BLUE,ls=(0,(4,2)),lw=1.0)
    ax.set_xlim(0,maxn);ax.set_ylim(0,1.03)
    ax.set_xlabel('Independent participants, N');ax.set_ylabel('Approximate 8-df discovery power')
    ax.set_title('Real CTAR effect profiles at MAF = 0.05',pad=8)
    ax.legend(frameon=False,loc='lower right',fontsize=6.2)
    tidy(ax)

    ax=axs[2];panel(ax,'c')
    z=n80[n80.threshold=='discovery'].copy()
    mafs=sorted(z.maf.unique());labels=['Lower','Median','Higher']
    mat=np.full((len(mafs),len(labels)),np.nan)
    for i,p in enumerate(mafs):
        for j,label in enumerate(labels):
            q=z[(np.isclose(z.maf,p))&(z.template_label==label)]
            if len(q)!=1:raise RuntimeError('N80 table has missing or duplicated MAF/trajectory')
            mat[i,j]=float(q.N80_first_tested.iloc[0]) if pd.notna(q.N80_first_tested.iloc[0]) else np.nan
    tmp=np.where(np.isfinite(mat),mat,max(maxn,5000)*1.25)
    # Rectangle patches, not imshow: retain fully editable vector PDF/SVG.
    cmap=plt.get_cmap('YlGnBu_r')
    minval=100.;maxval=max(maxn,5000)*1.25
    for i in range(len(mafs)):
        for j in range(len(labels)):
            c=cmap(float(np.clip((tmp[i,j]-minval)/(maxval-minval),0,1)))
            ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor=c,edgecolor='white',linewidth=1.0))
    ax.set_xlim(-.5,len(labels)-.5);ax.set_ylim(len(mafs)-.5,-.5)
    ax.set_xticks(range(3),['Lower','Median','Higher']);ax.set_yticks(range(len(mafs)),[f'{p:.2f}' for p in mafs])
    ax.set_xlabel('Real CTAR genetic-effect profile');ax.set_ylabel('Minor allele frequency')
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            text=f'{int(mat[i,j]):,}' if np.isfinite(mat[i,j]) else f'>{int(powr.n_participants.max()):,}'
            val=tmp[i,j];ax.text(j,i,text,ha='center',va='center',fontsize=8,
                                   color='white' if val>2000 else '#0B2440')
    ax.set_title('First tested N reaching 80% power',pad=8)
    ax.spines[['top','right','left','bottom']].set_visible(False)
    ax.tick_params(length=0)

    fig.subplots_adjust(left=.065,right=.995,bottom=.29,top=.86,wspace=.40)
    fig.text(.5,.135,f'Stage I participants: PPMI {nPP:,}; PDBP {nPD:,}.  '
         f'Full-cohort point uses all tested SNP–gene pairs; subsampling curve uses a separate selected subset.',
         ha='center',fontsize=7.1,color=GREY)
    fig.text(.5,.08,'b–c: fixed empirical PPMI BH raw-P cutoff, assumed residual correlation, real 8-coefficient trajectories '
         'and real covariate/visit design. N80 is a grid-based approximation, not an exact minimum.',
         ha='center',fontsize=6.9,color=GREY)
    dst=figroot/'supplementary';dst.mkdir(parents=True,exist_ok=True)
    name=dst/'Supplementary_Figure_6_CTAR_empirical_trajectory_power_v4'
    for ext in ['pdf','svg','png']:
        fig.savefig(name.with_suffix('.'+ext),dpi=600 if ext=='png' else None,bbox_inches='tight',pad_inches=.07, backend=(ext if ext in ("svg", "pdf") else None))
    plt.close(fig)
    print('[FIGURE]',str(name)+'.pdf',flush=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('--figroot',required=True);a.add_argument('--max-n',type=int,default=5000);args=a.parse_args()
    setup();out=Path(args.figroot)/'power_analysis_empirical_v4';plot(out,Path(args.figroot),args.max_n)
if __name__=='__main__':main()
