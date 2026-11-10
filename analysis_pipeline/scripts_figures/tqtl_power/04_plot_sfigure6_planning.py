#!/usr/bin/env python3
"""New SFig6: independent replication support and prospective CTAR sample-size planning."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import binom
import matplotlib
matplotlib.use('cairo')  # Use Cairo backend for high-quality output
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

RED='#B83B45'; BLUE='#245A81'; TEAL='#34989D'; PURPLE='#73568B'; GRAY='#747E88'; GRID='#E7EBEF'

def style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
      'font.size':8,'axes.titlesize':9.6,'axes.labelsize':8.4,'legend.fontsize':7.1,
      'xtick.labelsize':7.2,'ytick.labelsize':7.2,'axes.linewidth':.75,'pdf.fonttype':42,
      'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':600})

def tidy(ax):
    for sp in ['top','right']:ax.spines[sp].set_visible(False)
    ax.tick_params(length=3,direction='out')
    ax.grid(axis='y',color=GRID,lw=.6,zorder=0)
    ax.set_axisbelow(True)

def panel(ax,l):ax.text(-.075,1.105,l,transform=ax.transAxes,fontsize=11,fontweight='bold',va='top',ha='right')

def plot(out:Path,figdir:Path):
    summary_file=figdir/'power_analysis_independent_v2'/'PDBP_replication_vs_sample_size.csv'
    if not summary_file.is_file():
        raise FileNotFoundError('Independent PDBP subsampling summary not found: '+str(summary_file)+
                '\nRun v2 02_plot_sfigure6_independent.py once to summarize existing checkpoint results. '
                'No new CTAR refits are needed for this new figure.')
    s=pd.read_csv(summary_file).sort_values('fraction')
    req={'fraction','mean_replication_rate','n_total_PDBP','n_selected_pairs'}
    if not req.issubset(s.columns):raise RuntimeError('Missing v2 summary columns: '+str(req-set(s.columns)))
    z=pd.read_csv(out/'prospective_power_grid.csv')
    ns=pd.read_csv(out/'minimum_n_80pct_by_maf_effect.csv')
    cfg=json.loads((out/'analysis_manifest.json').read_text())
    if not np.isclose(s.fraction.max(),1):raise ValueError('Missing 100% independent replication in v2')
    if not {'PPMI_profile','PDBP_profile','empirical_PPMI_discovery_p_cutoff'}.issubset(cfg):raise ValueError('Invalid design manifest')
    NPP=int(cfg['PPMI_profile']['n_participants']); NPD=int(cfg['PDBP_profile']['n_participants'])
    a=float(cfg['empirical_PPMI_discovery_p_cutoff'])
    rho=float(cfg['rho_ref_assumed']);shape=cfg['design_profile']['shape']
    fig,axs=plt.subplots(2,2,figsize=(10.5,7.2))
    (aa,bb),(cc,dd)=axs
    part=s[s.fraction<1]
    x=part.fraction.to_numpy(float)*100
    aa.plot(x,part.mean_replication_rate,color=RED,marker='o',lw=1.85,label='PDBP subsets (without replacement)')
    if {'resampling_interval_low','resampling_interval_high'}.issubset(part.columns):
        aa.fill_between(x,part.resampling_interval_low.to_numpy(float),part.resampling_interval_high.to_numpy(float),
              color=RED,alpha=.13,edgecolor='none',label='2.5–97.5% resampling interval')
    f=s[s.fraction==1].iloc[0]
    aa.scatter([100],[f.mean_replication_rate],color=BLUE,s=45,marker='D',zorder=5,label='Full independent PDBP')
    if len(x):aa.plot([x[-1],100],[part.mean_replication_rate.iloc[-1],f.mean_replication_rate],color=GRAY,ls='--',lw=.9)
    aa.set_xticks(sorted(set((s.fraction*100).round().astype(int))))
    aa.set_xlim(45,105);aa.set_ylim(0,1.04)
    aa.set_title('Independent replication versus cohort size',fontweight='semibold')
    aa.set_xlabel('PDBP participants (% of full cohort)')
    aa.set_ylabel('Independent replication fraction')
    aa.legend(loc='lower right',frameon=False,fontsize=6.7)
    tidy(aa);panel(aa,'a')
    # Only observed Stage-I discovery threshold (not nominal replication) in prospective discovery curves
    ss=z[(z.threshold=='discovery')&np.isclose(z.maf,.05)]
    pal=[BLUE,TEAL,PURPLE,RED]
    for i,(eff,g) in enumerate(ss.groupby('effect_sd_per_allele',sort=True)):
        g=g.sort_values('n_participants')
        bb.plot(g.n_participants,g.estimated_power,color=pal[i%len(pal)],lw=1.8,marker='o',markersize=3,
                label=f'{float(eff):.2f} residual SD/allele')
    bb.axhline(.8,color='#777',ls=(0,(3,3)),lw=.9,label='80% power')
    bb.axvline(NPP,color='#D08046',ls=':',lw=1.0,label=f'PPMI N={NPP}')
    bb.axvline(NPD,color=GRAY,ls='-.',lw=1.0,label=f'PDBP N={NPD}')
    bb.set_xscale('log');bb.set_xlim(max(70,ss.n_participants.min()*.9),ss.n_participants.max()*1.08)
    bb.set_ylim(-.01,1.02);bb.set_xlabel('Independent participants (log scale)')
    bb.set_ylabel('Approximate discovery power')
    bb.set_title('MAF = 0.05: CTAR 8-df detection power',fontweight='semibold')
    bb.legend(frameon=False,loc='lower right',fontsize=6.1,ncol=2)
    tidy(bb);panel(bb,'b')
    # model-based N80 table
    m=ns[ns.threshold=='discovery']
    mafs=sorted(m.maf.unique()); effects=sorted(m.effect_sd_per_allele.unique())
    heat=np.full((len(mafs),len(effects)),np.nan)
    for (maf,eff),q in m.groupby(['maf','effect_sd_per_allele']):
        nval=q.N80_grid_first.iloc[0]
        if pd.notna(nval):heat[mafs.index(maf),effects.index(eff)]=float(nval)
    import matplotlib.colors as colors
    # missing cells are 'not attained up to grid maximum' and must not be displayed as 0
    masked=np.ma.masked_invalid(heat)
    cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#F0F0F0')
    im=cc.pcolormesh(np.arange(len(effects)+1)-.5,np.arange(len(mafs)+1)-.5,masked,
        cmap=cmap,vmin=0,vmax=max(1000,np.nanmax(heat)) if np.isfinite(heat).any() else 1000,
        edgecolors='white',linewidth=.9,shading='flat')
    cc.set_xlim(-.5,len(effects)-.5);cc.set_ylim(len(mafs)-.5,-.5)
    for i,maf in enumerate(mafs):
        for j,eff in enumerate(effects):
            v=heat[i,j]
            lab=f'{int(v):,}' if np.isfinite(v) else f'>{int(z.n_participants.max()):,}'
            cc.text(j,i,lab,ha='center',va='center',fontsize=8,color='white' if np.isfinite(v) and v>np.nanmax(heat)*.55 else '#1B2F40',fontweight='medium')
    cc.set_xticks(range(len(effects)),[f'{e:.2f}' for e in effects]);cc.set_yticks(range(len(mafs)),[f'{m:.2f}' for m in mafs]);
    cc.set_xlabel('True effect (residual SD per allele)')
    cc.set_ylabel('Minor allele frequency (MAF)')
    cc.set_title('First tested N reaching 80% discovery power',fontweight='semibold')
    # Heatmap cells are directly annotated with N; avoid rasterized gradient colorbars in PDF.
    cc.tick_params(length=0);panel(cc,'c')
    # sample coverage, distinguish 1 and >=5 minor homozygotes; not formal validity threshold
    N=np.arange(50,max(int(z.n_participants.max()),NPP,NPD)+1)
    dd.plot(N,binom.sf(0,N,.05**2),color=TEAL,lw=1.9,label='≥1 minor homozygote')
    dd.plot(N,binom.sf(4,N,.05**2),color=RED,lw=1.9,label='≥5 minor homozygotes')
    dd.axhline(.95,color=GRAY,ls=(0,(3,3)),lw=.9,label='95% representation chance')
    dd.axvline(NPP,color='#D08046',ls=':',lw=1.,label=f'PPMI N={NPP}')
    dd.axvline(NPD,color=GRAY,ls='-.',lw=1.,label=f'PDBP N={NPD}')
    dd.set_xlabel('Independent participants')
    dd.set_ylabel('Probability of ≥k minor homozygotes')
    dd.set_xlim(0,N.max());dd.set_ylim(0,1.02)
    dd.set_title('MAF = 0.05: rare genotype representation',fontweight='semibold')
    dd.legend(frameon=False,fontsize=6.6,loc='lower right')
    tidy(dd);panel(dd,'d')
    fig.subplots_adjust(left=.11,right=.96,top=.91,bottom=.16,hspace=.44,wspace=.37)
    foot=(f'a: {int(f.n_selected_pairs)} PPMI candidates, independent PDBP P < 0.05; '
          f'b,c: planning approximations using fixed ρ={rho:g}, {shape} effect and PPMI observed BH raw-P boundary {a:.2g}.')
    fig.text(.5,.084,foot,ha='center',fontsize=7,color='#525C68')
    fig.text(.5,.060,'b–d assume HWE, 8-df CTAR genotype coding and observed visit patterns; power varies with effect, MAF and design.',
             ha='center',fontsize=7,color='#525C68')
    fig.text(.5,.036,'Genotype representation in d is a separate planning diagnostic; theoretical power does not certify adequate sample size for all tQTLs.',
             ha='center',fontsize=6.8,color='#525C68')
    dst=figdir/'supplementary';dst.mkdir(exist_ok=True,parents=True)
    stem=dst/'Supplementary_Figure_6_CTAR_replication_and_prospective_power'
    for ext in ['pdf','svg','png']:
        fig.savefig(stem.with_suffix('.'+ext),bbox_inches='tight',pad_inches=.085,dpi=600 if ext=='png' else None, backend=(ext if ext in ("svg", "pdf") else None))
    plt.close(fig)
    print(f'[DONE] {stem}.pdf [vector]',flush=True)


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    a.add_argument('--figroot',default='')
    args=a.parse_args()
    root=Path(args.root);fig=Path(args.figroot) if args.figroot else root/'manuscript_figures'/'tqtl_benchmark_v4'
    style();plot(fig/'power_analysis_planning_v3',fig)
if __name__=='__main__':main()
