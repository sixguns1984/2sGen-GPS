#!/usr/bin/env python3
"""Publication-quality vector Supplementary Fig. S6; no invented user results."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('cairo')  # Use Cairo backend for high-quality output
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

COL={'PPMI':'#AF3653','PDBP':'#255E83','100%':'#234F74','75%':'#299A99','50%':'#D98B3C'}
PHASES={1.0:('Observed fitted coefficients',COL['100%']),.75:('0.75× coefficient sensitivity',COL['75%']),.5:('0.50× coefficient sensitivity',COL['50%'])}


def config():
    mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':8.4,'axes.titlesize':10,
      'axes.labelsize':8.8,'xtick.labelsize':8,'ytick.labelsize':8,
      'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
      'axes.linewidth':.75,'savefig.dpi':400,'legend.frameon':False})


def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',c='#E2E8EA',linewidth=.6,alpha=.95)
    ax.set_axisbelow(True)


def panel(ax,s):
    ax.text(-.12,1.05,s,transform=ax.transAxes,fontsize=15,fontweight='bold',ha='right',va='bottom')


def replication(ax,figroot,summary):
    df=pd.read_csv(summary)
    allrow=df.iloc[0]
    rate=float(allrow['PDBP_nominal_replication_rate'])
    ax.scatter([100],[rate],s=65,marker='D',c=COL['PDBP'],zorder=6,
      label=f'All matched tests ({int(allrow["PDBP_matched_testable_pairs"]):,})')
    names=[figroot/'v5_subsampling/power_analysis_independent_v2/PDBP_replication_vs_sample_size.csv',
           figroot/'v4_subsampling/power_analysis_independent_v2/PDBP_replication_vs_sample_size.csv',
           figroot/'power_analysis_independent_v2/PDBP_replication_vs_sample_size.csv']
    src=next((s for s in names if s.exists()),None)
    if src:
        z=pd.read_csv(src)
        # Fully provenance-aware: reject unknown layout instead of silently depicting a fabricated trend.
        fraction=next((c for c in ('fraction','fraction_full','sample_fraction','subsample_fraction') if c in z),None)
        mean=next((c for c in ('replication_rate_mean','mean','overall_mean','mean_detection_rate') if c in z),None)
        if fraction and mean:
            zz=z.copy()
            if 'rho_bin' in zz:zz=zz[zz.rho_bin.isna()] if zz.rho_bin.isna().any() else zz
            xx=pd.to_numeric(zz[fraction],errors='coerce').to_numpy(float)
            yy=pd.to_numeric(zz[mean],errors='coerce').to_numpy(float)
            keep=np.isfinite(xx)&np.isfinite(yy)
            xx=xx[keep];yy=yy[keep]
            if np.nanmax(xx)<=1.01:xx*=100
            ii=np.argsort(xx)
            ax.plot(xx[ii],yy[ii],color=COL['PPMI'],marker='o',markersize=4.4,lw=1.7,
              label='Separate PPMI candidate subset')
        else:
            ax.text(.03,.07,'Subset data available, but schema not recognized',transform=ax.transAxes,fontsize=7,color='#888')
    ax.set(xlim=(42,105),ylim=(0,1.02),xlabel='PDBP participants (% of full cohort)',ylabel='Nominal PDBP replication fraction',title='Observed independent replication')
    ax.legend(loc='lower left',fontsize=7)
    clean(ax);panel(ax,'a')


def observed_pve(ax,calib):
    z=calib.copy()
    z['pve']=pd.to_numeric(z['observed_PVE_original_VarGenetic_over_VarRNA'],errors='coerce')
    z['MAF']=pd.to_numeric(z['MAF_observed'],errors='coerce')
    z=z[z.pve.gt(0)&z.MAF.gt(0)&z.MAF.le(.5)].copy()
    if z.empty: raise RuntimeError('No actual Var(genetic trajectory)/Var(RNA) values')
    cap=max(.03,min(.5,float(z.pve.quantile(.95))))
    y=np.minimum(z.pve.to_numpy(float),cap)*100
    ax.scatter(z.MAF,y,c=COL['PDBP'],s=12,alpha=.45,linewidths=0)
    ax.axvline(z.MAF.median(),color='#C06E42',lw=1,ls='--')
    ax.text(.95,.95,f'Median MAF = {z.MAF.median():.3f}\nn={len(z)} frozen pairs',
        transform=ax.transAxes,va='top',ha='right',fontsize=7.5,color='#2E414A')
    if np.any(z.pve>cap):
        ax.text(.04,.96,f'{int(np.sum(z.pve>cap))} values above axis cap',transform=ax.transAxes,
                fontsize=7,va='top',color='#697985')
    ax.set(xlim=(0,.51),ylim=(0,cap*100*1.04),xlabel='Observed PPMI SNP minor allele frequency (MAF)',
           ylabel='Observed PVE: Var[genetic curve] / Var[RNA] (%)',
           title='Real-data full-trajectory effect size')
    clean(ax);panel(ax,'b')


def n_curve(ax,summary,manifest,actual=None):
    for fac,(lab,c) in PHASES.items():
        z=summary[np.isclose(summary.effect_multiplier,fac)].sort_values('n_participants')
        if z.empty:continue
        ax.plot(z.n_participants,z.pct_profiles_80_joint,marker='o',ms=3.6,lw=1.6,c=c,label=lab)
    ax.axhline(.8,color='#7B858C',lw=.8,ls=(0,(4,3)))
    pp=manifest['PPMI_actual_N'];pdn=manifest['PDBP_actual_N']
    ax.axvspan(min(pp,pdn),max(pp,pdn),color='#AFB9C1',alpha=.18,lw=0)
    ax.axvline(pdn,color=COL['PDBP'],lw=1,ls=':')
    ax.set(xlim=(0,max(summary.n_participants)*1.015),ylim=(0,1.04),
        xlabel='Independent participants per cohort, N',
        ylabel='Fraction of sampled pairs with joint power ≥80%',
        title='Conditional power across cohort sizes')
    if actual is not None and not actual.empty:
        cur=actual[np.isclose(actual.effect_multiplier,1.)]
        if not cur.empty:
            v=float(cur.fraction_joint_power80.iloc[0]);n=int(cur.n_profiles.iloc[0])
            ax.text(.98,.04,f'Actual {pp:,}/{pdn:,} participants:\n{100*v:.1f}% of {n} pairs achieve ≥80%\nconditional joint power',
                transform=ax.transAxes,ha='right',va='bottom',fontsize=7,
                bbox={'facecolor':'white','alpha':.9,'edgecolor':'none','pad':3.})
    ax.legend(loc='upper left',fontsize=7)
    clean(ax);panel(ax,'c')


def n80_maf(ax,n80,manifest):
    z=n80[np.isclose(n80.effect_multiplier,1.0)].copy()
    if z.empty: raise RuntimeError('Missing unshrunk N80 pair table')
    bins=[0,.05,.10,.20,.50]
    labels=['<0.05','0.05–0.10','0.10–0.20','0.20–0.50']
    z['MAF_bin']=pd.cut(z.MAF,bins=bins,labels=labels,include_lowest=True,right=False)
    maxn=max(manifest['grid_N'])
    rng=np.random.default_rng(2026)
    for i,label in enumerate(labels):
        zz=z[z.MAF_bin==label]
        if zz.empty:continue
        val=pd.to_numeric(zz.N80_joint_first_grid,errors='coerce')
        yy=val.fillna(maxn*1.08).to_numpy(float)
        jitter=rng.uniform(-.18,.18,len(yy))
        ax.scatter(i+jitter,yy,s=9,color=COL['PDBP'],alpha=.37)
        if val.notna().any():
            med=float(val.median())
            ax.hlines(med,i-.26,i+.26,colors=COL['PPMI'],lw=2.5)
        ax.text(i,maxn*1.19,f'n={len(zz)}\n>{maxn:,}: {int(val.isna().sum())}',
                ha='center',fontsize=7.1,va='bottom')
    ax.axhline(maxn,color='#808080',ls='--',lw=.7)
    ax.set_xticks(np.arange(len(labels)),labels)
    ax.set_xlim(-.55,len(labels)-.45)
    ax.set_ylim(0,maxn*1.36)
    ax.set(xlabel='Observed MAF bin (no common MAF assumption)',
           ylabel='First tested N with joint power ≥80%',
           title='Conditional N80 by empirical MAF')
    ax.text(.98,.03,'Points > maximum tested N shown above dashed line',transform=ax.transAxes,
            ha='right',va='bottom',fontsize=6.3,color='#55646B')
    clean(ax);panel(ax,'d')


def main():
    p=argparse.ArgumentParser();p.add_argument('--figroot',required=True);a=p.parse_args();config()
    figroot=Path(a.figroot);out=figroot/'power_analysis_CTAR_v6'
    calib=pd.read_csv(out/'calibrated_real_full8_CTAR_PVE.csv')
    summary=pd.read_csv(out/'power_summary_by_N.csv')
    n80=pd.read_csv(out/'N80_by_real_pair.csv')
    manifest=json.loads((out/'power_manifest_v6.json').read_text())
    actual_path=out/'actual_cohort_conditional_power_summary.csv'
    actual=pd.read_csv(actual_path) if actual_path.exists() else None
    full=out/'full_PPMI_PDBP_replication_summary.csv'
    if not full.exists():raise FileNotFoundError(full)
    f,axs=plt.subplots(2,2,figsize=(10.8,7.1))
    replication(axs[0,0],figroot,full)
    observed_pve(axs[0,1],calib)
    n_curve(axs[1,0],summary,manifest,actual)
    n80_maf(axs[1,1],n80,manifest)
    f.subplots_adjust(left=.085,right=.975,top=.95,bottom=.16,hspace=.46,wspace=.38)
    f.text(.5,.075,f'Actual CTAR cohorts: PPMI N={manifest["PPMI_actual_N"]:,}; PDBP N={manifest["PDBP_actual_N"]:,}. '
           'b–d use PPMI fitted 8-parameter genetic trajectories and real donor-visit designs.',
          fontsize=7.5,ha='center',color='#50616B')
    f.text(.5,.048,'b: original Var(genetic effect)/Var(RNA) PVE (descriptive); c–d: fixed-rho 8-df GLS approximation '
           'and empirical BH threshold; selected-hit and effect-shrinkage sensitivity.',
           fontsize=6.8,ha='center',color='#60717D')
    f.text(.5,.021,'c–d condition on an ascertainment-selected IV subset; conditional N80 is not a universal sample-size minimum. '
           'Joint detection assumes equal cohort N and unchanged effects.',
           fontsize=6.8,ha='center',color='#60717D')
    figout=figroot/'supplementary';figout.mkdir(parents=True,exist_ok=True)
    stem=figout/'Supplementary_Figure_6_CTAR_empirical_full8_power_v6'
    for ext in ('pdf','svg','png'):
        f.savefig(str(stem)+'.'+ext,dpi=600 if ext=='png' else None,bbox_inches='tight',pad_inches=.06, backend=(ext if ext in ("svg", "pdf") else None))
    plt.close(f)
    print('[FIGURE]',str(stem)+'.pdf')
    # numerical readout for reviewer and text
    z=summary[np.isclose(summary.effect_multiplier,1.0)]
    x=z[z.n_participants.isin([manifest['PPMI_actual_N'],manifest['PDBP_actual_N']])]
    x.to_csv(out/'power_at_actual_N_v6.csv',index=False)

if __name__=='__main__':main()
