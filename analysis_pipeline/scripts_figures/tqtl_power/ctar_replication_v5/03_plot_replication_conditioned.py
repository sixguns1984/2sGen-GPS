#!/usr/bin/env python3
"""Publication figure mirroring original S6 goals, without dynamic-QTL panels."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('cairo')  # Use Cairo backend for high-quality output
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'svg.fonttype':'none', 'pdf.fonttype':42, 'ps.fonttype':42, 'pdf.use14corefonts':False})
from scipy.stats import gaussian_kde

COL={'PP':'#2B5678','PD':'#248D8B','line':'#BE4B55','orange':'#D49B52','gray':'#6F7F8D'}

def cluster_ci(frame,cols=('PP','PD','joint'),seed=2468,B=500):
    rng=np.random.default_rng(seed)
    if frame.empty:return {}
    genes=frame.gene.astype(str).unique()
    out={}
    for c in cols:
        if c not in frame.columns:continue
        val=(frame[c].to_numpy(float)>=.8).astype(float)
        groups={g:val[frame.gene.to_numpy(str)==g] for g in genes}
        sims=np.empty(B)
        for i in range(B):
            draws=rng.choice(genes,len(genes),replace=True)
            arrays=[groups[g] for g in draws]
            sims[i]=np.concatenate(arrays).mean()
        out[c]={'point':float(np.mean(val)),'ci_low':float(np.quantile(sims,.025)),
                'ci_high':float(np.quantile(sims,.975))}
    return out

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt');a.add_argument('--outdir');a.add_argument('--bootstraps',type=int,default=300);a.add_argument('--allow-lrt-mismatch',action='store_true');a.add_argument('--plot-only',action='store_true',help='Reuse existing matched_pair_power.csv and power_summary.json without rewriting numerical results');x=a.parse_args()
    out=Path(x.outdir or Path(x.root)/'manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_refit_v3')
    r=pd.read_csv(out/'locus_bootstrap_results.csv')
    r['gene']=r['gene'].astype(str);r['variant']=r['variant'].astype(str)
    assert set(r.cohort).issubset({'PP','PD'})
    pp=r.query('cohort=="PP"').copy(); pd_=r.query('cohort=="PD"').copy()
    if x.plot_only:
        # The real v5 run has already produced these summaries. Never overwrite
        # completed bootstrap/power numerical results during an editable redraw.
        for name in ('locus_bootstrap_results.csv', 'matched_pair_power.csv', 'power_summary.json'):
            p=out/name
            if not p.is_file() or p.stat().st_size==0:
                raise FileNotFoundError(f'[PLOT ONLY] Required completed result missing: {p}')
        paired=pd.read_csv(out/'matched_pair_power.csv')
        summary=json.loads((out/'power_summary.json').read_text())
        for c in ('variant','gene','PP','PD','joint'):
            if c not in paired.columns:
                raise ValueError(f'[PLOT ONLY] Missing {c} in matched_pair_power.csv')
        for c in ('PPMI','PDBP','Matched'):
            if c not in summary:
                raise ValueError(f'[PLOT ONLY] Missing {c} in power_summary.json')
        if int(summary['PPMI']['n'])!=len(pp) or int(summary['PDBP']['n'])!=len(pd_):
            raise ValueError('[PLOT ONLY] Summary n mismatches cohort rows in locus_bootstrap_results.csv')
        if int(summary['Matched']['n'])!=len(paired):
            raise ValueError('[PLOT ONLY] Summary matched n mismatches matched_pair_power.csv')
        print('[PLOT ONLY] Reusing existing result tables. Numerical files will not be modified.',flush=True)
    else:
        paired=pp[['variant','gene','bootstrap_power']].merge(
            pd_[['variant','gene','bootstrap_power']],on=['variant','gene'],suffixes=('_PP','_PD'),validate='one_to_one')
        paired['joint']=paired.bootstrap_power_PP*paired.bootstrap_power_PD
        paired=paired.rename(columns={'bootstrap_power_PP':'PP','bootstrap_power_PD':'PD'})
        paired.to_csv(out/'matched_pair_power.csv',index=False)
        summary={'PPMI':{'n':len(pp),'n80':int((pp.bootstrap_power>=.8).sum()),'fraction80':float((pp.bootstrap_power>=.8).mean())},
                 'PDBP':{'n':len(pd_),'n80':int((pd_.bootstrap_power>=.8).sum()),'fraction80':float((pd_.bootstrap_power>=.8).mean())},
                 'Matched':{'n':len(paired),'both_above80':int(((paired.PP>=.8)&(paired.PD>=.8)).sum()),
                            'joint_product_above80':int((paired.joint>=.8).sum())},
                 'note':'Subject-bootstrap re-detection rates conditional on observed samples and effects; not prospective power'}
        summary['cluster_ci_matched']=cluster_ci(paired,B=x.bootstraps)
        if not pp.empty:
            valid=pp[np.isfinite(pp.get('lrt_abs_diff',pd.Series(np.nan,index=pp.index)))]
            summary['model_LRT_audit']={'n':len(valid),'median_abs_difference':float(valid.lrt_abs_diff.median()) if len(valid) else None,
               'fraction_abs_difference_gt_2':float((valid.lrt_abs_diff>2).mean()) if len(valid) else None}
        (out/'power_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
        audit=summary.get('model_LRT_audit',{})
        if audit.get('n',0)<5 and len(pp)>=20 and not x.allow_lrt_mismatch:
            raise ValueError('Insufficient Stage-I vs refit LRT comparison. Use --allow-lrt-mismatch only for debugging')
        if audit.get('median_abs_difference') is not None and audit['median_abs_difference']>2 and not x.allow_lrt_mismatch:
            raise ValueError('CTAR refit does not reproduce Stage-I LRT. Median absolute difference: '+str(audit['median_abs_difference'])+'; fix model/covariates/rho before plotting')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.85,'savefig.dpi':260})
    fig,ax=plt.subplots(2,3,figsize=(15.6,8.8),constrained_layout=True)
    A,B,C,D,E,F=ax.ravel()
    # a: representative real-bootstrap alternative LRT densities (if saved)
    dist=[]
    for p in sorted((out/'bootstrap_LRT_examples').glob('*_PP.csv')):
        try:
            d=pd.read_csv(p)
            idx=int(p.name.split('_')[0])
            row=pp[(pp.gene==str(d.gene.iloc[0]))&(pp.variant==str(d.variant.iloc[0]))]
            if len(row)>0 and len(d)>=50: dist.append((abs(float(row.bootstrap_power.iloc[0])-.8),p,d,row))
        except Exception:pass
    dist.sort(key=lambda z:z[0]);samples=[]
    if len(dist)>=2:
        samples=[dist[0]]
        remainder=sorted([r for r in dist[1:] if r[1]!=dist[0][1]],key=lambda z:abs(float(z[3].bootstrap_power.iloc[0])-.9))
        if remainder:samples.append(remainder[0])
    for idx,(_,p,d,row) in enumerate(samples):
        v=d.lrt.to_numpy(float);v=v[np.isfinite(v)]
        if len(v)<10 or np.std(v)<=0:continue
        xs=np.linspace(0,np.quantile(v,.995)*1.12,350)
        A.plot(xs,gaussian_kde(v)(xs),color=[COL['PP'],COL['orange']][idx],lw=2,
              label=f'{row.gene.iloc[0]}: {row.bootstrap_power.iloc[0]:.0%}')
    if len(pp):
        cutoff=float(pp.alpha_test.iloc[0]);from scipy.stats import chi2
        critical=float(chi2.isf(cutoff,8));A.axvline(critical,ls='--',c=COL['line'],lw=1.3,label='PPMI threshold')
    A.set(xlabel='Overall CTAR 8-df LRT',ylabel='Bootstrap density',title='a  Refit LRT distributions');A.legend(frameon=False,fontsize=8)
    for axis,df,color,title in [(B,pp,COL['PP'],'b  PPMI discovery'),(C,pd_,COL['PD'],'c  PDBP replication')]:
        v=df.bootstrap_power.dropna().to_numpy(float)
        axis.hist(v,bins=np.linspace(0,1,22),color=color,alpha=.87,edgecolor='white',linewidth=.5)
        axis.axvline(.8,color=COL['line'],ls='--',lw=1.5)
        n80=int((v>=.8).sum());axis.text(.97,.96,f'≥80%: {n80:,}/{len(v):,} ({100*n80/max(len(v),1):.1f}%)',
             transform=axis.transAxes,ha='right',va='top',weight='bold',fontsize=9)
        axis.set(xlim=(0,1),xlabel='Donor-bootstrap detection fraction',ylabel='SNP–gene pairs',title=title)
    sensfile=out/'donor_sample_size_sensitivity.csv'
    if sensfile.exists():
        s=pd.read_csv(sensfile)
        if len(s):
            agg=s.assign(power80=s.frac_pass>=.8).groupby(['cohort','fraction'],as_index=False)['power80'].mean()
            for co in ('PP','PD'):
                z=agg[agg.cohort==co]
                D.plot(z.fraction*100,z.power80*100,'o-',color=COL[co],label=('PPMI' if co=='PP' else 'PDBP'))
        D.set(xlabel='Available participants (%)',ylabel='Evaluated loci with ≥80% bootstrap detection (%)',title='d  Donor subsampling sensitivity')
        D.legend(frameon=False,fontsize=8)
    else:D.text(.5,.5,'Enable --sensitivity-reps for actual donor subsampling',transform=D.transAxes,ha='center',va='center',wrap=True)
    # e: PDBP sensitivity in a PRE-SELECTED replicated cohort. Do not plot observed full-cohort
    # replication as evidence: it is exactly 100% by selection and is tautological.
    E.text(.06,.92,'PDBP P<0.05 was required for locus selection;\nfull-sample replication is therefore 100% by design.',
           transform=E.transAxes,fontsize=8,color=COL['gray'],va='top')
    if sensfile.exists():
        # Subset rate among fixed significant PP hits, only for the sensitivity-locus subset
        s=pd.read_csv(sensfile)
        ss=s[(s.cohort=='PD')]
        if len(ss):
            agg=ss.groupby('fraction',as_index=False)['frac_pass'].mean()
            E.plot(agg.fraction*100,agg.frac_pass*100,'o--',color=COL['PD'],label='Subset CTAR refit rate (locus subset)')
    E.set(xlim=(20,105),ylim=(0,100),xlabel='PDBP participants (% of full cohort)',ylabel='Nominal re-detection (%)',title='e  Replication-cohort subsampling (selected loci)')
    E.legend(frameon=False,fontsize=8)
    values=[summary['PPMI']['fraction80']*100,summary['PDBP']['fraction80']*100,100*float((paired.PP.ge(.8)&paired.PD.ge(.8)).mean()) if len(paired) else 0]
    labels=['PPMI','PDBP','Both ≥80%'];bars=F.bar(labels,values,color=[COL['PP'],COL['PD'],COL['orange']],width=.58)
    for bar,v in zip(bars,values):F.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.2,f'{v:.1f}%',ha='center',fontsize=9)
    F.set(ylim=(0,105),ylabel='Loci reaching ≥80% bootstrap detection (%)',title='f  Summary at observed cohort sizes')
    fig.suptitle('CTAR-MPTGA: bootstrap redetection among replicated temporal tQTLs',fontsize=15,weight='bold')
    fig.text(.5,-.007,'PPMI BH + PDBP nominal replicated loci | overall 8-df CTAR | conditional donor-bootstrap detection; not unbiased prospective power',
        ha='center',fontsize=9,color=COL['gray'])
    for ext in ('png','pdf','svg'):
        fig.savefig(out/f'Supplementary_Figure_6_CTAR_REPLICATED_v5.{ext}',bbox_inches='tight',dpi=260, backend=(ext if ext in ('svg', 'pdf') else None))
    plt.close(fig)
    print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
