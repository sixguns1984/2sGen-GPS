#!/usr/bin/env python3
"""CTAR practical single-locus N80: discovery (nominal/BH-reference) and replication.

Scientific scope:
* one prespecified variant-gene hypothesis, 8-df CTAR overall LRT;
* compare nominal .05 and the actual PPMI scan's raw BH cutpoint (REFERENCE
  threshold, not a claim of BH q=.05 at hypothetical cohort sizes);
* independently show replication power at .05 and joint discovery-replication;
* represent effect size by prospective PVE at reference MAF=0.20;
* actual real PPMI/PDBP visit templates; analytic OU fixed-rho nc-chi-square,
  NOT the full profile-rho 100-point GPU CTAR-LRT; see README.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('cairo')  # Use Cairo backend for high-quality output
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'svg.fonttype':'none', 'pdf.fonttype':42, 'ps.fonttype':42})
from ctar_sfig6_n80 import (meta_templates, ordered_templates, mode_values,
    ou_grams, compute_power, dynamic_joint_power, overall_power, ncps,
    find_min, resolve_thresholds, sample_loci, empirical_replication,
    COLORS)


def n80(ns,power,target=.8):
    return find_min(ns,power,target)


def format_n(v,max_n):
    return f'{int(v):,}' if np.isfinite(v) else f'>{max_n:,}'


def scalar_power(n,pve,maf,gd,gr,alpha_disc,alpha_rep,ret,f2,mode='discovery',test='temporal',alpha_inter=None):
    a,b,j=compute_power(n,np.asarray([maf]),pve,gd,gr,alpha_disc,alpha_rep,
        alpha_inter=alpha_inter,retain=ret,test=test,f2_reference=f2,
        min_minor_hom=0)  # no arbitrary 5-minor-homozygote hard power cutoff
    return float({'discovery':a[0],'replication':b[0],'joint':j[0]}[mode])


def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    ap.add_argument('--output',default=None)
    ap.add_argument('--metadata',default=None)
    ap.add_argument('--loci-file',default=None)
    ap.add_argument('--replication-file',default=None)
    ap.add_argument('--plink-prefix',default=None)
    ap.add_argument('--existing-loci',default=None,help='Reuse previous SFig6 validated real MAF CSV, avoiding heavy PLINK MAF extraction')
    ap.add_argument('--alpha-disc',type=float,default=None,help='Verified real PPMI genome-wide BH raw-P boundary (reference scenario)')
    ap.add_argument('--alpha-inter',type=float,default=None)
    ap.add_argument('--n-pairs',type=int,default=1500)
    ap.add_argument('--min-pairs',type=int,default=1000)
    ap.add_argument('--seed',type=int,default=20261030)
    ap.add_argument('--min-n',type=int,default=50)
    ap.add_argument('--max-n',type=int,default=5000)
    ap.add_argument('--step-n',type=int,default=50)
    ap.add_argument('--pves',nargs='+',type=float,default=[.005,.01,.02,.04,.08])
    ap.add_argument('--focus-pve',type=float,default=.02)
    ap.add_argument('--reference-maf',type=float,default=.20)
    ap.add_argument('--rho',type=float,default=.5)
    ap.add_argument('--rhos',nargs='+',type=float,default=[0.,.3,.6,.9])
    ap.add_argument('--visits',nargs='+',type=int,default=[2,3,4,5])
    ap.add_argument('--horizon',type=float,default=4.)
    ap.add_argument('--pattern',choices=['linear','late','early','static'],default='linear')
    ap.add_argument('--retention',type=float,default=1.)
    ap.add_argument('--replication-alpha',type=float,default=.05)
    ap.add_argument('--target-power',type=float,default=.8)
    ap.add_argument('--demo',action='store_true')
    args=ap.parse_args()
    if not all(0<p<1 for p in args.pves):ap.error('PVE must be (0,1) fractions, e.g., .02 for 2%')
    if not 0<args.focus_pve<1:ap.error('Bad --focus-pve')
    if not 0<args.reference_maf<=.5:ap.error('Bad reference MAF')
    if args.min_n>=args.max_n:ap.error('Bad sample-size range')
    root=Path(args.root)
    out=Path(args.output or str(root/'manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_practical_v1_3'))
    out.mkdir(parents=True,exist_ok=True)
    qtl=Path(args.loci_file or root/'qtl/PP/submitted_followup_ctar_mptga/final/significant_tqtl.parquet')
    rep=Path(args.replication_file or root/'qtl/PD/submitted_followup_ctar_mptga/final/all_results.parquet')
    if args.demo:
        groups=ordered_templates(seed=args.seed)
        actual={'PPMI':800,'PDBP':800}
        alpha_bh=args.alpha_disc if args.alpha_disc is not None else .001
        alpha_int=args.alpha_inter if args.alpha_inter is not None else .0005
        loci=None
        empirical={'note':'DEMO ONLY'}
        source='synthetic DEMO thresholds'
    else:
        meta=Path(args.metadata or root/'data/metadata/amp_pd_samples.parquet')
        if not meta.is_file():raise FileNotFoundError(meta)
        groups=meta_templates(meta)
        actual={k:len(v) for k,v in groups.items()}
        alpha_bh,alpha_int,source=resolve_thresholds(qtl,args.alpha_disc,args.alpha_inter)
        old=Path(args.existing_loci) if args.existing_loci else root/'manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_N80_final/loci_1500_real_or_demo.csv'
        if old.is_file():
            loci=pd.read_csv(old)
            if not {'variant_id','gene_id','maf'}<=set(loci):raise ValueError('Previous loci CSV missing required fields')
            loci=loci[np.isfinite(pd.to_numeric(loci.maf,errors='coerce')) & loci.maf.between(.005,.5)].copy()
            if len(loci)<args.min_pairs:raise ValueError('Previously validated loci < required minimum')
            loci=loci.head(args.n_pairs)
            print(f'[REUSE] {len(loci)} validated PPMI variants from {old}',flush=True)
        else:
            plink=args.plink_prefix or str(root/'data/genotype/PP/amp_pd_PP_qc')
            loci=sample_loci(qtl,args.seed,args.n_pairs,args.min_pairs,plink)
        empirical=empirical_replication(loci,rep)
        loci.to_csv(out/'loci_1500_real_MAF.csv',index=False)
    ns=np.arange(args.min_n,args.max_n+1,args.step_n,dtype=int)
    gd=ou_grams(groups['PPMI'],args.rho,args.pattern,args.horizon)
    gr=ou_grams(groups['PDBP'],args.rho,args.pattern,args.horizon)
    f2=float(np.mean(mode_values(np.linspace(0,args.horizon,5),args.pattern,args.horizon)**2))
    if f2<=0:raise ValueError('Invalid PVE normalization')
    rows=[];minrows=[]
    for pve in args.pves:
        vectors={k:[] for k in ('discovery_nominal','discovery_BH_reference','replication_nominal',
                                'joint_nominal','joint_BH_reference','dynamic_discovery_BH_reference')}
        for n in ns:
            for key,kind,alpha,test in (
                    ('discovery_nominal','discovery',.05,'temporal'),
                    ('discovery_BH_reference','discovery',alpha_bh,'temporal'),
                    ('replication_nominal','replication',.05,'temporal'),
                    ('joint_nominal','joint',.05,'temporal'),
                    ('joint_BH_reference','joint',alpha_bh,'temporal'),
                    ('dynamic_discovery_BH_reference','discovery',alpha_bh,'dynamic')):
                v=scalar_power(n,pve,args.reference_maf,gd,gr,alpha,args.replication_alpha,
                               args.retention,f2,mode=kind,test=test,alpha_inter=alpha_int)
                vectors[key].append(v)
        for i,n in enumerate(ns):
            rows.append({'pve':pve,'N_each_cohort':n,**{k:v[i] for k,v in vectors.items()}})
        for key,v in vectors.items():minrows.append({'pve':pve,'test':key,'N80':n80(ns,v,args.target_power)})
    curves=pd.DataFrame(rows)
    minimum=pd.DataFrame(minrows)
    curves.to_csv(out/'N_by_PVE_power_curves.csv',index=False)
    minimum.to_csv(out/'N80_by_PVE_and_significance_policy.csv',index=False)
    from ctar_sfig6_n80 import planned_templates
    vrows=[]
    for m in args.visits:
        for rho in args.rhos:
            templates=planned_templates(m,missing=.15,jitter=.08,seed=args.seed,horizon=args.horizon)
            grams=ou_grams(templates,rho,args.pattern,args.horizon)
            x=[scalar_power(n,args.focus_pve,args.reference_maf,grams,grams,alpha_bh,
                            args.replication_alpha,args.retention,f2) for n in ns]
            vrows.append({'visits':m,'rho':rho,'N80_discovery_BH_reference':n80(ns,x,args.target_power)})
    visits=pd.DataFrame(vrows)
    visits.to_csv(out/'N80_by_visits_and_rho.csv',index=False)
    mafrows=[]
    for maf in [.05,.10,.20,.30,.40]:
        for test,alpha in [('nominal',.05),('BH_reference',alpha_bh)]:
            x=[scalar_power(n,args.focus_pve,maf,gd,gr,alpha,args.replication_alpha,args.retention,f2) for n in ns]
            nstar=n80(ns,x,args.target_power)
            mafrows.append({'MAF':maf,'test':test,'N80':nstar,
                            'expected_minor_homozygotes_at_N80':float(nstar*maf**2),
                            'sparse_homozygotes_flag':bool(np.isfinite(nstar) and nstar*maf**2<5)})
    mafs=pd.DataFrame(mafrows)
    mafs.to_csv(out/'N80_by_MAF.csv',index=False)
    # Fixed actual PPMI discovery N, prospective PDBP replication N (not both varied together).
    rep_rows=[]
    n_disc=actual['PPMI']
    for pve in [args.focus_pve,max(args.focus_pve,.04)]:
        d=scalar_power(n_disc,pve,args.reference_maf,gd,gr,alpha_bh,
                       args.replication_alpha,args.retention,f2,'discovery')
        for n_rep in ns:
            r=scalar_power(n_rep,pve,args.reference_maf,gd,gr,alpha_bh,
                           args.replication_alpha,args.retention,f2,'replication')
            rep_rows.append({'pve':pve,'PPMI_N_fixed':n_disc,'PDBP_N':n_rep,
                             'PPMI_discovery_power_BH_reference':d,
                             'PDBP_replication_power_nominal':r,'joint_power':d*r})
    replicas=pd.DataFrame(rep_rows)
    replicas.to_csv(out/'replication_N_with_fixed_actual_PPMI.csv',index=False)
    fig,axs=plt.subplots(2,3,figsize=(16.8,9.7),constrained_layout=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.3,'axes.titlesize':11.5,
                         'axes.linewidth':.85,'pdf.fonttype':42,'ps.fonttype':42})
    palette=[COLORS['gold'],COLORS['teal'],COLORS['blue'],COLORS['navy'],COLORS['purple']]
    ax=axs[0,0]
    for i,pve in enumerate(args.pves):
        d=curves[np.isclose(curves.pve,pve)]
        ax.plot(d.N_each_cohort,d.discovery_nominal,color=palette[i%len(palette)],lw=2.0,
                label=f'PVE {pve*100:g}%')
        ax.plot(d.N_each_cohort,d.discovery_BH_reference,color=palette[i%len(palette)],lw=1.4,ls='--',alpha=.85)
    ax.axhline(.8,ls=':',color=COLORS['gray'])
    ax.axvline(actual['PPMI'],ls=':',color=COLORS['red'],lw=1.2)
    ax.set(title='a  80% power for a single temporal tQTL',xlabel='Independent discovery participants',ylabel='CTAR detection power',ylim=(0,1.01))
    ax.legend(loc='lower right',frameon=False,fontsize=7,ncol=2)
    ax.text(.02,.98,'Solid: nominal P<0.05\nDashed: observed BH raw-P reference',transform=ax.transAxes,
            va='top',fontsize=8,color=COLORS['gray'])
    ax=axs[0,1]
    groups_b=[('discovery_nominal','Nominal P<0.05',COLORS['teal']),('discovery_BH_reference','BH reference',COLORS['navy'])]
    x=np.arange(len(args.pves)); width=.34
    for k,(col,lab,color) in enumerate(groups_b):
        data=minimum[minimum.test==col].set_index('pve').reindex(args.pves)
        values=data.N80.to_numpy(dtype=float)
        ax.bar(x+(k-.5)*width,np.where(np.isfinite(values),values,args.max_n),width,
               color=color,alpha=.91,label=lab)
        for i,val in enumerate(values):
            if not np.isfinite(val):ax.text(x[i]+(k-.5)*width,args.max_n*.99,f'>{args.max_n}',rotation=90,fontsize=7,ha='center',va='top',color='white')
    ax.axhline(actual['PPMI'],lw=1.1,ls=':',color=COLORS['red'],label=f'Actual PPMI N={actual["PPMI"]:,}')
    ax.set(xticks=x,xticklabels=[f'{p*100:g}%' for p in args.pves],
           title='b  Minimum discovery N at 80% power',xlabel='Prospective PVE (MAF 0.20)',
           ylabel='Required independent participants',ylim=(0,args.max_n*1.06))
    ax.legend(frameon=False,fontsize=7,loc='upper right')
    ax=axs[0,2]
    tab=visits.pivot(index='visits',columns='rho',values='N80_discovery_BH_reference').sort_index()
    vv=tab.to_numpy(float);shown=np.where(np.isfinite(vv),vv,args.max_n+args.step_n)
    im=ax.imshow(shown,aspect='auto',cmap='YlGnBu',vmin=args.min_n,vmax=args.max_n)
    ax.set_xticks(np.arange(len(tab.columns)),[f'{i:g}' for i in tab.columns]);ax.set_yticks(np.arange(len(tab.index)),tab.index)
    ax.set(title=f'c  Visits × residual correlation (PVE {args.focus_pve*100:g}%)',xlabel='Within-participant rho per year',ylabel='Visits per participant')
    for i in range(vv.shape[0]):
        for j in range(vv.shape[1]):ax.text(j,i,format_n(vv[i,j],args.max_n),ha='center',va='center',fontsize=8,
                                             color='white' if shown[i,j]>args.max_n*.52 else 'black')
    fig.colorbar(im,ax=ax,fraction=.046,pad=.025,label='N80, BH reference')
    ax=axs[1,0]
    for name,color in [('nominal',COLORS['teal']),('BH_reference',COLORS['navy'])]:
        d=mafs[mafs.test==name]
        ax.plot(d.MAF,np.where(np.isfinite(d.N80),d.N80,np.nan),'-o',color=color,lw=2,
                label='Nominal' if name=='nominal' else 'BH reference')
        for _,r in d[d.N80.isna()].iterrows():ax.text(r.MAF,args.max_n*.97,f'>{args.max_n}',ha='center',fontsize=7,color=color)
        for _,r in d[d.sparse_homozygotes_flag].iterrows():
            ax.scatter([r.MAF],[r.N80],facecolor='none',edgecolor=COLORS['red'],s=100,lw=1.4,zorder=8)
    ax.set(title=f'd  Allele frequency matters (PVE {args.focus_pve*100:g}%)',xlabel='SNP minor allele frequency (MAF)',
           ylabel='Single-locus discovery N80',ylim=(0,args.max_n*1.07),xlim=(.035,.415))
    ax.legend(frameon=False,fontsize=8)
    ax.text(.98,.98,'Red ring: expected minor homozygotes <5\nApproximation uncertain',
            transform=ax.transAxes,va='top',ha='right',fontsize=7,color=COLORS['red'])
    ax=axs[1,1]
    for pve,color in [(args.focus_pve,COLORS['teal']),(max(args.focus_pve,.04),COLORS['navy'])]:
        d=replicas[np.isclose(replicas.pve,pve)]
        ax.plot(d.PDBP_N,d.PDBP_replication_power_nominal,lw=2.0,color=color,label=f'Replication: PVE {pve*100:g}%')
        ax.plot(d.PDBP_N,d.joint_power,lw=1.4,color=color,ls='--',label=f'Joint: PVE {pve*100:g}%')
    ax.axhline(.8,ls=':',lw=1,color=COLORS['gray'])
    ax.axvline(actual['PDBP'],ls=':',lw=1.1,color=COLORS['red'])
    ax.set(title='e  Replication N with discovery N fixed',xlabel='Independent PDBP participants',
           ylabel='Prospective statistical power',ylim=(0,1.02))
    ax.legend(frameon=False,fontsize=7,loc='lower right')
    ax=axs[1,2]
    d=curves[np.isclose(curves.pve,args.focus_pve)]
    ax.plot(d.N_each_cohort,d.discovery_BH_reference,lw=2.0,color=COLORS['navy'],label='Temporal discovery: 8 df')
    ax.plot(d.N_each_cohort,d.dynamic_discovery_BH_reference,lw=2.0,color=COLORS['red'],label='Dynamic confirmation: 6 df')
    ax.plot(d.N_each_cohort,d.joint_BH_reference,lw=1.5,ls='--',color=COLORS['teal'],label='Temporal discovery + replication')
    ax.axhline(.8,color=COLORS['gray'],ls=':',lw=1)
    ax.axvline(actual['PPMI'],color=COLORS['red'],ls=':',lw=1)
    ax.set(title=f'f  Different research objectives (PVE {args.focus_pve*100:g}%)',
           xlabel='Independent participants per cohort',ylabel='Power',ylim=(0,1.02))
    if empirical.get('rate') is not None:
        ax.text(.98,.04,f"Observed nominal replication: {empirical['replicated']}/{empirical['matched']} ({empirical['rate']*100:.1f}%)",
                transform=ax.transAxes,ha='right',va='bottom',fontsize=7.5,
                bbox=dict(facecolor='white',edgecolor='#DDE3E8',pad=3))
    ax.legend(frameon=False,loc='upper left',fontsize=7)
    for ax in axs.ravel():
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.spines[['right','top']].set_visible(False)
    fig.suptitle('CTAR-MPTGA: practical sample-size planning by effect size and study objective',fontsize=15.2,fontweight='bold')
    fig.supxlabel(('DEMONSTRATION ONLY — synthetic visits and thresholds' if args.demo else
                   'Real PPMI/PDBP visit templates and observed PPMI BH cutoff')+
                   '  |  Fixed-rho OU / noncentral chi-square approximation; 80% power per prespecified locus',
                   fontsize=8.6,color=COLORS['gray'])
    for ext in ('png','pdf','svg'):
        fig.savefig(out/f'Supplementary_Figure_6_CTAR_practical.{ext}',dpi=300,bbox_inches='tight', backend=(ext if ext in ("svg", "pdf") else None))
    plt.close(fig)
    manifest={'version':'v1.3-practical','demo':bool(args.demo),'cohort_N':actual,
              'cutoff_reference':alpha_bh,'alpha_stagewise':alpha_int,'threshold_source':source,
              'empirical_replication':empirical,'n_loci_maf_audited':0 if loci is None else len(loci),
              'n_genes':None if loci is None else int(loci.gene_id.nunique()),
              'reference_maf':args.reference_maf,'focus_pve':args.focus_pve,'rho':args.rho,
              'discovery_nominal':0.05,'replication_nominal':args.replication_alpha,
              'assumptions':['PVE defined at reference MAF and 5 visits','OU fixed rho, not profiled rho',
                             'Observed BH raw-P is a REFERENCE threshold, not recalculated at each candidate N',
                             'Each N80 is for ONE prespecified locus at a given PVE and MAF',
                             'No ad hoc minor homozygote hard cutoff: QC requires separate genotype-count verification',
                             'Joint discovery+replication N is distinct from discovery N80',
                             'PLINK MAF-locus selection is descriptive, not a distribution of ALL causal QTLs']}
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=float))
    print('[SUCCESS] Output:',out,flush=True)
    print('[THRESHOLD] actual PPMI raw-P BH reference:',alpha_bh,flush=True)
    print('[ACTUAL] donors:',actual,'; empirical replication:',empirical,flush=True)
    print(minimum.to_string(index=False),flush=True)

if __name__=='__main__':main()
