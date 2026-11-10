#!/usr/bin/env python3
"""SFig6: real selected temporal tQTL power, overall 8-df CTAR LRT ONLY.

This is an empirical-Bayes noncentral-chi-square approximation to refitted
CTAR profile-likelihood bootstrap. It avoids the observed-power = p-value trap,
accounts for PPMI discovery selection, uses independent PDBP statistics to
estimate the signal-strength distribution, and reports transparent assumptions.
No genotype x time (dynamic) tests are performed anywhere.
"""
from __future__ import annotations
import argparse, json, math, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import chi2, ncx2
import matplotlib
matplotlib.use('cairo')  # Use Cairo backend for high-quality output
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'svg.fonttype':'none', 'pdf.fonttype':42, 'ps.fonttype':42})

DF=8
PALETTE={'navy':'#214D70','blue':'#4B8CB7','teal':'#218C88','orange':'#CF9654',
         'red':'#C15D65','gray':'#788594','light':'#E6EDF2'}

def colname(columns, names, required=True):
    canonical={str(c).lower().replace('_','').replace('-',''):c for c in columns}
    for name in names:
        k=name.lower().replace('_','').replace('-','')
        if k in canonical:return canonical[k]
    if required:raise ValueError(f'Required column {names} absent; present: {list(columns)[:65]}')
    return None

def schema(path):
    p=Path(path)
    if p.suffix=='.csv':return list(pd.read_csv(p,nrows=0).columns)
    import pyarrow.dataset as ds
    return ds.dataset(str(p),format='parquet').schema.names

def iter_batches(path,columns,batch_size=180000):
    p=Path(path)
    if p.suffix=='.csv':
        for chunk in pd.read_csv(p,usecols=columns,chunksize=batch_size):yield chunk
    else:
        import pyarrow.dataset as ds
        data=ds.dataset(str(p),format='parquet')
        for batch in data.to_batches(columns=columns,batch_size=batch_size):
            yield batch.to_pandas()

def column_spec(path,kind):
    s=schema(path)
    specs={'gene': ['Ensembl_ID','gene_id','gene','phenotype_id'],
           'variant':['Variant_ID','variant_id','snp_id','SNP','rsid'],
           'p':['Pvalue','overall_pvalue','temporal_pvalue','p_value'],
           'fdr':['FDR','overall_FDR','qvalue','q_value'],
           'lrt':['LRT','Overall_LRT','LR_statistic','LRT_statistic'],
           'npart':['n_participants','n_donors','n_individuals','N_participants'],
           'nobs':['n_obs','n_observations','n_samples']}
    matched={k:colname(s,v,required=(k in ('gene','variant','p'))) for k,v in specs.items()}
    if not matched['lrt'] and not matched['p']:
        raise ValueError(f'No overall LRT or Pvalue in {path}')
    if kind=='discovery' and matched['fdr'] is None:
        if 'significant_tqtl' not in str(path):
            raise ValueError('Discovery input has no FDR: use the finalized significant_tqtl.parquet')
        warnings.warn('FDR field absent: assuming this is a complete, prefiltered significant_tqtl file.')
    return matched

def normalize_frame(frame, spec):
    rename={src:dst for dst,src in spec.items() if src is not None}
    frame=frame.rename(columns=rename)
    frame['gene']=frame['gene'].astype(str).str.strip()
    frame['variant']=frame['variant'].astype(str).str.strip()
    frame['p']=pd.to_numeric(frame['p'],errors='coerce')
    if 'lrt' in frame:
        frame['lrt']=pd.to_numeric(frame['lrt'],errors='coerce')
    else:
        frame['lrt']=chi2.isf(np.clip(frame['p'].astype(float),1e-300,1),DF)
    if 'fdr' in frame:frame['fdr']=pd.to_numeric(frame['fdr'],errors='coerce')
    for k in ('npart','nobs'):
        if k in frame:frame[k]=pd.to_numeric(frame[k],errors='coerce')
    return frame

def audit_lrt_vs_p(df, label):
    """Catch accidental use of a non-8df LRT or mislabeled Pvalue field."""
    good=df['p'].between(1e-100,0.99)&np.isfinite(df['lrt'])
    if int(good.sum())<25:
        print(f'[WARN] {label}: insufficient moderate-p rows for 8-df LRT/P consistency audit',flush=True)
        return
    expected=chi2.sf(df.loc[good,'lrt'].to_numpy(float),DF)
    actual=df.loc[good,'p'].to_numpy(float)
    discrepancy=np.nanmedian(np.abs(np.log10(np.clip(expected,1e-300,1))-
                                      np.log10(np.clip(actual,1e-300,1))))
    if discrepancy>.20:
        raise ValueError(f'{label}: LRT and Pvalue disagree with an 8-df chi-square reference '
                         f'(median absolute log10 P difference={discrepancy:.3g}). '
                         'Do not use these summary statistics for the assumed 8-df power model.')
    print(f'[AUDIT] {label}: 8-df LRT/P consistency median log10 difference={discrepancy:.3g}',flush=True)


def sample_discovery(path,n,seed,alpha=.05):
    spec=column_spec(path,'discovery')
    cols=list(dict.fromkeys(x for x in spec.values() if x is not None))
    rng=np.random.default_rng(seed)
    selected=[];n_pass=0;p_cutoff=-1.;n_seen=0
    for chunk in iter_batches(path,cols):
        x=normalize_frame(chunk,spec)
        valid=x['p'].between(0,1)&np.isfinite(x['lrt'])&(x['lrt']>=0)
        valid &= (x['variant']!='')&(x['gene']!='')
        if 'fdr' in x:valid &= x['fdr'].between(0,alpha, inclusive='left')
        x=x.loc[valid].copy()
        if not len(x):continue
        n_pass+=len(x)
        p_cutoff=max(p_cutoff,float(x['p'].max()))
        x['_priority']=rng.random(len(x))
        selected.append(x)
        if sum(len(z) for z in selected)>max(3*n,15000):
            selected=[pd.concat(selected,ignore_index=True).nsmallest(n,'_priority')]
    if not selected:raise ValueError('No significant overall temporal tQTL rows were found')
    sampled=pd.concat(selected,ignore_index=True).nsmallest(n,'_priority').drop(columns=['_priority'])
    if len(sampled)<min(n,1000):warnings.warn(f'Only {len(sampled)} selected pairs (<1000)')
    audit_lrt_vs_p(sampled,'PPMI')
    if sampled.duplicated(['variant','gene']).any():
        raise ValueError('Duplicated SNP-gene keys in discovery; audit Stage-I output')
    if p_cutoff<=0:raise ValueError('Cannot estimate positive raw discovery cutoff from hits; provide --alpha-disc explicitly')
    return sampled.reset_index(drop=True),{'n_discovery_hits':int(n_pass),'sampled':len(sampled),
      'max_selected_raw_p':p_cutoff,'discovery_p_threshold':p_cutoff,'spec':spec}

def load_replication(path,sample):
    spec=column_spec(path,'replication')
    cols=list(dict.fromkeys(x for x in spec.values() if x is not None))
    wanted=set(zip(sample.variant.astype(str),sample.gene.astype(str)))
    variants=set(sample.variant.astype(str))
    found=[];n_scanned=0
    for chunk in iter_batches(path,cols):
        n_scanned+=len(chunk)
        x=normalize_frame(chunk,spec)
        x=x.loc[x.variant.isin(variants)]
        if not len(x):continue
        keys=list(zip(x.variant.astype(str),x.gene.astype(str)))
        take=np.fromiter((k in wanted for k in keys),bool,count=len(keys))
        if take.any():found.append(x.loc[take])
    if not found:raise ValueError(f'No PDBP variant-gene keys match {path}. Check source files and ID conventions.')
    z=pd.concat(found,ignore_index=True)
    audit_lrt_vs_p(z,'PDBP')
    if z.duplicated(['variant','gene']).any():raise ValueError('Duplicate PDBP variant/gene rows')
    z=z.rename(columns={x:'rep_'+x for x in z.columns if x not in ('variant','gene')})
    merged=sample.merge(z,on=['variant','gene'],how='left',validate='one_to_one')
    merged['rep_p']=pd.to_numeric(merged['rep_p'],errors='coerce')
    merged['rep_lrt']=pd.to_numeric(merged['rep_lrt'],errors='coerce')
    matched=merged['rep_p'].between(0,1)&np.isfinite(merged['rep_lrt'])
    return merged,{'replication_file':str(path),'scanned_rows':n_scanned,
        'matched':int(matched.sum()),'unmatched':int((~matched).sum()),
        'replicated_p005':int((merged.rep_p[matched]<.05).sum()),'spec':spec}

def lambda_grid(x,points=120):
    x=np.asarray(x,float)
    ceiling=max(150.,float(np.nanmax(x))*1.20)
    return np.r_[0.,np.geomspace(.02,ceiling,points-1)]

def fit_rep_prior(x,grid,n_iter=90):
    x=np.asarray(x,float)
    if len(x)<50:raise ValueError('Need >=50 independent PDBP-tested tQTL pairs to calibrate prior')
    # The PDBP set is not selected on PDBP significance: no PDBP truncation.
    logf=ncx2.logpdf(np.maximum(x[:,None],1e-300),DF,grid[None,:])
    if not np.all(np.isfinite(logsumexp(logf,axis=1))):
        raise ValueError('Unstable noncentral chi-square fit: inspect LRT scale / 8df assumption')
    w=np.ones(len(grid))/len(grid)
    # mild pseudocount avoids degenerate all-or-none EB weights
    for _ in range(n_iter):
        prob=np.exp(logf+np.log(w[None,:])-logsumexp(logf+np.log(w[None,:]),axis=1,keepdims=True))
        w=(prob.sum(axis=0)+0.15)/(len(x)+.15*len(grid))
        w/=w.sum()
    return w

def posterior_and_power(df,alpha_disc,alpha_rep=.05,info='participants',prior_iterations=90):
    valid=df['rep_lrt'].notna()&np.isfinite(df['rep_lrt'])&df['rep_p'].between(0,1)
    m=df.loc[valid].copy().reset_index(drop=True)
    if len(m)<50:raise ValueError(f'Only {len(m)} matched pairs, need >=50')
    if alpha_disc<=0 or not alpha_disc<1:raise ValueError('Invalid discovery P threshold')
    g=lambda_grid(m.rep_lrt.to_numpy())
    w=fit_rep_prior(m.rep_lrt.to_numpy(),g,prior_iterations)
    a=colname(m.columns,['npart','n_participants'] if info=='participants' else ['nobs'],False)
    b=colname(m.columns,['rep_npart','rep_n_participants'] if info=='participants' else ['rep_nobs'],False)
    if a is None or b is None:
        raise ValueError(f'Required {info} per-cohort counts missing. Do not invent cohort N.')
    n_pp=pd.to_numeric(m[a],errors='coerce').to_numpy(float)
    n_pd=pd.to_numeric(m[b],errors='coerce').to_numpy(float)
    if np.any(~np.isfinite(n_pp)|~np.isfinite(n_pd)|(n_pp<=0)|(n_pd<=0)):
        raise ValueError(f'Nonfinite or nonpositive {info} counts for matched pairs')
    ratio=n_pp/n_pd
    lp=g[None,:]*ratio[:,None]
    # Discovery was selected: model PPMI overall LRT from truncated 8-df distribution.
    lpp=np.maximum(m.lrt.to_numpy(float),1e-300)
    lpd=np.maximum(m.rep_lrt.to_numpy(float),1e-300)
    c_disc=chi2.isf(alpha_disc,DF)
    # If selected-qtl list includes p>observed max due to manual override, explain.
    if np.any(m.p.to_numpy(float)>alpha_disc*(1+1e-8)):
        raise ValueError('Some discovered P values exceed alpha_disc; audit discovery selection.')
    logw=(np.log(w[None,:])
          +ncx2.logpdf(lpp[:,None],DF,lp)
          -ncx2.logsf(c_disc,DF,lp)
          +ncx2.logpdf(lpd[:,None],DF,g[None,:]))
    norm=logsumexp(logw,axis=1,keepdims=True)
    if not np.all(np.isfinite(norm)):
        raise ValueError('EB posterior normalization failed; inspect LRT values and lambda grid')
    weights=np.exp(logw-norm)
    component_pp=ncx2.sf(c_disc,DF,lp)
    component_pd=ncx2.sf(chi2.isf(alpha_rep,DF),DF,g[None,:])
    pow_pp=(weights*component_pp).sum(axis=1)
    pow_pd=(weights*component_pd).sum(axis=1)
    pow_joint=(weights*component_pp*component_pd).sum(axis=1)
    m['power_ppmi']=pow_pp;m['power_pdbp']=pow_pd
    m['power_joint']=pow_joint # E[p_PP(theta) * p_PD(theta) | observed summary data]
    # Diagnostic only: the naive inversion of already-selected significance is
    # subject to winner's curse. Do not make it the main reported estimate.
    m['naive_power_ppmi']=ncx2.sf(c_disc,DF,np.maximum(m.lrt.to_numpy(float)-DF,0))
    m['naive_power_pdbp']=ncx2.sf(chi2.isf(alpha_rep,DF),DF,np.maximum(m.rep_lrt.to_numpy(float)-DF,0))
    m['posterior_lambda_pd']=(weights*g[None,:]).sum(axis=1)
    m['posterior_lambda_pp']=(weights*lp).sum(axis=1)
    return m,{'grid':g,'weights':w,'per_locus':weights,'ratio':ratio,
              'lambda_pp':lp,'lambda_pd':g,'info_scale':info,
              'alpha_disc':alpha_disc,'alpha_rep':alpha_rep}

def power_at_fraction(state, fraction, cohort):
    wt=state['per_locus']
    if cohort=='PPMI':
        lam=state['lambda_pp']*fraction
        a=state['alpha_disc']
    else:
        lam=state['lambda_pd'][None,:]*fraction
        a=state['alpha_rep']
    return (wt*ncx2.sf(chi2.isf(a,DF),DF,lam)).sum(axis=1)

def ci_cluster_by_gene(df,columns,nboot=500,seed=11):
    rng=np.random.default_rng(seed)
    genes=df['gene'].astype(str).to_numpy()
    unique,idx=np.unique(genes,return_inverse=True)
    n_gene=len(unique)
    cnt=np.bincount(idx).astype(float)
    ci={}
    for col in columns:
        arr=(df[col].to_numpy(float)>=.8).astype(float)
        num=np.bincount(idx,weights=arr,minlength=n_gene)
        if n_gene<=2:ci[col]=[float('nan'),float('nan')];continue
        pick=rng.integers(0,n_gene,size=(nboot,n_gene))
        fractions=num[pick].sum(axis=1)/np.maximum(cnt[pick].sum(axis=1),1)
        ci[col]=np.percentile(fractions,[2.5,97.5]).tolist()
    return ci

def make_summary(df,meta,ci):
    rep=(df.rep_p<.05)
    n=len(df)
    res={'selection':'PPMI overall temporal FDR<0.05; PDBP matched tested pairs',
         'sampled_ppmi':meta['sampled'],'n_ppmi_screened':meta['n_discovery_hits'],
         'n_pdbp_matched':n,'pdbp_observed_p005':int(rep.sum()),
         'observed_replication_rate':float(rep.mean()),
         'effect_basis':'Empirical-Bayes 8df noncentral-chi-square conditional LRT, selection correction',
         'warning':'Not a direct donor-bootstrap or CTAR profile refit; power estimates are approximate and model dependent.'}
    for name,col in [('discovery','power_ppmi'),('replication','power_pdbp'),('both_80','power_joint')]:
        k=int((df[col]>=.8).sum())
        res[name]={'n_above_80':k,'n_evaluated':n,'fraction':k/n,'gene_bootstrap_ci95':ci[col]}
    both_components=((df.power_ppmi>=.8)&(df.power_pdbp>=.8))
    res['both_components_80']={'n_above_80':int(both_components.sum()),'fraction':float(both_components.mean())}
    return res

def format_pc(x):return f'{x:.1%}'

def plot_figure(df,state,summary,output,seed=11):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10.8,
        'axes.labelsize':9.5,'axes.linewidth':.75,'figure.facecolor':'white'})
    fig,axes=plt.subplots(2,3,figsize=(16,9),constrained_layout=True)
    for ax in axes.ravel():
        ax.spines[['top','right']].set_visible(False)
        ax.set_axisbelow(True);ax.grid(alpha=.14,axis='y')
    rng=np.random.default_rng(seed)
    # (a) posterior predictive alternative distributions, near 80% and 90% power.
    ax=axes[0,0]
    xgrid=np.linspace(chi2.isf(.999,DF),max(chi2.isf(.00001,DF)*2,100),350)
    for target,color in [(0.8,PALETTE['blue']),(0.9,PALETTE['orange'])]:
        i=int(np.argmin(np.abs(df.power_ppmi.to_numpy()-target)))
        wd=state['per_locus'][i,:]
        noncen=state['lambda_pp'][i,:]
        density=np.sum(ncx2.pdf(xgrid[:,None],DF,noncen[None,:])*wd[None,:],axis=1)
        ax.plot(xgrid,density,lw=1.8,color=color,label=f"Example power {df.power_ppmi.iat[i]:.0%}")
    cut=chi2.isf(state['alpha_disc'],DF)
    ax.axvline(cut,ls='--',color=PALETTE['red'],lw=1.15,label='Actual PPMI cutoff')
    ax.set(xlabel='Overall 8-df likelihood-ratio statistic',ylabel='Predictive density',
           title='a  Posterior-predictive alternative distributions')
    ax.legend(frameon=False,fontsize=8)
    # (b, c) native power distributions
    bins=np.linspace(0,1,21)
    for ax,key,title,color in [(axes[0,1],'power_ppmi','b  PPMI temporal tQTL discovery',PALETTE['navy']),
                               (axes[0,2],'power_pdbp','c  PDBP nominal replication',PALETTE['teal'])]:
        vals=df[key].to_numpy(float)
        ax.hist(vals,bins=bins,color=color,alpha=.88,edgecolor='white',linewidth=.4)
        ax.axvline(.8,ls='--',lw=1.3,color=PALETTE['red'])
        k=int((vals>=.8).sum());N=len(vals)
        ax.text(.97,.96,f'>=80% power: {k:,}/{N:,}\n({k/N:.1%})',transform=ax.transAxes,ha='right',va='top',fontweight='bold',fontsize=9)
        ax.set(xlabel='Estimated statistical power',ylabel='Number of SNP–gene pairs',title=title,xlim=(0,1))
    # (d) original supplementary fig-like sample reduction
    ax=axes[1,0]; fractions=np.array([.25,.4,.5,.6,.75,.85,1.0])
    for cohort,color in [('PPMI',PALETTE['navy']),('PDBP',PALETTE['teal'])]:
        prop=[np.mean(power_at_fraction(state,float(fr),cohort)>=.8) for fr in fractions]
        ax.plot(fractions*100,np.array(prop)*100,'o-',lw=1.8,ms=4,color=color,label=cohort)
    ax.set(xlabel='Percent of actual cohort sample size',ylabel='Pairs with estimated power ≥80% (%)',
           title='d  Sensitivity to participant subsampling',ylim=(0,100),xlim=(20,103))
    ax.legend(frameon=False)
    # (e) direct 2-cohort power concordance
    ax=axes[1,1]
    ax.scatter(df.power_ppmi,df.power_pdbp,s=9,alpha=.22,c=PALETTE['blue'],rasterized=True)
    ax.axhline(.8,color=PALETTE['red'],lw=1,ls='--');ax.axvline(.8,color=PALETTE['red'],lw=1,ls='--')
    ax.set(xlabel='PPMI discovery power',ylabel='PDBP replication power',
           title='e  Per-locus discovery vs replication',xlim=(0,1.02),ylim=(0,1.02))
    # (f) summary take-home
    ax=axes[1,2]
    keys=['power_ppmi','power_pdbp','power_joint']
    names=['Discovery','Replication','Joint']
    vals=[float(np.mean(df[k]>=.8))*100 for k in keys]
    cols=[PALETTE['navy'],PALETTE['teal'],PALETTE['orange']]
    ax.bar(names,vals,color=cols,width=.56)
    for j,(key,v) in enumerate(zip(keys,vals)):
        n=int((df[key]>=.8).sum())
        ax.text(j,v+1,f'{n:,}/{len(df):,}\n{v:.1f}%',ha='center',va='bottom',fontsize=8.5)
    ax.set(ylim=(0,110),ylabel='Proportion of evaluated temporal tQTLs (%)',
           title='f  Fraction with ≥80% estimated power')
    fig.suptitle('CTAR-MPTGA: power among observed temporal tQTLs',fontsize=15,fontweight='bold')
    fig.supxlabel('Observed PPMI overall tQTLs and PDBP matched tests | empirical-Bayes 8-df LRT approximation; not ctAR refitting',
                  fontsize=8.6,color=PALETTE['gray'])
    for s in ('png','pdf','svg'):
        fig.savefig(output/f'Supplementary_Figure_6_CTAR_empirical_power.{s}',dpi=300,bbox_inches='tight', backend=(s if s in ("svg", "pdf") else None))
    plt.close(fig)

def run(args):
    root=Path(args.root)
    path_disc=Path(args.discovery) if args.discovery else root/'qtl/PP/submitted_followup_ctar_mptga/final/significant_tqtl.parquet'
    path_rep=Path(args.replication) if args.replication else root/'qtl/PD/submitted_followup_ctar_mptga/final/all_results.parquet'
    output=Path(args.output) if args.output else root/'manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_empirical_power_v2'
    output.mkdir(parents=True,exist_ok=True)
    for p in (path_disc,path_rep):
        if not p.exists():raise FileNotFoundError(f'Actual Stage-I file not found: {p}')
    disc,meta=sample_discovery(path_disc,args.n_loci,args.seed)
    if args.alpha_disc is not None:
        a=float(args.alpha_disc)
        if not (0<a<1):raise ValueError('--alpha-disc must be in (0,1)')
        if a < meta['max_selected_raw_p']-1e-10:
            raise ValueError('Explicit discovery P cutoff is below max P among selected hits')
        meta['discovery_p_threshold']=a
    joined,repmeta=load_replication(path_rep,disc)
    matched=joined.rep_p.notna()&np.isfinite(joined.rep_lrt)
    joined=joined.loc[matched].reset_index(drop=True)
    print(f"[DATA] {meta['n_discovery_hits']:,} PPMI temporal hits; sampled {len(disc):,}; "
          f"matched PDBP {len(joined):,}; 8-df discovery cutoff {meta['discovery_p_threshold']:.5g}",flush=True)
    if len(joined)<args.min_matched:
        raise ValueError(f'Only {len(joined)} matched PDBP pairs; min {args.min_matched}. Check file naming/IDs.')
    scored,state=posterior_and_power(joined,meta['discovery_p_threshold'],args.alpha_rep,args.info_scale)
    ci=ci_cluster_by_gene(scored,['power_ppmi','power_pdbp','power_joint'],args.bootstraps,args.seed+1)
    summary=make_summary(scored,meta,ci)
    summary['real_inputs']={'discovery':str(path_disc),'replication':str(path_rep)}
    summary['metadata']={k:v for k,v in meta.items() if k!='spec'}
    summary['replication_metadata']={k:v for k,v in repmeta.items() if k!='spec'}
    summary['settings']=vars(args)
    summary['naive_sensitivity']={
        'ppmi_naive_fraction_power_ge80':float(np.mean(scored.naive_power_ppmi>=.8)),
        'pdbp_naive_fraction_power_ge80':float(np.mean(scored.naive_power_pdbp>=.8)),
        'note':'Uncorrected selected-LRT inversion is a sensitivity reference only, not the manuscript estimate.'} 
    scored.to_csv(output/'locus_power_PP_PD.csv',index=False)
    pd.DataFrame({'lambda_rep_grid':state['grid'],'prior_weight':state['weights']}).to_csv(output/'prior_lambda_grid.csv',index=False)
    curves=[]
    for fr in [.25,.4,.5,.6,.75,.85,1.]:
        for cohort in ('PPMI','PDBP'):
            p=power_at_fraction(state,fr,cohort)
            curves.append({'cohort':cohort,'fraction_of_actual_n':fr,'n_total_loci':len(p),
                           'fraction_power_ge80':float(np.mean(p>=.8)),
                           'mean_power':float(np.mean(p))})
    pd.DataFrame(curves).to_csv(output/'power_by_sample_fraction.csv',index=False)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    plot_figure(scored,state,summary,output,args.seed)
    print('[RESULT] Among matched PPMI selected temporal tQTL pairs:')
    for k in ('discovery','replication','both_80'):
        s=summary[k]
        print(f"  {k}: {s['n_above_80']}/{s['n_evaluated']} = {s['fraction']:.2%} "
              f"(gene bootstrap 95% CI {s['gene_bootstrap_ci95'][0]:.2%}–{s['gene_bootstrap_ci95'][1]:.2%})")
    print('[NOTE] Only overall 8-df temporal tQTL. Model-based empirical Bayes, not donor-level refits.')
    print(f'[DONE] {output}')

def main():
    parser=argparse.ArgumentParser(description='Empirical power of real PPMI/PDBP overall temporal tQTLs; never dynamic QTL')
    parser.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    parser.add_argument('--discovery',default=None)
    parser.add_argument('--replication',default=None)
    parser.add_argument('--output',default=None)
    parser.add_argument('--alpha-disc',type=float,default=None,help='Actual raw P-value cutoff, if known; otherwise max selected P')
    parser.add_argument('--alpha-rep',type=float,default=.05)
    parser.add_argument('--n-loci',type=int,default=5000)
    parser.add_argument('--min-matched',type=int,default=1000)
    parser.add_argument('--info-scale',choices=['participants','observations'],default='participants',
                        help='Assumption for cross-cohort NCP scaling; default independent participants')
    parser.add_argument('--bootstraps',type=int,default=500)
    parser.add_argument('--seed',type=int,default=20261030)
    args=parser.parse_args()
    run(args)
if __name__=='__main__':main()
