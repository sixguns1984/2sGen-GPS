#!/usr/bin/env python3
"""AD MMSE Figure 4: Fig2-matched A-tier panel + observed 3-genotype examples.

Strict requirements: frozen Stage-II component, matching observed MMSE,
three sufficiently populated genotype classes, observed-sample MAF,
real PLINK BIM allele pair. No arbitrary absent-genotype trace is drawn.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('cairo')  # Use Cairo backend for high-quality output
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.panel_a_label_utils import select_panel_a_labels, write_label_audit
from common.editable_export import save_figure

ROOT_DEFAULT='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt'
COLORS={0:'#D8A031',1:'#2A6DB3',2:'#D95F02'}
POS='#C96549'; NEG='#39799C'; MULTI='#D432B5'
OUTCOME_NAMES=('mmse','MMSE','mmse_total','MMSE_total','mmse_score',
               'MMSE_score','mmse_total_score','MMSE_total_score')
INTEREST=('PLD2','CLCN7','NDUFA3','CARS2','GDPD5','GSDMB','FAM19A2',
          'SEMA3C','TRPC6','TIRAP','RAB11FIP4','BAK1','TREML4','TFIP11',
          'LY75','SLC12A7','DYNLL1','GBP3','TPM1','TRPC6')


def load(path, columns=None):
    p=Path(path)
    if not p.exists() and p.suffix=='.parquet' and p.with_suffix('.csv').exists():
        p=p.with_suffix('.csv')
    if not p.exists():raise FileNotFoundError(str(p))
    if p.suffix=='.parquet':return pd.read_parquet(p,columns=columns)
    if p.suffix=='.gz' or p.suffix=='.csv':return pd.read_csv(p,usecols=columns,low_memory=False)
    raise ValueError('Unsupported file '+str(p))


def locate_mmse(root, explicit=''):
    b=root/'single_iv_causal'/'ad_mmse'
    options=[Path(explicit)] if explicit else [
        b/'temporal_causal_agecubic_full_followup_fast'/'phenotype_with_covariates.parquet',
        b/'temporal_causal_agecubic_full_followup_fast'/'phenotype_with_covariates.csv',
        b/'temporal_causal'/'phenotype_with_covariates.parquet',
        b/'temporal_causal'/'phenotype_with_covariates.csv',
        b/'standard_final'/'phenotype_with_covariates.parquet',
        b/'standard_final'/'phenotype_with_covariates.csv',
        b/'phenotype_with_covariates.parquet', b/'phenotype_with_covariates.csv',
        b/'phenotype_cov_restored_for_postprocess.csv',
        b/'prep'/'AD_MMSE_phenotype_cov.csv',
    ]
    for p in options:
        if not p.exists():continue
        x=load(p)
        found=[c for c in OUTCOME_NAMES if c in x.columns]
        if not found or not {'cohort','participant_id','time'}.issubset(x):continue
        c=found[0]
        x=x[['cohort','participant_id','time',c]].copy().rename(columns={c:'MMSE'})
        x['cohort']=x.cohort.astype(str)
        x['participant_id']=x.participant_id.astype(str)
        x['time']=pd.to_numeric(x.time,errors='coerce')
        x['MMSE']=pd.to_numeric(x.MMSE,errors='coerce')
        x=x.dropna(subset=['cohort','participant_id','time','MMSE'])
        x=x.drop_duplicates(['cohort','participant_id','time'])
        if not x.empty:
            print('[OBSERVED MMSE]',p, 'rows=',len(x),'cohorts=',x.cohort.value_counts().to_dict(),flush=True)
            return x
    raise FileNotFoundError('Could not find measured MMSE with cohort/participant_id/time under '+str(b)+
                            '. Pass --phenotype-mmse /path/to/exact/MMSE.csv')


def load_bim(root,cohort,explicit='',snps=None):
    b=root/'single_iv_causal'/'ad_mmse'
    if explicit:
        p=Path(explicit)
        if not p.name.endswith('.bim'):p=Path(str(p)+'.bim')
        candidates=[p]
    else:
        candidates=[b/'genotype'/'single_iv_snps.bim',
                    b/'prep'/f'{cohort}_harmonized_A1.bim',
                    b/'prep'/f'{cohort}_common_IVs.bim',
                    b/'prep'/'MAP_ADNI_harmonized_merged.bim']
    for p in candidates:
        if not p.exists():continue
        m={}
        wanted=set(map(str,snps)) if snps is not None else None
        with p.open() as f:
            for line in f:
                fields=line.strip().split()
                if len(fields)<6:continue
                if wanted is None or fields[1] in wanted:
                    m[fields[1]]=(fields[4],fields[5])
                    if wanted and len(m)==len(wanted):break
        if m:
            print('[BIM ALLELES]',p,'matched=',len(m),flush=True)
            return m,p
    raise FileNotFoundError('No matching SNP alleles in an existing BIM. Tried: '+', '.join(map(str,candidates))+
                            '. Provide --bim /real/path/to/harmonized.bim')


def read_component(root,row,cohort,observed,max_years):
    aid=str(row['artifact_id'])
    p=root/'single_iv_causal'/'ad_mmse'/'single_iv_genetic_component'/f'{aid}.parquet'
    x=load(p,columns=['cohort','participant_id','time','iv_dosage','grex_single_iv'])
    x=x[x['cohort'].astype(str).eq(cohort)].copy()
    for q in (x,):
        q['participant_id']=q.participant_id.astype(str)
        q['time']=pd.to_numeric(q.time,errors='coerce')
    x['iv_dosage']=pd.to_numeric(x.iv_dosage,errors='coerce')
    x['grex_single_iv']=pd.to_numeric(x.grex_single_iv,errors='coerce')
    x=x[(x.time>=0)&(x.time<=max_years)].copy()
    y=observed.loc[observed.cohort.eq(cohort)].copy()
    z=x.merge(y,on=['cohort','participant_id','time'],how='inner',validate='many_to_one')
    z=z.dropna(subset=['iv_dosage','grex_single_iv','MMSE','time'])
    dose=np.rint(z.iv_dosage.to_numpy(float))
    z=z.loc[np.isclose(z.iv_dosage.to_numpy(float),dose,atol=1e-6,rtol=0)&np.isin(dose,[0,1,2])].copy()
    z['genotype']=np.rint(z.iv_dosage).astype(int)
    z=z.drop_duplicates(['cohort','participant_id','time','genotype'])
    if (z.groupby('participant_id')['genotype'].nunique()>1).any():
        raise ValueError('Frozen IV dosage changes genotype group across visits')
    return z


def genotype_stats(z):
    counts={g:int(z.loc[z.genotype.eq(g),'participant_id'].nunique()) for g in range(3)}
    den=sum(counts.values())
    maf=(min((counts[1]+2*counts[2])/(2*den),
             1-(counts[1]+2*counts[2])/(2*den)) if den else np.nan)
    bins={g:int(z.loc[z.genotype.eq(g),'time'].round().nunique()) for g in range(3)}
    # Only exact 0/1/2 dosage hard calls are retained; this is the plotted subset MAF.
    return {**{f'n{g}':counts[g] for g in range(3)},
            **{f'timepoints{g}':bins[g] for g in range(3)},
            'n_total':den,'maf_plotted':maf,'n_observations':len(z)}


def rank_candidates(df,root,cohort,observed,max_years,min_genotype_n,min_timepoints):
    audit=[]
    pref={g:i for i,g in enumerate(dict.fromkeys(INTEREST))}
    for _,r in df.iterrows():
        d={'gene':r.gene,'lead_snp':r.lead_snp,'artifact_id':r.artifact_id,
           'MAP_pvalue':r.MAP_pvalue,'ADNI_pvalue':r.ADNI_pvalue,
           'MAP_coef_sum':r.MAP_coef_sum,'ADNI_coef_sum':r.ADNI_coef_sum,
           'tier':r.descriptive_evidence_tier}
        try:
            z=read_component(root,r,cohort,observed,max_years)
            d.update(genotype_stats(z))
            d['eligible']=(min(d['n0'],d['n1'],d['n2'])>=min_genotype_n and
                           min(d['timepoints0'],d['timepoints1'],d['timepoints2'])>=min_timepoints)
            d['failure']=''
        except Exception as e:
            d.update({'eligible':False,'n0':0,'n1':0,'n2':0,'n_total':0,'maf_plotted':np.nan,
                      'failure':str(e)[:300]})
        d['priority']=pref.get(str(r.gene),len(pref)+1)
        audit.append(d)
    audit=pd.DataFrame(audit)
    audit['balanced_n']=audit[['n0','n1','n2']].min(axis=1)
    # Interest preference first, then rank by replication P and actual third genotype support.
    audit=audit.sort_values(['eligible','priority','ADNI_pvalue','balanced_n'],
                            ascending=[False,True,True,False]).reset_index(drop=True)
    return audit


def select_stories(df,audit,requested):
    eligible=audit[audit.eligible.eq(True)]
    if requested:
        out=[]
        for token in requested:
            name=token.split(':',1)[0]
            snp=token.split(':',1)[1] if ':' in token else ''
            s=eligible[eligible.gene.eq(name)]
            if snp:s=s[s.lead_snp.eq(snp)]
            if s.empty:
                q=audit[audit.gene.eq(name)]
                sample=q[['gene','lead_snp','n0','n1','n2','failure']].head(5).to_dict('records')
                raise ValueError(f'{token}: no eligible 3-genotype example. Audit: {sample}. '
                                 'Choose a gene from Figure4_example_genotype_audit.csv')
            out.append(s.iloc[0]);
        ans=pd.DataFrame(out)
    else:
        ans=eligible.drop_duplicates('gene').head(2).copy()
    if len(ans)!=2 or ans.gene.nunique()!=2:
        raise ValueError('Need two distinct eligible genes with three well-populated genotypes. '
                         'See Figure4_example_genotype_audit.csv, or lower --min-genotype-n cautiously.')
    return ans


def draw_panel_a(ax,df):
    d=df.copy().reset_index(drop=True)
    if 'panel_a_label' not in d:
        raise RuntimeError('AD Panel a was not passed through the shared label selector')
    y=d['neglog10p']
    color=np.where(d.MAP_coef_sum>=0,POS,NEG)
    # Outline follows the actual table's BH-adjusted multi-IV result, not a substring of the tier name.
    edge=np.where(d.panel_a_multi_iv_bh_fdr05,MULTI,'#2E3439')
    for i in range(len(d)):
        ax.vlines(i,0,y.iloc[i],color=color[i],alpha=.33,lw=.8,zorder=1)
        ax.scatter(i,y.iloc[i],s=46,facecolor=color[i],edgecolor=edge[i],lw=1.1,zorder=3)
    ax.set_xlim(-.8,len(d)-.15)
    ax.set_ylim(0,max(float(y.max())*1.14,2))
    ax.set_xticks([])  # Unified PD/AD convention: labels are next to selected points only.
    ax.set_ylabel(r'$-\log_{10}(P_{ADNI})$',fontsize=10)
    ax.set_title('a  A-tier MMSE-associated IV–genes',loc='left',fontsize=11.2,weight='bold',pad=10)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#E1E5E9',lw=.7)
    ax.tick_params(axis='x',length=0)
    legend=[Line2D([0],[0],marker='o',linestyle='',markerfacecolor=POS,markeredgecolor='#333',label='Positive summed-lag coefficient',markersize=6),
            Line2D([0],[0],marker='o',linestyle='',markerfacecolor=NEG,markeredgecolor='#333',label='Negative summed-lag coefficient',markersize=6),
            Line2D([0],[0],marker='o',linestyle='',markerfacecolor='white',markeredgecolor=MULTI,label='Multi-IV BH-FDR < 0.05',markersize=6)]
    ax.legend(handles=legend,frameon=False,ncol=2,loc='lower left',bbox_to_anchor=(0,1.06),fontsize=7.5,
              columnspacing=.9,handlelength=1.0)
    for _,r in d[d['panel_a_label']].iterrows():
        i=int(r.plot_x)
        ax.annotate(str(r.gene),(i,float(r.neglog10p)),xytext=(1,4),textcoords='offset points',
                    ha='left',rotation=58,fontsize=6.8)


def plot_story(fig,spec,letter,r,z,alleles,cohort,max_years):
    L=int(r.selected_lag_order)
    gs=GridSpecFromSubplotSpec(3,1,subplot_spec=spec,height_ratios=[1.78,1.17,.55],hspace=.31)
    ax=fig.add_subplot(gs[0]);gx=fig.add_subplot(gs[1],sharex=ax);lx=fig.add_subplot(gs[2])
    gstats=genotype_stats(z)
    a1,a2=alleles
    ax.set_title(f'{letter}  {r.gene} | {r.lead_snp} | MMSE (L={L})',loc='left',fontsize=9.4,pad=17)
    ax.text(0,1.025,f'{cohort} plotted-sample MAF={gstats["maf_plotted"]:.3f} | PLINK A1/A2={a1}/{a2} (Stage-II dosage orientation not assumed)',
            transform=ax.transAxes,ha='left',va='bottom',fontsize=7.4)
    for g in (0,1,2):
        d=z[z.genotype.eq(g)].copy(); cc=COLORS[g]
        d['time_bin']=np.round(d.time).astype(int)
        grp=d.groupby('time_bin',as_index=False).agg(mmse=('MMSE','mean'),sd=('MMSE','std'),n=('participant_id','nunique'))
        grp=grp[grp.n>=3]
        if not grp.empty:
            ax.plot(grp.time_bin,grp.mmse,color=cc,lw=1.6,marker='o',ms=3.5,label=f'Dosage {g} (n={gstats[f"n{g}"]})')
            se=grp.sd.fillna(0).to_numpy()/np.sqrt(grp.n.to_numpy())
            ax.fill_between(grp.time_bin.to_numpy(float),grp.mmse.to_numpy()-se,
                            grp.mmse.to_numpy()+se,color=cc,alpha=.12,lw=0)
        pred=d.groupby('time',as_index=False)['grex_single_iv'].mean().sort_values('time')
        # Display only observed frozen predicted GReX values; no refit, no extrapolation.
        gx.plot(pred.time,pred.grex_single_iv,color=cc,lw=1.6)
        ix=np.unique(np.linspace(0,len(pred)-1,min(9,len(pred))).astype(int))
        gx.scatter(pred.time.iloc[ix],pred.grex_single_iv.iloc[ix],color='white',edgecolor=cc,s=14,zorder=3,lw=.85)
    ax.legend(frameon=False,fontsize=6.7,loc='upper left',ncol=3,handlelength=1.1,columnspacing=.7)
    ax.set_ylabel('MMSE',fontsize=8.3)
    gx.set_ylabel('Predicted\ntGReX(t)',fontsize=8.0)
    gx.set_xlabel('Years since baseline',fontsize=8.0)
    ax.set_xlim(0,max_years)
    s1=float(r.MAP_coef_sum);s2=float(r.ADNI_coef_sum)
    gx.text(.99,.98,f'Σβ: MAP {s1:+.2g}; ADNI {s2:+.2g}',transform=gx.transAxes,
            ha='right',va='top',fontsize=7,color='#526B83')
    for q in (ax,gx):
        q.spines[['top','right']].set_visible(False)
        q.grid(axis='y',color='#E6EAEE',lw=.65)
        q.tick_params(labelsize=7)
    xx=[];yy=[]
    for k in range(1,L+1):
        val=pd.to_numeric(pd.Series([r.get(f'MAP_coef_x_lag{k}')]),errors='coerce').iloc[0]
        if pd.notna(val):xx.append(k);yy.append(float(val))
    for x,y in zip(xx,yy):
        lx.vlines(x,0,y,lw=1.2,color='#C9D2D9')
        lx.scatter(x,y,color='#BD5670' if y>=0 else '#3884A8',s=22,zorder=3)
    lx.axhline(0,color='#444',lw=.7)
    lx.set_xticks(xx)
    lx.set_xlim(.5,max(L,1)+.5)
    lx.set_ylabel('Lag β',fontsize=7.5)
    lx.set_xlabel('Previous visits (lag)',fontsize=7.5)
    lx.spines[['top','right']].set_visible(False)
    lx.grid(axis='y',color='#E6EAEE',lw=.6)
    lx.tick_params(labelsize=7)
    return gstats


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',default=ROOT_DEFAULT)
    ap.add_argument('--publication-table',default='')
    ap.add_argument('--outdir',default='')
    ap.add_argument('--top-n',type=int,default=30)
    ap.add_argument('--top-labels-per-phenotype',type=int,default=8)
    ap.add_argument('--example-cohort',choices=['MAP','ADNI'],default='MAP')
    ap.add_argument('--story-examples',nargs='*',default=[])
    ap.add_argument('--max-years',type=float,default=10.0,
                    help='Figure-only restriction of the observed time axis; does not change Stage II fit')
    ap.add_argument('--min-genotype-n',type=int,default=10)
    ap.add_argument('--min-timepoints',type=int,default=3)
    ap.add_argument('--phenotype-mmse',default='')
    ap.add_argument('--bim',default='',help='Exact harmonized genotype BIM; auto-detected by default')
    ap.add_argument('--panel-a-only',action='store_true')
    args=ap.parse_args()
    if args.min_genotype_n<3 or args.max_years<=0 or args.min_timepoints<1:
        ap.error('min genotype n must be >=3, max-years positive, min-timepoints >=1')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','font.size':9})
    root=Path(args.root)
    table=Path(args.publication_table) if args.publication_table else root/'single_iv_causal'/'ad_mmse'/'standard_final'/'posthoc_evidence'/'Supplement_AD_MMSE_IV_gene_publication_POWER80.csv'
    out=Path(args.outdir) if args.outdir else root/'manuscript_figures'/'tqtl_benchmark_v4'/'figure4_ad_mmse'
    out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(table,low_memory=False)
    req={'gene','lead_snp','artifact_id','descriptive_evidence_tier','MAP_coef_sum','ADNI_coef_sum','MAP_pvalue','ADNI_pvalue','selected_lag_order'}
    if not req.issubset(df.columns):raise KeyError('Missing columns in AD table: '+str(sorted(req-set(df.columns))))
    df=df[df.descriptive_evidence_tier.astype(str).str.startswith('A_')].copy()
    df=df.dropna(subset=['MAP_coef_sum','ADNI_coef_sum','ADNI_pvalue'])
    df=df[df.MAP_coef_sum*df.ADNI_coef_sum>0]
    df=df.sort_values(['ADNI_pvalue','MAP_pvalue']).drop_duplicates(['gene','lead_snp'])
    if df.empty:raise ValueError('No A-tier, coef_sum-concordant rows in AD MMSE POWER80 input')
    panel=df.drop_duplicates('gene').head(args.top_n).copy()
    # The AD panel is un-clipped. Use the exact ADNI P value plotted below.
    panel=panel.reset_index(drop=True)
    panel['plot_x']=np.arange(len(panel))
    panel['neglog10p']=-np.log10(np.maximum(pd.to_numeric(panel.ADNI_pvalue, errors='coerce'),1e-300))
    panel=select_panel_a_labels(
        panel, gene_col='gene', x_col='plot_x', y_col='neglog10p',
        source_tier_col='descriptive_evidence_tier',
        top_n_per_group=args.top_labels_per_phenotype,
    )
    write_label_audit(panel, out/'Figure4A_label_audit.csv',
                      extra_cols=('lead_snp','ADNI_pvalue','MAP_pvalue','multiIV_p_q_BH'))
    # Export the same plotted A-tier rows with the ORIGINAL source tier, unchanged.
    panel.to_csv(out/'Figure4A_selected_top_genes.csv',index=False)
    figA,ax=plt.subplots(figsize=(10.8,5.8))
    draw_panel_a(ax,panel)
    figA.subplots_adjust(left=.09,right=.99,top=.77,bottom=.25)
    for ext in ('pdf','svg','png'):
        save_figure(figA, out/f'Figure4A_AD_A_tier_top{args.top_n}.{ext}',bbox_inches='tight',dpi=350 if ext=='png' else None)
    plt.close(figA)
    if args.panel_a_only:
        print('[DONE] Panel A only',out,flush=True);return
    obs=locate_mmse(root,args.phenotype_mmse)
    audit=rank_candidates(df,root,args.example_cohort,obs,args.max_years,args.min_genotype_n,args.min_timepoints)
    audit.to_csv(out/'Figure4_example_genotype_audit.csv',index=False)
    print('[EXAMPLE AUDIT] three-genotype eligible=',int(audit.eligible.sum()),'of',len(audit),flush=True)
    chosen=select_stories(df,audit,args.story_examples)
    bim,which=load_bim(root,args.example_cohort,args.bim,chosen.lead_snp.tolist())
    missing=set(chosen.lead_snp)-set(bim)
    if missing:raise KeyError(f'Alleles not in selected BIM {which}: {sorted(missing)}. Specify --bim with matching harmonized SNP file.')
    # Rejoin ranked details with full source table (same artefact prevents false SNP pairing)
    stories=chosen[['artifact_id']].merge(df,on='artifact_id',how='left',validate='one_to_one')
    fig=plt.figure(figsize=(12.2,8.0))
    gs=GridSpec(100,100,figure=fig,left=.055,right=.985,top=.91,bottom=.13,wspace=0,hspace=0)
    ax=fig.add_subplot(gs[9:86,3:51]);draw_panel_a(ax,panel)
    report=[]
    for idx,(_,r) in enumerate(stories.iterrows()):
        z=read_component(root,r,args.example_cohort,obs,args.max_years)
        s=genotype_stats(z)
        if min(s[f'n{k}'] for k in range(3))<args.min_genotype_n:
            raise ValueError('Genotype counts changed since audit for '+str(r.gene))
        spec=gs[6:44,64:99] if idx==0 else gs[56:94,64:99]
        plot_story(fig,spec,'b' if idx==0 else 'c',r,z,bim[str(r.lead_snp)],args.example_cohort,args.max_years)
        report.append({'gene':r.gene,'lead_snp':r.lead_snp,'artifact_id':r.artifact_id,
                       'MAP_coef_sum':r.MAP_coef_sum,'ADNI_coef_sum':r.ADNI_coef_sum,
                       'MAP_pvalue':r.MAP_pvalue,'ADNI_pvalue':r.ADNI_pvalue,
                       'cohort':args.example_cohort,'bim_source':str(which),
                       'A1':bim[str(r.lead_snp)][0],'A2':bim[str(r.lead_snp)][1],
                       **s})
    pd.DataFrame(report).to_csv(out/'selected_story_examples.csv',index=False)
    fig.text(.06,.06,'Summed-lag coefficient: aggregate visit-based association, not a fixed-year causal effect. '
             'MAF estimated from plotted cohort genotype-dosage groups.',fontsize=8,color='#64758A')
    for ext in ('pdf','svg','png'):
        save_figure(fig, out/f'Figure4_AD_MMSE_temporal_association_main.{ext}',bbox_inches='tight',dpi=350 if ext=='png' else None)
    plt.close(fig)
    print('[DONE]',out,flush=True)
    print(pd.DataFrame(report)[['gene','lead_snp','n0','n1','n2','maf_plotted','A1','A2']].to_string(index=False),flush=True)

if __name__=='__main__':main()
