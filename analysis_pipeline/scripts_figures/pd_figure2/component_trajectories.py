"""Figure 2 case trajectories from EXACT frozen Stage II single-IV component files.

No PLINK re-export, no independent genotype recoding, no manual polynomial
reconstruction. `grex_single_iv` and `iv_dosage` are the values actually analyzed.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec

from helpers_pd_figure2 import PHENO_LABEL, OUTCOME_COL, GENO_COLORS


def _read(path):
    p=Path(path)
    if p.exists():
        if p.suffix.lower()=='.parquet':
            try: return pd.read_parquet(p)
            except ImportError as e:
                raise ImportError('Reading Stage II Parquet requires pyarrow or fastparquet') from e
        return pd.read_csv(p, low_memory=False)
    # CSV alternative is for test/audit exports; never silently use a different analysis directory.
    csv=p.with_suffix('.csv')
    if csv.exists(): return pd.read_csv(csv, low_memory=False)
    raise FileNotFoundError(f'Missing Stage II file: {p}; alternate {csv}')


def get_artifact_id(row):
    for c in ('stage1_artifact_id','PPMI_artifact_id','PD_artifact_id','artifact_id'):
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip() not in ('','nan'):
            return str(row[c]).strip()
    raise KeyError('Publication row lacks Stage II artifact identifier')


def _find_outcome_column(df, phenotype):
    primary=OUTCOME_COL[phenotype]
    aliases={
        'updrs1':['mds_updrs_part_i_summary_score','mds_updrs_part_i_total_score','updrs1'],
        'updrs2':['mds_updrs_part_ii_summary_score','mds_updrs_part_ii_total_score','updrs2'],
        'updrs3':['mds_updrs_part_iii_summary_score','mds_updrs_part_iii_total_score','updrs3'],
        'moca':['moca_total_score','moca_total','moca','moca_score','moca_total_score_corrected',
                'moca_score_total','moca_total_corrected','moca_total_30'],
    }
    names=list(dict.fromkeys([primary]+aliases[phenotype]))
    for name in names:
        if name in df.columns: return name
    raise KeyError(f'{phenotype}: outcome column absent; tried {names}; available={list(df.columns)[:60]}')


def _candidate_phenotype_paths(base):
    """Prefer Stage II analysis inputs with measured outcomes, not Step-05 metadata caches."""
    exact=[base/'temporal_causal_agecubic_full_followup_fast'/'phenotype_with_covariates.parquet',
           base/'temporal_causal_agecubic_full_followup_fast'/'phenotype_with_covariates.csv',
           base/'temporal_causal'/'phenotype_with_covariates.parquet',
           base/'temporal_causal'/'phenotype_with_covariates.csv',
           base/'phenotype_with_covariates.parquet',
           base/'phenotype_with_covariates.csv']
    discovered=sorted(list(base.glob('temporal_causal*/phenotype_with_covariates.parquet'))+
                      list(base.glob('temporal_causal*/phenotype_with_covariates.csv')))
    raw=[base/'phenotype_cov_restored_for_postprocess.csv',base/'phenotype_clean.parquet',
         base/'phenotype_clean.csv']
    seen=set()
    for p in exact+discovered+raw:
        if p not in seen:
            seen.add(p)
            if p.is_file(): yield p


def _resolve_phenotype_cov(base, phenotype, override=None):
    if override:
        paths=[Path(override)]
        if not paths[0].is_file():raise FileNotFoundError(f'Explicit phenotype file missing: {paths[0]}')
    else:
        paths=list(_candidate_phenotype_paths(base))
    rejected=[]
    for p in paths:
        try:
            x=_read(p)
            col=_find_outcome_column(x,phenotype)
            if not {'cohort','participant_id','time'}.issubset(x.columns):
                raise ValueError('missing cohort/participant_id/time visit key')
            if not pd.to_numeric(x[col],errors='coerce').notna().any():
                raise ValueError('all outcome values missing')
            print(f'[PHENOTYPE SOURCE] {phenotype}: {p} (column={col})',flush=True)
            return p,x,col
        except (KeyError,ValueError) as e:
            rejected.append(f'{p}: {e}')
    raise FileNotFoundError(
        f'{phenotype}: no Stage II input with real outcome found under {base}. '
        'The Step-05 phenotype_clean.parquet may contain metadata only. '
        'Pass --phenotype-moca (or corresponding --phenotype-...) to specify '
        'the exact Stage II clinical input file. Inspected candidates:\n'+'\n'.join(rejected)
    )


def load_example_from_stage2(root, row, cohort='PP', min_participants=8, phenotype_override=None):
    phenotype=str(row['phenotype']).lower(); gene=str(row['gene']); snp=str(row['lead_snp']); aid=get_artifact_id(row)
    base=Path(root)/'single_iv_causal'/phenotype
    comp=base/'single_iv_genetic_component'/f'{aid}.parquet'
    comp_df=_read(comp)
    required={'participant_id','cohort','time','grex_single_iv','iv_dosage'}
    missing=required-set(comp_df.columns)
    if missing: raise KeyError(f'{comp}: missing {sorted(missing)}')
    if {'gene','lead_snp'}.issubset(comp_df.columns):
        pairs=comp_df[['gene','lead_snp']].drop_duplicates().astype(str)
        if len(pairs)!=1 or pairs.iloc[0].to_list()!=[gene,snp]:
            raise ValueError(f'Frozen component {comp} is not {gene}/{snp}: {pairs.to_dict("records")}')
    covfile,outcome_df,outcome_col=_resolve_phenotype_cov(base,phenotype,override=phenotype_override)
    req_cov={'cohort','participant_id','time',outcome_col}
    if not req_cov.issubset(outcome_df.columns):
        raise KeyError(f'{covfile}: missing {req_cov-set(outcome_df.columns)}')
    lhs=outcome_df[['cohort','participant_id','time',outcome_col]].copy()
    rhs=comp_df[['cohort','participant_id','time','iv_dosage','grex_single_iv']].copy()
    for df in (lhs,rhs):
        df['cohort']=df['cohort'].astype(str)
        df['participant_id']=df['participant_id'].astype(str)
        df['time']=pd.to_numeric(df['time'],errors='coerce')
    if cohort!='ALL':
        lhs=lhs[lhs['cohort'].eq(cohort)].copy()
        rhs=rhs[rhs['cohort'].eq(cohort)].copy()
    keys=['cohort','participant_id','time']
    for name,df in [('phenotype',lhs),('frozen component',rhs)]:
        dup=df.duplicated(keys,keep=False)
        if dup.any():
            raise ValueError(f'{name}: duplicated cohort/participant/time keys; {df.loc[dup,keys].head().to_dict("records")}')
    z=lhs.merge(rhs,on=keys,how='inner',validate='one_to_one')
    nmerge=len(z)
    for c in ('iv_dosage','grex_single_iv',outcome_col): z[c]=pd.to_numeric(z[c],errors='coerce')
    z=z.dropna(subset=['time','iv_dosage','grex_single_iv',outcome_col]).copy()
    # Missing genotype dosage is intentionally not rounded into a genotype group.
    hardcall=z['iv_dosage'].between(0,2)&np.isclose(z['iv_dosage'],np.rint(z['iv_dosage']),atol=1e-6)
    z=z.loc[hardcall].copy()
    z['genotype']=np.rint(z['iv_dosage']).astype(int)
    if z['participant_id'].nunique()<min_participants:
        raise ValueError(f'{gene} {snp} {phenotype} {cohort}: only {z.participant_id.nunique()} participants after merge. '
                         f'Check that phenotype and frozen component files belong to the same Stage II version.')
    # Freeze an individual's dosage across visits.
    gtconf=z.groupby(['cohort','participant_id'])['genotype'].nunique()
    if (gtconf>1).any():raise ValueError(f'{gene} {snp}: inconsistent IV dosage across same participant visits')
    z=z.rename(columns={outcome_col:'outcome','time':'time_years'})
    metadata=dict(phenotype=phenotype,gene=gene,lead_snp=snp,artifact_id=aid,
                  cohort=cohort,component=str(comp),phenotype_input=str(covfile),
                  n_intersected_rows=int(nmerge),n_plot_rows=len(z),
                  n_participants=int(z.participant_id.nunique()),
                  dosage_orientation='exact Stage II iv_dosage (allele not inferred)',
                  genotype_counts={int(g):int(n) for g,n in z.groupby('genotype').participant_id.nunique().items()},
                  time_min=float(z.time_years.min()),time_max=float(z.time_years.max()))
    return z,metadata


def _summary_binned(z, step):
    t=z.copy()
    t['time_bin']=np.floor(t['time_years']/step+0.5)*step
    g=t.groupby(['genotype','time_bin'],observed=True)['outcome']
    d=g.agg(mean='mean',n='size',sd='std').reset_index()
    d['se']=d['sd']/np.sqrt(d['n'])
    return d


def plotted_donor_genotype_summary(z):
    """Genotype frequency in unique donor IDs actually plotted, not visit rows.

    The minor allele frequency is orientation invariant: dosage-0/1/2 may refer
    to either genetic allele. A1/A2 from PLINK BIM is reported separately.
    """
    required={'cohort','participant_id','genotype'}
    missing=required-set(z.columns)
    if missing:
        raise KeyError(f'Plotted donor frequency requires {sorted(missing)}')
    donors=z[['cohort','participant_id','genotype']].drop_duplicates().copy()
    conflict=donors.groupby(['cohort','participant_id'])['genotype'].nunique()
    if (conflict>1).any():
        raise ValueError('Conflicting genotype dosage for one participant across plotted visits')
    donors['genotype']=pd.to_numeric(donors['genotype'],errors='raise')
    if not donors['genotype'].isin([0,1,2]).all():
        raise ValueError('Only exact dosage 0/1/2 is permitted for plotted MAF')
    counts={j:int((donors.genotype==j).sum()) for j in (0,1,2)}
    n=sum(counts.values())
    if n==0:
        raise ValueError('No plotted donors for MAF')
    freq=(counts[1]+2*counts[2])/(2*n)
    return dict(n0=counts[0],n1=counts[1],n2=counts[2],
                n_donors=n,maf_plotted=min(freq,1-freq))


def plot_example(fig,spec,row,z,letter,year_bin=1.0,show_lag=True,
                 cohort_label="PPMI", bim_alleles=None):
    # 3 panels for L>1; otherwise 2 panels. Bins represent observed outcome means,
    # predicted tGReX curve fitted *to the cached predictions* (never reencoded).
    lag=int(row['selected_lag_order'])
    nrows=3 if show_lag and lag>1 else 2
    heights=[1.62,1.08,0.58] if nrows==3 else [1.70,1.08]
    inner=GridSpecFromSubplotSpec(nrows,1,subplot_spec=spec,height_ratios=heights,hspace=.28)
    top=fig.add_subplot(inner[0])
    bottom=fig.add_subplot(inner[1],sharex=top)
    lagax=fig.add_subplot(inner[2]) if nrows==3 else None
    groups=sorted(z.genotype.unique())
    info=plotted_donor_genotype_summary(z)
    counts={g:info[f'n{g}'] for g in (0,1,2)}
    first=str(row['gene']);phen=str(row['phenotype']);snp=str(row['lead_snp'])
    if bim_alleles is None or len(bim_alleles)!=2:
        raise ValueError(f'{first}/{snp}: require verified BIM A1/A2 for both story panels')
    a1,a2=bim_alleles
    title=f'{letter}  {first}  |  {snp}  |  {PHENO_LABEL[phen]} (L={lag})'
    # Single consistent annotation logic (including story-csv cases), replacing
    # the old conditional row.get("maf_PP") that silently hid b/c SNP metadata.
    meta=(f'{cohort_label} plotted-sample MAF={info["maf_plotted"]:.3f} | '
          f'PLINK A1/A2={a1}/{a2}\n'
          '(Stage-II dosage orientation not assumed)')
    top.set_title(title,loc='left',weight='bold',fontsize=8.4,pad=11)
    top.text(0.0,.970,meta,transform=top.transAxes,
             ha='left',va='bottom',fontsize=6.1,color='#3F4C5A',linespacing=1.0)
    binned=_summary_binned(z,year_bin)
    # Plotting bins are rounded only for summaries; never extend x beyond measured outcome follow-up.
    observed_t_min=float(z.time_years.min());observed_t_max=float(z.time_years.max())
    for g in groups:
        col=GENO_COLORS[g]
        q=binned[binned.genotype.eq(g)]
        # omit bins with very few participants to avoid unstable connected traces
        q=q[(q.n>=3)&(q.time_bin>=observed_t_min-1e-8)&(q.time_bin<=observed_t_max+1e-8)]
        if not q.empty:
            top.plot(q.time_bin,q['mean'],color=col,lw=1.5,marker='o',ms=3, label=f'Dosage {g} (n={counts[g]})')
            top.fill_between(q.time_bin.to_numpy(float),(q['mean']-q.se.fillna(0)).to_numpy(float),
                             (q['mean']+q.se.fillna(0)).to_numpy(float),color=col,alpha=.12,lw=0)
        zz=z[z.genotype.eq(g)]
        if counts[g]<3:
            # One participant cannot support a reliable genotype-stratified trend.
            print(f'[SPARSE GENOTYPE] {first}/{snp}: dosage {g} has n={counts[g]}; '
                  'hide its tGReX curve, retain in provenance',flush=True)
            continue
        pred=zz.groupby('time_years',as_index=False)['grex_single_iv'].mean().sort_values('time_years')
        xp=pred.time_years.to_numpy(float); yp=pred.grex_single_iv.to_numpy(float)
        if len(np.unique(xp))>=4:
            # polynomial visualization of the actual cached frozen tGReX values
            t0=float(np.mean(xp)); sc=max(float(np.std(xp)),1e-6)
            v=(xp-t0)/sc; co=np.polyfit(v,yp,3)
            grid=np.linspace(xp.min(),xp.max(),180)
            fit=np.polyval(co,(grid-t0)/sc)
            err=float(np.max(np.abs(np.polyval(co,v)-yp)))
            if err>max(1e-5,1e-5*float(np.max(np.abs(yp)))):
                # If a non-cubic prediction was cached, preserve original points.
                bottom.plot(xp,yp,color=col,lw=1.4)
            else:
                bottom.plot(grid,fit,color=col,lw=1.5)
        elif len(xp)>1:
            bottom.plot(xp,yp,color=col,lw=1.4)
        if len(xp):
            qidx=np.unique(np.linspace(0,len(xp)-1,min(8,len(xp))).astype(int))
            bottom.scatter(xp[qidx],yp[qidx],color=col,facecolors='white',s=11,zorder=3,lw=.8)
    top.set_xlim(observed_t_min,observed_t_max)
    bottom.set_xlim(observed_t_min,observed_t_max)
    # Compact aggregate lag annotation from the selected model; this is NOT a total causal effect.
    pp_sum=pd.to_numeric(pd.Series([row.get('PPMI_coef_sum')]),errors='coerce').iloc[0]
    rep_sum=pd.to_numeric(pd.Series([row.get('PD_PLUS_SU_coef_sum')]),errors='coerce').iloc[0]
    if np.isfinite(pp_sum) and np.isfinite(rep_sum):
        bottom.text(.98,.93,f'Σβ: PPMI {pp_sum:+.2g}; PD+SU {rep_sum:+.2g}',
                    transform=bottom.transAxes,ha='right',va='top',fontsize=6.2,color='#404A59')
    top.set_ylabel(PHENO_LABEL[phen],fontsize=7.8)
    bottom.set_ylabel('Predicted\ntGReX(t)',fontsize=7.6)
    bottom.set_xlabel('Years since baseline',fontsize=7.6)
    for ax in (top,bottom):
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=6.8)
        ax.grid(axis='y',color='#E6EAEE',lw=.65)
    top.legend(frameon=False,fontsize=5.8,loc='upper left',ncol=min(len(groups),3),handlelength=0.95,columnspacing=0.8,handletextpad=0.33)
    sparse=[f'dosage {g}: n={counts[g]}' for g in groups if counts[g]<3]
    if sparse:
        bottom.text(.98,.96,'Not shown (<3 participants): '+', '.join(sparse),
                    transform=bottom.transAxes,ha='right',va='top',fontsize=6.2,
                    color='#9B5460')
    # Clearly distinguish discovery vs external replication associations.
    p=pd.to_numeric(pd.Series([row.get('selected_replication_pvalue')]),errors='coerce').iloc[0]
    if np.isfinite(p):
        top.text(.98,.02,f'Replication P={p:.2g}',ha='right',va='bottom',transform=top.transAxes,fontsize=6.8)
    if lagax is not None:
        lags=[]
        for k in range(1,lag+1):
            v=pd.to_numeric(pd.Series([row.get(f'PPMI_coef_x_lag{k}')]),errors='coerce').iloc[0]
            if np.isfinite(v):lags.append((k,float(v)))
        if lags:
            for k,v in lags:
                lagax.vlines(k,0,v,color='#C6CDD4',lw=1.2)
                lagax.scatter(k,v,c='#B95569' if v>=0 else '#3884A8',s=22,zorder=3)
            lagax.axhline(0,color='#555',lw=.7)
            lagax.set_xticks([x for x,_ in lags])
            lagax.set_ylabel('Lag β',fontsize=7.1)
            lagax.set_xlabel('Previous visits (lag)',fontsize=7.1)
            lagax.grid(axis='y',color='#EEF0F2',lw=.5)
            lagax.tick_params(labelsize=6.6)
            lagax.spines[['top','right']].set_visible(False)
        else:lagax.set_axis_off()
    return fig
