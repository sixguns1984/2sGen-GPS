from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec
from helpers_ad_figure4 import OUTCOME_ALIASES, GENO_COLORS


def _read(path):
    p=Path(path)
    if p.exists():
        if p.suffix.lower()=='.parquet':
            try:return pd.read_parquet(p)
            except ImportError as e: raise ImportError('Reading Parquet requires pyarrow or fastparquet') from e
        return pd.read_csv(p,low_memory=False)
    csv=p.with_suffix('.csv')
    if csv.exists(): return pd.read_csv(csv,low_memory=False)
    raise FileNotFoundError(f'Missing Stage II file: {p}; alternate {csv}')


def _find_outcome_column(df):
    for nm in OUTCOME_ALIASES:
        if nm in df.columns: return nm
    raise KeyError(f'MMSE outcome column absent; tried {OUTCOME_ALIASES}; available={list(df.columns)[:80]}')


def _candidate_phenotype_paths(base):
    exact=[
        base/'temporal_causal_agecubic_full_followup_fast'/'phenotype_with_covariates.parquet',
        base/'temporal_causal_agecubic_full_followup_fast'/'phenotype_with_covariates.csv',
        base/'temporal_causal'/'phenotype_with_covariates.parquet',
        base/'temporal_causal'/'phenotype_with_covariates.csv',
        base/'phenotype_with_covariates.parquet', base/'phenotype_with_covariates.csv',
        base/'standard_final'/'phenotype_with_covariates.parquet', base/'standard_final'/'phenotype_with_covariates.csv',
        base/'phenotype_cov_restored_for_postprocess.csv'
    ]
    seen=set()
    for p in exact:
        if p not in seen and p.is_file():
            seen.add(p); yield p


def _resolve_phenotype_cov(base, override=None):
    paths=[Path(override)] if override else list(_candidate_phenotype_paths(base))
    rejected=[]
    for p in paths:
        try:
            x=_read(p)
            col=_find_outcome_column(x)
            if not {'cohort','participant_id','time'}.issubset(x.columns):
                raise ValueError('missing cohort/participant_id/time visit key')
            if not pd.to_numeric(x[col],errors='coerce').notna().any():
                raise ValueError('all outcome values missing')
            print(f'[PHENOTYPE SOURCE] MMSE: {p} (column={col})',flush=True)
            return p,x,col
        except (KeyError,ValueError) as e:
            rejected.append(f'{p}: {e}')
    raise FileNotFoundError('ad_mmse: no Stage II input with real MMSE outcome found under '+str(base)+'\n'+'\n'.join(rejected))


def read_bim_alleles(prefix, variants):
    if not prefix or not variants: return {}
    bim=Path(str(prefix)).with_suffix('.bim')
    if not bim.exists(): return {}
    wanted=set(variants); out={}
    with bim.open() as f:
        for line in f:
            parts=line.strip().split()
            if len(parts)>=6 and parts[1] in wanted:
                out[parts[1]]=(parts[4],parts[5])
                if len(out)==len(wanted): break
    return out


def read_freq_mafs(freq_path, variants):
    if not freq_path: return {}
    p=Path(freq_path)
    if not p.exists(): return {}
    data=pd.read_csv(p, sep=r'\s+', dtype=str)
    id_col=next((x for x in ['ID','SNP','Variant_ID'] if x in data),None)
    maf_col=next((x for x in ['MAF','ALT_FREQS','ALT_FREQ','A1_FREQ'] if x in data),None)
    out={}
    if not id_col or not maf_col: return out
    for row in data.loc[data[id_col].isin(set(variants)), [id_col, maf_col]].itertuples(index=False, name=None):
        try:
            f=float(str(row[1]).split(',')[0])
            if np.isfinite(f) and 0<=f<=1: out[str(row[0])]=min(f,1-f)
        except: pass
    return out


def _lookup_variant_meta(row, cohort, map_bfile=None, adni_bfile=None, map_freq=None, adni_freq=None):
    snp=str(row['lead_snp'])
    if cohort=='MAP':
        maf=read_freq_mafs(map_freq,[snp]).get(snp, None)
        al=read_bim_alleles(map_bfile,[snp]).get(snp, None)
    elif cohort=='ADNI':
        maf=read_freq_mafs(adni_freq,[snp]).get(snp, None)
        al=read_bim_alleles(adni_bfile,[snp]).get(snp, None)
    else:
        maf=al=None
    return maf, al


def load_example_from_stage2(root,row,cohort='MAP',min_participants=8,phenotype_override=None):
    gene=str(row['gene']); snp=str(row['lead_snp']); aid=str(row.get('artifact_id',row.get('stage1_artifact_id','')))
    if not aid or aid=='nan': raise KeyError('Publication row lacks artifact_id')
    base=Path(root)/'single_iv_causal'/'ad_mmse'
    comp=base/'single_iv_genetic_component'/f'{aid}.parquet'
    comp_df=_read(comp)
    req={'participant_id','cohort','time','grex_single_iv','iv_dosage'}
    miss=req-set(comp_df.columns)
    if miss: raise KeyError(f'{comp}: missing {sorted(miss)}')
    covfile,outcome_df,outcome_col=_resolve_phenotype_cov(base,override=phenotype_override)
    lhs=outcome_df[['cohort','participant_id','time',outcome_col]].copy()
    rhs=comp_df[['cohort','participant_id','time','iv_dosage','grex_single_iv']].copy()
    for df in (lhs,rhs):
        df['cohort']=df['cohort'].astype(str)
        df['participant_id']=df['participant_id'].astype(str)
        df['time']=pd.to_numeric(df['time'],errors='coerce')
    if cohort!='ALL':
        lhs=lhs[lhs['cohort'].eq(cohort)].copy(); rhs=rhs[rhs['cohort'].eq(cohort)].copy()
    z=lhs.merge(rhs,on=['cohort','participant_id','time'],how='inner')
    z=z.rename(columns={outcome_col:'outcome'})
    z['genotype']=pd.to_numeric(z['iv_dosage'],errors='coerce').round().astype('Int64')
    z['time_years']=pd.to_numeric(z['time'],errors='coerce')
    z=z[z['genotype'].isin([0,1,2]) & pd.to_numeric(z['outcome'],errors='coerce').notna()].copy()
    z['outcome']=pd.to_numeric(z['outcome'],errors='coerce')
    if z['participant_id'].nunique()<min_participants:
        raise ValueError(f'{gene}/{snp}: only {z["participant_id"].nunique()} participants in cohort {cohort}')
    return z


def _summary_binned(z,year_bin=1.0):
    d=z.copy(); d['time_bin']=(d['time_years']/year_bin).round()*year_bin
    g=d.groupby(['genotype','time_bin'],as_index=False).agg(mean=('outcome','mean'), sd=('outcome','std'), n=('participant_id','nunique'))
    g['se']=g['sd']/np.sqrt(np.maximum(g['n'],1)); return g


def plot_example(fig,spec,row,z,letter,year_bin=1.0,show_lag=True, cohort='MAP', maf=None, alleles=None):
    lag=int(row['selected_lag_order'])
    nrows=3 if show_lag and lag>1 else 2
    heights=[1.95,1.15,0.68] if nrows==3 else [2.0,1.25]
    inner=GridSpecFromSubplotSpec(nrows,1,subplot_spec=spec,height_ratios=heights,hspace=.30)
    top=fig.add_subplot(inner[0]); bottom=fig.add_subplot(inner[1],sharex=top); lagax=fig.add_subplot(inner[2]) if nrows==3 else None
    groups=sorted([int(x) for x in z.genotype.dropna().unique()])
    counts={int(g):int(n) for g,n in z.groupby('genotype').participant_id.nunique().items()}
    gene=str(row['gene']); snp=str(row['lead_snp'])
    top.set_title(f'{letter}  {gene}  |  {snp}  |  MMSE (L={lag})',loc='left',weight='bold',fontsize=9.4,pad=8)
    if maf is not None or alleles is not None:
        maf_txt=f'MAF={maf:.3f}' if maf is not None else 'MAF=NA'
        a_txt=f'PLINK A1/A2={alleles[0]}/{alleles[1]}' if alleles is not None else 'PLINK A1/A2=NA'
        top.text(0.0,1.03,f'{cohort} {maf_txt} | {a_txt} (Stage-II dosage orientation not assumed)', transform=top.transAxes, ha='left', va='bottom', fontsize=7.2)
    binned=_summary_binned(z,year_bin)
    for g in groups:
        col=GENO_COLORS[g]
        q=binned[binned.genotype.eq(g)]; q=q[q.n>=3]
        if not q.empty:
            top.plot(q.time_bin,q['mean'],color=col,lw=1.5,marker='o',ms=3,label=f'Dosage {g} (n={counts[g]})')
            top.fill_between(q.time_bin.to_numpy(float),(q['mean']-q.se.fillna(0)).to_numpy(float),(q['mean']+q.se.fillna(0)).to_numpy(float),color=col,alpha=.12,lw=0)
        zz=z[z.genotype.eq(g)]
        if counts[g] < 3: continue
        pred=zz.groupby('time_years',as_index=False)['grex_single_iv'].mean().sort_values('time_years')
        xp=pred.time_years.to_numpy(float); yp=pred.grex_single_iv.to_numpy(float)
        if len(np.unique(xp))>=4:
            t0=float(np.mean(xp)); sc=max(float(np.std(xp)),1e-6); v=(xp-t0)/sc; co=np.polyfit(v,yp,3)
            grid=np.linspace(xp.min(),xp.max(),180); fit=np.polyval(co,(grid-t0)/sc); bottom.plot(grid,fit,color=col,lw=1.5)
        elif len(xp)>1: bottom.plot(xp,yp,color=col,lw=1.4)
        if len(xp):
            qidx=np.unique(np.linspace(0,len(xp)-1,min(8,len(xp))).astype(int)); bottom.scatter(xp[qidx],yp[qidx],color=col,facecolors='white',s=11,zorder=3,lw=.8)
    top.set_ylabel('MMSE',fontsize=8.2); bottom.set_ylabel('Predicted\ntGReX(t)',fontsize=8); bottom.set_xlabel('Years since baseline',fontsize=8)
    for ax in (top,bottom):
        ax.spines[['top','right']].set_visible(False); ax.tick_params(labelsize=7); ax.grid(axis='y',color='#E6EAEE',lw=.65)
    top.legend(frameon=False,fontsize=6.4,loc='upper left',ncol=min(len(groups),3),handlelength=1.2)
    p=pd.to_numeric(pd.Series([row.get('ADNI_pvalue')]),errors='coerce').iloc[0]
    if np.isfinite(p): top.text(.98,.02,f'Replication P={p:.2g}',ha='right',va='bottom',transform=top.transAxes,fontsize=7.2)
    disc_sum=pd.to_numeric(pd.Series([row.get('MAP_coef_sum')]),errors='coerce').iloc[0]
    repl_sum=pd.to_numeric(pd.Series([row.get('ADNI_coef_sum')]),errors='coerce').iloc[0]
    if np.isfinite(disc_sum) or np.isfinite(repl_sum):
        s1='NA' if not np.isfinite(disc_sum) else ('+' if disc_sum>=0 else '')+f'{disc_sum:.3g}'
        s2='NA' if not np.isfinite(repl_sum) else ('+' if repl_sum>=0 else '')+f'{repl_sum:.3g}'
        bottom.text(.98,.96,f'Σβ: MAP {s1}; ADNI {s2}', transform=bottom.transAxes, ha='right', va='top', fontsize=6.8, color='#55697A')
    if lagax is not None:
        lags=[]
        for k in range(1,lag+1):
            v=pd.to_numeric(pd.Series([row.get(f'MAP_coef_x_lag{k}')]),errors='coerce').iloc[0]
            if np.isfinite(v): lags.append((k,float(v)))
        if lags:
            for k,v in lags:
                lagax.vlines(k,0,v,color='#C6CDD4',lw=1.2); lagax.scatter(k,v,c='#B95569' if v>=0 else '#3884A8',s=22,zorder=3)
            lagax.axhline(0,color='#555',lw=.7); lagax.set_xticks([x for x,_ in lags]); lagax.set_ylabel('Lag β',fontsize=7.5); lagax.set_xlabel('Previous visits (lag)',fontsize=7.5); lagax.grid(axis='y',color='#EEF0F2',lw=.5); lagax.tick_params(labelsize=7); lagax.spines[['top','right']].set_visible(False)
        else: lagax.set_axis_off()
    return fig
