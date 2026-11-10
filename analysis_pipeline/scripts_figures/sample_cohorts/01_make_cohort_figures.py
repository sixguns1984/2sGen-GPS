#!/usr/bin/env python3
"""Two publication figures from actual frozen CTAR and Stage-II input records.

No old manuscript sample counts. No genotype QC or model fitting.
Overlap means participants represented by the input records and eligible for
at least one visit-order lag-1 analysis, not the union of all exact regression
complete-case participants (those individual model ID lists were not saved).
"""
from __future__ import annotations
import argparse, json, re, sys, warnings
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CTAR='submitted_followup_ctar_mptga'
CLINICAL={'updrs1':'UPDRS I','updrs2':'UPDRS II','updrs3':'UPDRS III','moca':'MoCA'}
# Logical figure category; the physical analysis directory is configured separately.
CSF_KEY='csf_olink'
DEFAULT_CSF_ANALYSIS_NAME='csf_olink_product_adjusted_D01_preferred_v3'
CHARTS={**CLINICAL,CSF_KEY:'CSF Olink','mri_dbm_lag1':'MRI DBM'}
COHORTS={'PP':'PPMI','PD':'PDBP','SU':'SU'}
TEAL='#1EA4A3';PINK='#BD2796';PD_BLUE='#557CB3';CTRL_GREEN='#9DCB64'
RACE_COLORS=['#0094C5','#159C9B','#8C329A','#E9A91F','#D75194','#424550','#9097A2']
OVERLAP_COLOR='#3279A8';ONLY_STAGE1='#9DC6C8';ONLY_STAGE2='#DF9D6F'

def require(p,desc):
    p=Path(p)
    if not p.is_file():raise FileNotFoundError(f'{desc} missing: {p}')
    return p

def canon_pid(v,cohort):
    s=str(v).strip()
    if s.lower() in {'','nan','none','<na>','null'}:return ''
    s=re.sub(r'\.0$','',s)
    # The phenotypes and WGS FAMs can use PP.123/PP-123/123.
    # Avoid mixing patients from different AMP-PD cohorts by re-attaching cohort.
    s=re.sub(r'^(PP|PD|SU)[.-]','',s,flags=re.I)
    return f'{cohort}-{s}'

def read_table(p,columns=None):
    p=Path(p)
    if p.suffix.lower()=='.parquet':
        if columns is not None:
            import pyarrow.parquet as pq
            available=set(pq.ParquetFile(p).schema.names)
            columns=[c for c in columns if c in available]
        return pd.read_parquet(p,columns=columns)
    return pd.read_csv(p,low_memory=False,usecols=columns) if columns else pd.read_csv(p,low_memory=False)

def set_style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
         'font.size':8.6,'axes.labelsize':8.5,'axes.titlesize':9.8,'legend.fontsize':7.6,
         'xtick.labelsize':7.8,'ytick.labelsize':7.8,'axes.linewidth':.75,
         'pdf.fonttype':42,'svg.fonttype':'none','savefig.dpi':500})

def save_figure(fig,dest):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg','png']:
        fig.savefig(dest.with_suffix('.'+ext),dpi=500 if ext=='png' else None,
                    bbox_inches='tight',pad_inches=.11)
    plt.close(fig)

def stage1_eligible(paths,cohort):
    """Replay metadata steps from gengps.amp_pd.real_qtl.run_gene_chunk(current_v89).
    This is the all-gene metadata universe; individual gene expression missingness
    can further affect specific SNP-gene models.
    """
    from gengps.amp_pd.real_qtl import (load_cohort_metadata,_augment_fixed_genotype_pcs,
        _prepare_irregular_followup_rows,method_covariates,_drop_missing_covariates)
    import pyarrow.parquet as pq
    p=paths.data_dir/'expression'/'amp_pd_expression_z.parquet'
    require(p,'Stage I all-visits RNA expression')
    schema=set(pq.ParquetFile(p).schema.names)
    if 'sample_id' not in schema:raise KeyError(f'{p} lacks sample_id')
    eid=set(pd.read_parquet(p,columns=['sample_id']).sample_id.astype(str))
    meta=load_cohort_metadata(paths,cohort)
    meta=meta[meta.sample_id.astype(str).isin(eid)].copy()
    meta=_augment_fixed_genotype_pcs(paths,cohort,meta,n_pcs=8)
    empty=pd.DataFrame(index=meta.index)
    meta,empty,n_duplicates=_prepare_irregular_followup_rows(meta,empty)
    covars=method_covariates(CTAR,cohort,meta,cell_adjust=True)
    needed={'age_at_baseline','sex_binary','diagnosis_binary','RIN'}|{f'ctar_gPC{i}' for i in range(1,9)}
    if not needed.issubset(covars):raise RuntimeError(f'Stage I CTAR key covariates missing from selected model: {sorted(needed-set(covars))}')
    meta,empty=_drop_missing_covariates(meta,empty,covars)
    if meta.empty:raise RuntimeError(f'No retained Stage I CTAR observations: {cohort}')
    meta=meta.copy()
    meta['participant_key']=[canon_pid(v,cohort) for v in meta.participant_id]
    meta=meta[meta.participant_key.ne('')].copy()
    # valid cohort-specific counts are not inherited from old manuscript.
    return meta, {'cohort':cohort,'source_metadata':str(paths.data_dir/'metadata'/'amp_pd_samples.parquet'),
                   'source_expression':str(p),'n_unique_visits':len(meta),
                   'n_unique_people':meta.participant_key.nunique(),
                   'n_technical_duplicate_visits_removed':int(n_duplicates),
                   'covariates':covars,'definition':'Current-v8.9 CTAR aligned RNA sample and covariate universe'}

def pick_column(df,names):
    cols={str(c).lower().strip().replace(' ','_'):c for c in df.columns}
    for c in names:
        k=c.lower().strip().replace(' ','_')
        if k in cols:return cols[k]
    return None

def fill_demographics(stage_meta,demographics_csv,cohort):
    z=stage_meta.copy()
    z['participant_id']=z.participant_id.astype(str)
    # Stage I sample metadata has age + sex; raw AMP-PD demographic file is the
    # authoritative source for race (the main preprocess previously dropped it).
    raw=read_table(require(demographics_csv,'AMP-PD clinical demographics table'))
    idcol=pick_column(raw,['participant_id','participantid','participant','subject_id'])
    if idcol is None:raise KeyError(f'Demographics does not contain participant_id: {demographics_csv}')
    racecol=pick_column(raw,['race','race_category','race_ethnicity','self_reported_race','race_1','racial_background'])
    # May contain site-specific race column name. Never silently label as White.
    if racecol is None:
        matches=[c for c in raw if 'race' in str(c).lower()]
        if len(matches)==1:racecol=matches[0]
    if racecol is None:
        racecol=pick_column(z,['race','race_category','race_ethnicity'])
        if racecol is None:raise KeyError('Race is not found in AMP-PD clinical demographics or Stage I metadata; inspect columns and add --race-column to the script as appropriate. No fictional race proportions will be plotted.')
        return z
    raw=raw[[idcol,racecol]].copy()
    raw['participant_key']=[canon_pid(v,cohort) for v in raw[idcol]]
    raw=raw[raw.participant_key.ne('')].copy()
    values=raw.groupby('participant_key')[racecol].nunique(dropna=True)
    if (values>1).any():raise ValueError(f'Conflicting race labels for same participant: {values[values>1].head().to_dict()}')
    raw=raw.drop_duplicates('participant_key')
    z=z.drop(columns=['race','race_category','race_ethnicity'],errors='ignore')
    z=z.merge(raw[['participant_key',racecol]].rename(columns={racecol:'race'}),
              on='participant_key',how='left',validate='many_to_one')
    if z.race.notna().sum()==0:
        raise RuntimeError(f'No race labels matched Stage I {cohort} participant IDs using {demographics_csv}; inspect identity formatting. Do not plot an all-Unknown race chart.')
    return z

def race_name(x):
    if pd.isna(x) or str(x).strip().lower() in {'','nan','unknown','not reported','not available','undisclosed','missing','prefer not to answer'}:
        return 'Unknown / not reported'
    s=str(x).strip();lo=s.lower()
    if lo in {'white','caucasian','white/caucasian'}:return 'White'
    if any(w in lo for w in ['black','african american','african-american']):return 'Black / African American'
    if 'asian' in lo:return 'Asian'
    if 'multi' in lo or 'more than one' in lo:return 'Multiracial'
    if 'american indian' in lo or 'alaska native' in lo:return 'American Indian / Alaska Native'
    if 'native hawaiian' in lo or 'pacific islander' in lo:return 'Native Hawaiian / Pacific Islander'
    if lo in {'other','other race'}:return 'Other'
    return s

def demographic_people(meta,cohort):
    cols=['participant_key','age_at_baseline','sex','sex_binary','diagnosis','diagnosis_binary','race']
    missing=[x for x in ['participant_key','age_at_baseline','sex_binary','diagnosis_binary','race'] if x not in meta]
    if missing:raise KeyError(f'Missing Stage I demographics {missing}')
    z=meta[[x for x in cols if x in meta]].copy()
    # participant-level values must not change across RNA visits, except missing entries.
    for c in ['age_at_baseline','sex_binary','diagnosis_binary','race']:
        ng=z.groupby('participant_key')[c].nunique(dropna=True)
        if (ng>1).any():raise RuntimeError(f'{cohort}: contradictory {c} across visits: {ng[ng>1].head().to_dict()}')
    z=z.drop_duplicates('participant_key').copy()
    z['baseline_age']=pd.to_numeric(z.age_at_baseline,errors='coerce')
    z['sex_label']=pd.to_numeric(z.sex_binary,errors='coerce').map({1:'Male',0:'Female'}).fillna('Unknown')
    d=pd.to_numeric(z.diagnosis_binary,errors='coerce')
    z['diagnosis_label']=np.where(d.eq(1),'PD',np.where(d.eq(0),'Control','Other / unknown'))
    z['race_label']=z.race.map(race_name)
    z['cohort']=cohort
    return z

def name_for_cohort(c):return COHORTS.get(str(c),str(c))

def genotype_matched(base,cohort):
    pf=Path(base)/'participant_genotype_match.csv'
    require(pf,'Stage II phenotype-genotype matching audit')
    m=pd.read_csv(pf,dtype={'participant_id':str},low_memory=False)
    if 'participant_id' not in m or 'genotype_match' not in m:
        raise KeyError(f'{pf} must contain participant_id and genotype_match')
    m['participant_key']=[canon_pid(x,cohort) for x in m.participant_id]
    good=m.genotype_match.astype(str).str.lower().str.strip()
    # Empty or nan is not proof of a valid genotype.
    return set(m.loc[good.isin(['fid','iid','matched','true','yes','1']),'participant_key'])

def valid_lag_people(df,cohort,genotyped,feature=None):
    d=df.copy();d['cohort']=d.cohort.astype(str)
    d=d[d.cohort.eq(cohort)].copy()
    if d.empty:return set(),0
    if 'time' not in d or 'participant_id' not in d:
        raise KeyError('Stage II data missing cohort/participant_id/time')
    d['time']=pd.to_numeric(d.time,errors='coerce')
    d=d[np.isfinite(d.time)].copy()
    d['participant_key']=[canon_pid(x,cohort) for x in d.participant_id]
    if feature is not None:
        if feature not in d:raise KeyError(f'Stage II analysis phenotype column {feature!r} missing')
        d=d[pd.to_numeric(d[feature],errors='coerce').notna()].copy()
    d=d[d.participant_key.isin(genotyped)]
    d=d.drop_duplicates(['participant_key','time'])
    per=d.groupby('participant_key').time.nunique()
    good=set(per[per>=2].index.astype(str))
    return good,int(len(d[d.participant_key.isin(good)]))

def discover_run_dir(base):
    base=Path(base)
    q=list(base.glob('temporal_causal_agecubic_full_followup_fast/phenotype_with_covariates.parquet'))
    q+=list(base.glob('temporal_causal*/phenotype_with_covariates.parquet'))
    q+=list(base.glob('*/phenotype_with_covariates.parquet'))
    q+=list(base.glob('temporal_causal/phenotype_with_covariates.parquet'))
    q=list(dict.fromkeys(q))
    if not q:raise FileNotFoundError(f'Cannot find completed Stage II FAST phenotype_with_covariates.parquet under {base}. Avoid using raw unfiltered cov as if already analyzed.')
    preferred=[p for p in q if p.parent.name=='temporal_causal_agecubic_full_followup_fast']
    if preferred:return preferred[0]
    if len(q)>1:raise RuntimeError(f'Ambiguous Stage II run directories: {q}')
    return q[0]

def collect_clinical(root,analysis):
    base=Path(root)/'single_iv_causal'/analysis
    f=discover_run_dir(base)
    config=f.parent/'requested_run_config.json'
    require(config,'Completed Stage II run config')
    cfg=json.loads(config.read_text())
    feature=cfg.get('outcome_col')
    if not feature:raise ValueError(f'outcome_col is not registered in {config}')
    wanted=['cohort','participant_id','time',feature,'age','sex','AAO', 'upd23a_medication_for_pd']
    d=read_table(f,columns=wanted)
    if feature not in d:raise KeyError(f'{f}: missing actual outcome column {feature}')
    out={}
    for c in sorted(d.cohort.dropna().astype(str).unique()):
        g=genotype_matched(base,c)
        ids,n_vis=valid_lag_people(d,c,g,feature=feature)
        out[c]={'ids':ids,'n_visits':n_vis,'source':str(f),'feature':feature}
    return out

def collect_csf(root, analysis_name=DEFAULT_CSF_ANALYSIS_NAME):
    # Do not silently fall back to the obsolete csf_olink_pd_progress results.
    base=Path(root)/'single_iv_causal'/analysis_name
    f=require(base/'csf_prepared'/'csf_proteins.parquet',
              f'Prepared CSF protein measurements for analysis {analysis_name}')
    import pyarrow.parquet as pq
    cols=pq.ParquetFile(f).schema.names
    reserved={'cohort','participant_id','sample_id','time','age','age_visit','sex','diagnosis'}
    prot=[c for c in cols if c not in reserved and not str(c).startswith('olink_batch_')]
    if not prot:raise RuntimeError(f'{f}: no Olink protein columns')
    # Use the exact prepared table from the selected finished product-adjusted analysis.
    d=read_table(f)
    print(f'[CSF SOURCE] {f}; rows={len(d):,}; analysis={analysis_name}', flush=True)
    out={}
    for c in sorted(d.cohort.dropna().astype(str).unique()):
        geno=genotype_matched(base,c)
        if not geno:
            raise RuntimeError(f'{analysis_name}/{c}: genotype matching audit contains no matched participants; check exact Stage II preparation directory')
        x=d[d.cohort.astype(str).eq(c)].copy()
        x['participant_key']=[canon_pid(s,c) for s in x.participant_id]
        x['time']=pd.to_numeric(x.time,errors='coerce')
        matched_count=int(x.participant_key.isin(geno).sum())
        if matched_count==0:
            raise RuntimeError(f'{analysis_name}/{c}: no participants match prepared CSF data to participant_genotype_match.csv; inspect IDs')
        x=x[x.participant_key.isin(geno)&x.time.notna()].copy()
        # Need >=2 measurements for the SAME protein, not just any two with different proteins.
        x=x.sort_values(['participant_key','time']).drop_duplicates(['participant_key','time'])
        if x.empty:ids=set()
        else:
            good=x[prot].apply(pd.to_numeric,errors='coerce').notna()
            good['participant_key']=x.participant_key.values
            per=good.groupby('participant_key').sum(numeric_only=True)
            ids=set(per.index[(per>=2).any(axis=1)].astype(str))
        out[c]={'ids':ids,'n_visits':int(len(x[x.participant_key.isin(ids)])),
                'source':str(f),'feature':'at least one protein with >=2 valid visits'}
        print(f'[CSF COHORT] {c}: eligible_participants={len(ids):,}; valid_visits={out[c]["n_visits"]:,}', flush=True)
    return out

def collect_mri(root,mri_source):
    base=Path(root)/'single_iv_causal'/'mri_dbm_lag1'
    require(base/'mri_visit_covariates.csv','MRI prediction covariates')
    geno=genotype_matched(base,'PP')
    f=require(mri_source,'Original MRI DBM visit table')
    import pyarrow.parquet as pq
    if str(f).endswith('.parquet'):
        cols=pq.ParquetFile(f).schema.names
    else:
        cols=pd.read_csv(f,nrows=0).columns.tolist()
    roi=[c for c in cols if re.search(r'\.pc[123]$',str(c),re.I)]
    if not roi:raise KeyError(f'{f}: no regional DBM PC1/PC2/PC3 outcome columns')
    needed=['PATNO','time','age','sex','IQR']
    if not set(needed).issubset(cols):raise KeyError(f'{f}: MRI columns missing {set(needed)-set(cols)}')
    d=read_table(f,columns=needed+roi)
    d['cohort']='PP';d['participant_id']=d.PATNO.astype(str)
    d['participant_key']=[canon_pid(s,'PP') for s in d.participant_id]
    for n in ['time','age','IQR']:d[n]=pd.to_numeric(d[n],errors='coerce')
    d=d[d.participant_key.isin(geno)&d.time.notna()&d.age.notna()&d.IQR.notna()].copy()
    d=d.sort_values(['participant_key','time']).drop_duplicates(['participant_key','time'])
    if d.empty:ids=set()
    else:
        valid=d[roi].apply(pd.to_numeric,errors='coerce').notna()
        valid['participant_key']=d.participant_key.values
        groups=valid.groupby('participant_key').sum(numeric_only=True)
        ids=set(groups.index[(groups>=2).any(axis=1)].astype(str))
    return {'PP':{'ids':ids,'n_visits':int(len(d[d.participant_key.isin(ids)])),
                  'source':str(f),'feature':'at least one regional DBM PC with >=2 valid MRI visits'}}

def cohort_demographics_figure(people,metadata,out):
    fig,axs=plt.subplots(2,3,figsize=(12.4,7.7))
    race_order=['White','Black / African American','Asian','Multiracial',
                'American Indian / Alaska Native','Native Hawaiian / Pacific Islander','Other','Unknown / not reported']
    race_categories=sorted(set(pd.concat([people[c].race_label for c in ['PP','PD']]).dropna())-set(race_order))
    races=[r for r in race_order+race_categories if any((people[c].race_label==r).any() for c in ['PP','PD'])]
    rcolors={name:RACE_COLORS[i%len(RACE_COLORS)] for i,name in enumerate(races)}
    bins=np.arange(25,96,5)
    for row,c in enumerate(['PP','PD']):
        z=people[c]; ax=axs[row,0]
        counts=z.sex_label.value_counts();labels=['Male','Female','Unknown']
        vals=[int(counts.get(l,0)) for l in labels];use=[i for i,v in enumerate(vals) if v]
        colors=[TEAL,PINK,'#9298A1']
        ax.pie([vals[i] for i in use],startangle=90,counterclock=False,colors=[colors[i] for i in use],
               wedgeprops={'width':.28,'edgecolor':'white','linewidth':.8})
        ax.text(0,0,f'{name_for_cohort(c)}\nn = {len(z):,}',ha='center',va='center',fontsize=10,weight='bold')
        ax.legend(handles=[Patch(color=colors[i],label=f'{labels[i]}: {vals[i]} ({vals[i]/len(z):.1%})') for i in use],
                  loc='lower center',bbox_to_anchor=(.5,-.16),ncol=2,frameon=False,fontsize=7.5)
        ax.set_title(f'{name_for_cohort(c)} — sex',loc='left',weight='semibold')
        axs[row,1].hist([z.loc[z.diagnosis_label=='PD','baseline_age'].dropna(),z.loc[z.diagnosis_label=='Control','baseline_age'].dropna()],
                        bins=bins,color=[PD_BLUE,CTRL_GREEN],label=['PD','Control'],edgecolor='white',linewidth=.35,stacked=False,alpha=.75)
        a=axs[row,1];a.set_xlabel('Baseline age (years)');a.set_ylabel('Participants');a.legend(frameon=False,loc='upper left')
        a.set_title(f'{name_for_cohort(c)} — baseline age',loc='left',weight='semibold')
        for sp in ['top','right']:a.spines[sp].set_visible(False)
        a=axs[row,2]
        s=z.race_label.value_counts();xx=np.arange(len(races));yy=[int(s.get(r,0)) for r in races]
        a.bar(xx,yy,color=[rcolors[r] for r in races],width=.75)
        a.set_xticks(xx,labels=[r.replace(' / ','/').replace(' ','\n',1) if len(r)>15 else r for r in races],rotation=38,ha='right',fontsize=6.5)
        a.set_ylabel('Participants');a.set_title(f'{name_for_cohort(c)} — race',loc='left',weight='semibold')
        for i,n in enumerate(yy):
            if n:a.text(i,n+max(yy)*.015,f'{n}\n({n/len(z):.1%})',ha='center',va='bottom',fontsize=6.4)
        a.set_ylim(0,max(1,max(yy)*1.18));a.spines['top'].set_visible(False);a.spines['right'].set_visible(False)
    for lab,ax in zip(list('abcdef'),axs.flat):
        ax.text(-.1,1.06,lab,transform=ax.transAxes,size=13,weight='bold',ha='right')
    fig.subplots_adjust(hspace=.52,wspace=.4)
    fig.suptitle('Stage I CTAR-MPTGA: participants included in longitudinal tQTL mapping',fontsize=13,weight='semibold',y=1.02)
    save_figure(fig,out)

def overlap_figure(stage1_sets,analysis_sets,rows,out):
    # One row per cohort, one column per phenotype. Counts are explicit set arithmetic.
    groups=['PP','PD','SU'];cols=list(CHARTS)
    fig,axes=plt.subplots(3,1,figsize=(12.3,8.7),sharex=True)
    barwidth=.74;x=np.arange(len(cols));
    for i,cohort in enumerate(groups):
        a=axes[i]
        for j,analysis in enumerate(cols):
            rec=next((r for r in rows if r['cohort']==cohort and r['analysis']==analysis),None)
            if rec is None:continue
            a.bar(j,rec['only_stage1'],width=barwidth,color=ONLY_STAGE1,edgecolor='white',lw=.7)
            a.bar(j,rec['both'],bottom=rec['only_stage1'],width=barwidth,color=OVERLAP_COLOR,edgecolor='white',lw=.7)
            a.bar(j,rec['only_stage2'],bottom=rec['only_stage1']+rec['both'],width=barwidth,color=ONLY_STAGE2,edgecolor='white',lw=.7)
            end=rec['only_stage1']+rec['both']+rec['only_stage2']
            a.text(j,end+max(2,end*.018),f"{rec['both']} overlap\n{rec['stage2_n']} Stage II",ha='center',va='bottom',fontsize=7.5)
        a.set_ylabel('Participants');a.set_title(f'{COHORTS[cohort]} (Stage I CTAR n={len(stage1_sets.get(cohort,set())):,})',loc='left',weight='semibold')
        a.spines['right'].set_visible(False);a.spines['top'].set_visible(False)
        a.grid(axis='y',color='#ECF0F2',lw=.6);a.set_axisbelow(True)
        mx=max([r['only_stage1']+r['both']+r['only_stage2'] for r in rows if r['cohort']==cohort] or [0]);a.set_ylim(0,max(10,mx*1.20))
        for j,c in enumerate(cols):
            if not any(r['cohort']==cohort and r['analysis']==c for r in rows):
                a.text(j,.07,'N/A',transform=a.get_xaxis_transform(),ha='center',fontsize=7.6,color='#78828C')
    axes[-1].set_xticks(x,labels=[CHARTS[k] for k in cols]);
    fig.legend(handles=[Patch(facecolor=ONLY_STAGE1,label='Stage I only'),Patch(facecolor=OVERLAP_COLOR,label='Both stages'),Patch(facecolor=ONLY_STAGE2,label='Stage II only')],
               loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.5,.982))
    fig.suptitle('Participant overlap between Stage I CTAR and Stage II phenotype inputs',fontsize=13,weight='semibold',y=1.05)
    fig.text(.5,.005,'Distinct participants; Stage II = genotype-matched outcome inputs with ≥2 usable visits (lag-1 eligibility), not per-IV model-specific complete cases.',
             fontsize=7.6,ha='center',color='#59646F')
    fig.subplots_adjust(hspace=.39,top=.90,bottom=.07)
    save_figure(fig,out)

def main():
    ap=argparse.ArgumentParser(description='Rebuild Stage I CTAR cohort demographics and Stage I/Stage II input overlap from real project data.')
    ap.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    ap.add_argument('--code',default='')
    ap.add_argument('--out-root',default='')
    ap.add_argument('--demographics-file',default='')
    ap.add_argument('--mri-source',default='/public2/data/luojunfeng/2sTGCS_binary/outcome/DBM/confound_pass_snp/MRI_DBM.csv')
    ap.add_argument('--csf-analysis-name',default=DEFAULT_CSF_ANALYSIS_NAME,
                    help='Exact completed CSF Stage II analysis directory under single_iv_causal; no fallback to obsolete analyses')
    ap.add_argument('--allow-missing-analyses',action='store_true',help='Partial exploratory figure only; absent analyses not shown or counted as zero')
    a=ap.parse_args();root=Path(a.root);code=Path(a.code or root/'2sGenGPS_AMP_PD_tQTL_v8_9_clean_optimized')
    out=Path(a.out_root or root/'manuscript_figures'/'ctar_sample_cohorts_v1');out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(code))
    from gengps.amp_pd.config import AmpPdPaths
    paths=AmpPdPaths(root=str(root))
    demographic_file=a.demographics_file or paths.demographics_csv
    set_style();stage1={};people={};audits={}
    for cohort in ['PP','PD']:
        meta,audit=stage1_eligible(paths,cohort)
        meta=fill_demographics(meta,demographic_file,cohort)
        z=demographic_people(meta,cohort)
        stage1[cohort]=set(z.participant_key.astype(str));people[cohort]=z
        audits[cohort]=audit
        z.to_csv(out/f'Stage1_{cohort}_CTAR_participants_demographics.csv',index=False)
    stage1['SU']=set()
    sources={};missing={}
    for analysis in CLINICAL:
        try:sources[analysis]=collect_clinical(root,analysis)
        except Exception as exc:
            if not a.allow_missing_analyses:raise RuntimeError(f'Cannot establish Stage II {analysis} participant input set') from exc
            missing[analysis]=repr(exc)
    for label,func in [(CSF_KEY,lambda:collect_csf(root,a.csf_analysis_name)),('mri_dbm_lag1',lambda:collect_mri(root,a.mri_source))]:
        try:sources[label]=func()
        except Exception as exc:
            if not a.allow_missing_analyses:raise RuntimeError(f'Cannot establish Stage II {label} participant input set') from exc
            missing[label]=repr(exc)
    membership={};results=[]
    for analysis,dct in sources.items():
        for cohort,info in dct.items():
            ids=set(info['ids']);shared=ids&stage1.get(cohort,set())
            row={'analysis':analysis,'label':CHARTS[analysis],'cohort':cohort,
                 'stage1_n':len(stage1.get(cohort,set())),'stage2_n':len(ids),
                 'both':len(shared),'only_stage1':len(stage1.get(cohort,set())-ids),
                 'only_stage2':len(ids-stage1.get(cohort,set())),
                 'stage2_valid_visits':info['n_visits'],'stage2_source':info['source'],
                 'definition':info['feature']}
            results.append(row);membership[(analysis,cohort)]=ids
            assert row['stage1_n']==row['only_stage1']+row['both']
            assert row['stage2_n']==row['only_stage2']+row['both']
    overlap=pd.DataFrame(results)
    overlap.to_csv(out/'Stage1_Stage2_participant_overlap_counts.csv',index=False)
    universe=set().union(*stage1.values(),*membership.values()) if membership else set().union(*stage1.values())
    rows=[]
    for key in sorted(universe):
        cohort=key.split('-',1)[0]
        row={'participant_key':key,'cohort':cohort,'Stage1_CTAR':key in stage1.get(cohort,set())}
        for name in CHARTS:row['Stage2_'+name]=(key in membership.get((name,cohort),set()))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'Stage1_Stage2_participant_membership_matrix.csv',index=False)
    aud={'stage1':audits,'stage2_available':{k:list(v.keys()) for k,v in sources.items()},'stage2_missing':missing,
         'identity':'cohort-prefixed participant IDs; visit/sample IDs excluded',
         'meaning':'Stage1 current_v89 CTAR metadata analysis universe vs Stage2 genotype-matched phenotype inputs with >=2 usable visits. Not exact per-IV/lag fitted persons.',
         'no_new_genotype_QC':True,
         'source_paths':{'demographics':str(demographic_file),'mri':str(a.mri_source),
                         'csf_analysis_name':a.csf_analysis_name,
                         'csf_prepared_parquet':str(root/'single_iv_causal'/a.csf_analysis_name/'csf_prepared'/'csf_proteins.parquet')}}
    (out/'provenance_and_definitions.json').write_text(json.dumps(aud,indent=2,ensure_ascii=False),encoding='utf-8')
    cohort_demographics_figure(people,audits,out/'supplementary'/'Supplementary_Figure_2_CTAR_cohort_demographics')
    overlap_figure(stage1,membership,results,out/'supplementary'/'Supplementary_Figure_Stage1_Stage2_participant_overlap')
    print('[DONE] Stage I PPMI:',len(stage1['PP']),'PDBP:',len(stage1['PD']))
    print('[DONE] Participant overlap table:')
    print(overlap[['analysis','cohort','stage1_n','stage2_n','both','only_stage1','only_stage2']].to_string(index=False))
    print('[FIGS]',out/'supplementary')
if __name__=='__main__':main()
