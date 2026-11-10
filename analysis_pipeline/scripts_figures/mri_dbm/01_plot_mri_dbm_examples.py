#!/usr/bin/env python3
"""Publication MRI lag-1 trajectory examples (FLOT1 / C7orf25).

Uses the project's unchanged 14_mri_dbm_lag1.py data preparation, genotype PCs,
and unchanged Stage-I prediction cache. Both charts are descriptive displays
of MRI outcomes and corresponding frozen predicted genetic components, never
model-adjusted outcome trajectories. Stage-I out-of-support predictions are
shown for the full MRI visit window as EXTRAPOLATED VISUALIZATION ONLY; the
lag-1 statistical model continues to enforce the original support mask.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('cairo')  # Use Cairo backend for high-quality output
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

DEFAULT_ROOT = '/public2/data/luojunfeng/2sGenGPS_revised_chatgpt'
DEFAULT_MRI = '/public2/data/luojunfeng/2sTGCS_binary/outcome/DBM/confound_pass_snp/MRI_DBM.csv'
TARGETS = [
    ('FLOT1', 'rs2253588', 'ff9c4d59af4ca416', 'RightCaudatenucleus7.pc1',
     'Right caudate nucleus | ROI 7 (PC1)'),
    ('C7orf25', 'rs35162690', '50bdbe15dd2df1be',
     'LeftSubstantianigra.parsreticulata3.pc1',
     'Left SNr | ROI 3 (PC1)'),
]
COLORS = {0: '#D8A031', 1: '#2A6DB3', 2: '#D95F02'}


def load_mri_analysis_script(code):
    source = code / 'scripts_stage2' / '14_mri_dbm_lag1.py'
    if not source.exists():
        raise FileNotFoundError('Canonical MRI fit script not found: '+str(source))
    spec=importlib.util.spec_from_file_location('mri_dbm_official', source)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_exact_hit(df, gene, snp, aid, roi):
    required = ['gene','lead_snp','artifact_id','roi','beta_grex_lag1','p_cluster','q_global_BH']
    missing = set(required)-set(df.columns)
    if missing: raise ValueError('MRI FDR input missing columns: '+str(sorted(missing)))
    mask = (df.gene.astype(str).eq(gene) & df.lead_snp.astype(str).eq(snp)
            & df.artifact_id.astype(str).eq(aid) & df.roi.astype(str).eq(roi))
    hit=df.loc[mask]
    if len(hit)!=1:
        raise ValueError(f'Expected exactly 1 global-FDR result for {gene} {snp} {aid} {roi}, got {len(hit)}')
    return hit.iloc[0]


def load_bim_alleles(bim_file, wanted):
    if not bim_file.exists():
        warnings.warn(f'BIM missing: {bim_file}; allele subtitle will show NA (never inferred)')
        return {}
    result={}
    with bim_file.open() as fin:
        for line in fin:
            parts=line.split()
            if len(parts)>=6 and parts[1] in wanted:
                result[parts[1]]=(parts[4], parts[5])
                if len(result)==len(wanted):break
    return result


def read_component(path):
    if path.exists():
        comp=pd.read_parquet(path)
    elif path.with_suffix('.csv').exists():
        # CSV is for tests/audit only; official Stage II cache is parquet.
        comp=pd.read_csv(path.with_suffix('.csv'),low_memory=False)
    else:
        raise FileNotFoundError(f'MRI-specific frozen GReX missing: {path}. Do not substitute UPDRS GReX cache.')
    needed={'cohort','participant_id','time','grex_single_iv','iv_dosage',
            'within_stage1_followup_support'}
    missing=needed-set(comp.columns)
    if missing:raise ValueError(f'GReX cache {path} missing: {sorted(missing)}')
    return comp


def support_bool(s):
    if pd.api.types.is_bool_dtype(s):return s.fillna(False)
    return s.astype('string').str.strip().str.lower().isin(['true','1','yes','t'])


def model_rows(mri_mod, base, pcs, row, root, analysis_name, max_years):
    gene,snp,aid,roi,_=row
    if roi not in base.columns or 'prev__'+roi not in base:
        raise KeyError('MRI ROI missing from prepared matrix: '+roi)
    comp_file=root/'single_iv_causal'/analysis_name/'single_iv_genetic_component'/f'{aid}.parquet'
    c=read_component(comp_file)
    c=c[c.cohort.astype(str).eq('PP')].copy()
    for field, value in [('gene',gene),('lead_snp',snp),('artifact_id',aid)]:
        if field in c.columns:
            c=c[c[field].astype(str).eq(str(value))].copy()
    c['pid']=mri_mod.txtid(c.participant_id)
    c['time']=pd.to_numeric(c.time,errors='coerce')
    c['grex_single_iv']=pd.to_numeric(c.grex_single_iv,errors='coerce')
    c['iv_dosage']=pd.to_numeric(c.iv_dosage,errors='coerce')
    c=c[support_bool(c.within_stage1_followup_support)].dropna(subset=['pid','time','grex_single_iv']).copy()
    if c.duplicated(['pid','time']).any():
        raise ValueError(f'Duplicated MRI-specific frozen predictions for {gene} / {snp}')

    # The actual lag-1 predictor is the previous MRI visit's predicted GReX.
    z=base.merge(c[['pid','time','grex_single_iv','iv_dosage']].rename(columns={
        'time':'previous_time','grex_single_iv':'GReX_lag1'}),
        on=['pid','previous_time'],how='left',validate='many_to_one',sort=False)
    required_numeric=[roi,'prev__'+roi,'GReX_lag1', 'age_c','age_c2','age_c3','sex_num','IQR']+pcs
    for field in required_numeric+['iv_dosage','time','previous_time']:
        z[field]=pd.to_numeric(z[field],errors='coerce')
    fitted=z[required_numeric].notna().all(axis=1)
    model_rows=z.loc[fitted].copy()
    model_n=int(len(model_rows));model_people=int(model_rows.pid.nunique())
    # Mean-imputed, fractional genotypes can be used for regression but should
    # never be assigned to 0/1/2 genotype categories in the displayed curves.
    d=pd.to_numeric(model_rows.iv_dosage,errors='coerce')
    hardcall=d.notna() & d.between(0,2) & np.isclose(d, np.rint(d), atol=1e-6)
    plotted=model_rows.loc[hardcall].copy()
    plotted['genotype']=np.rint(plotted.iv_dosage).astype(int)
    conflicts=plotted.groupby('pid').genotype.nunique()
    if (conflicts>1).any():
        raise ValueError(f'Inconsistent imputed dosage across MRI visits for {gene}: {int((conflicts>1).sum())} participants')
    if max_years is not None:
        plotted=plotted.loc[plotted.time.between(0,max_years)].copy()
    if plotted.empty:
        raise ValueError(f'No drawable model-eligible MRI pairs for {gene} {roi}. Check time window.')
    donors=plotted.groupby('pid').genotype.first()
    n={i:int((donors==i).sum()) for i in (0,1,2)}
    N=sum(n.values())
    af=(n[1]+2*n[2])/(2*N) if N else np.nan
    maf=min(af,1-af) if np.isfinite(af) else np.nan
    audit=dict(gene=gene,lead_snp=snp,artifact_id=aid,roi=roi,
               model_valid_rows=model_n,model_valid_people=model_people,
               plotted_rows=int(len(plotted)),plotted_people=int(N),
               excluded_nonhardcall_model_rows=int(len(model_rows)-int(hardcall.sum())),
               max_years_display=max_years,
               dosage0_n=n[0],dosage1_n=n[1],dosage2_n=n[2],plotted_sample_maf=maf,
               previous_time_min=float(plotted.previous_time.min()),
               previous_time_max=float(plotted.previous_time.max()),
               current_time_min=float(plotted.time.min()),
               current_time_max=float(plotted.time.max()))
    return plotted,audit


def build_current_visit_display(mri_mod, mri_file, root, analysis_name, row, eligible_pairs, max_years):
    """Plot observed ROI and CURRENT-visit frozen tGReX; lag-1 regression unchanged.

    Eligible subjects are obtained from the original lag-1 model's covariate,
    previous-visit GReX and outcome-valid observations. At baseline we add their
    observed MRI and supported GReX values for visualization (not to the fit).
    """
    gene, snp, aid, roi, _ = row
    people=eligible_pairs[['pid','genotype']].drop_duplicates()
    conflict=people.groupby('pid').genotype.nunique()
    if (conflict>1).any(): raise ValueError('Conflicting genotype calls for plotted participants')
    people=people.drop_duplicates('pid')
    mri=pd.read_csv(mri_file,usecols=['PATNO','time',roi],low_memory=False)
    mri['pid']=mri_mod.txtid(mri.PATNO)
    mri['time']=pd.to_numeric(mri.time,errors='coerce')
    mri[roi]=pd.to_numeric(mri[roi],errors='coerce')
    mri=mri[['pid','time',roi]].dropna()
    mri=mri[mri.time.ge(0)].copy()
    if max_years is not None: mri=mri[mri.time.le(max_years)].copy()
    if mri.duplicated(['pid','time']).any():
        raise ValueError(f'Duplicate MRI visit in source for {roi}')
    mri=mri.merge(people,on='pid',how='inner',validate='many_to_one')
    mri=mri.rename(columns={roi:'value'})

    path=root/'single_iv_causal'/analysis_name/'single_iv_genetic_component'/f'{aid}.parquet'
    comp=read_component(path)
    comp=comp[comp.cohort.astype(str).eq('PP')].copy()
    for col,value in [('gene',gene),('lead_snp',snp),('artifact_id',aid)]:
        if col in comp: comp=comp[comp[col].astype(str).eq(value)].copy()
    comp['pid']=mri_mod.txtid(comp.participant_id)
    comp['time']=pd.to_numeric(comp.time,errors='coerce')
    comp['grex_single_iv']=pd.to_numeric(comp.grex_single_iv,errors='coerce')
    # Figure only: use all ALREADY-CACHED frozen predictions at observed MRI
    # visits, including those marked outside the Stage-I follow-up domain. This
    # does NOT change model_rows(), MRI inference, or the original GReX values.
    # Outside-support predictions are extrapolations; audit them explicitly.
    comp['outside_stage1_support']=~support_bool(comp.within_stage1_followup_support)
    comp=comp.dropna(subset=['pid','time','grex_single_iv'])
    comp=comp[comp.time.ge(0)].copy()
    if max_years is not None: comp=comp[comp.time.le(max_years)].copy()
    if comp.duplicated(['pid','time']).any():
        raise ValueError(f'Duplicate frozen current-visit GReX for {gene}')
    current=comp[['pid','time','grex_single_iv','outside_stage1_support']].merge(
        mri[['pid','time','genotype']],on=['pid','time'],how='inner',validate='one_to_one')
    current=current.rename(columns={'grex_single_iv':'value'})
    if mri.empty or current.empty:
        raise ValueError(f'No MRI or frozen current-visit GReX to draw for {gene}')
    if current.time.max() + 1e-8 < mri.time.max():
        warnings.warn(f'{gene}: last observed MRI visit t={mri.time.max():g} is later than '
                      f'last cached tGReX visit t={current.time.max():g}. '
                      'No predictions are invented; inspect the cache.')
    return mri,current


def bin_summary(z, bin_years):
    x=z[['pid','genotype','time','value']].dropna().copy()
    x['bin']=np.floor(x.time.to_numpy(float)/bin_years+0.5)*bin_years
    # One vote per participant/bin; do not allow repeat scans to inflate precision.
    x=x.groupby(['pid','genotype','bin'],as_index=False).value.mean()
    q=x.groupby(['genotype','bin']).value.agg(['mean','std','count']).reset_index()
    q['se']=q['std']/np.sqrt(q['count'])
    return q


def limits_from_summary(summary,min_bin_n,include_zero=False):
    use=summary[summary['count'].ge(min_bin_n)].copy()
    if use.empty:return None
    # No confidence interval/shading: scale from visible GROUP MEANS only.
    vals=use['mean'].to_numpy(float)
    lo=float(np.min(vals));hi=float(np.max(vals))
    median=float(np.median(vals))
    span=max(hi-lo,0.025*max(abs(median),0.12))
    mid=(lo+hi)/2
    lo=mid-span*.61;hi=mid+span*.61
    if include_zero:lo=min(lo,0.0);hi=max(hi,0.0)
    return lo,hi


def draw_axis(ax,data,min_bin_n,bin_years,ylim_zero=False):
    q=bin_summary(data,bin_years)
    n={k:int((data[['pid','genotype']].drop_duplicates().genotype==k).sum())
       for k in (0,1,2)}
    for k in (0,1,2):
        v=q[(q.genotype.eq(k)) & (q['count'].ge(min_bin_n))].sort_values('bin')
        if len(v)==0:continue
        x=v['bin'].to_numpy(float); y=v['mean'].to_numpy(float)
        # Descriptive group means only, no SEM band or CI.
        # No fitted cubic, spline, or artificially extended visit points.
        ax.plot(x,y,color=COLORS[k],lw=2.0,marker='o',markersize=3.8,zorder=3,
                markeredgecolor='white',markeredgewidth=.4)
    lim=limits_from_summary(q,min_bin_n,include_zero=ylim_zero)
    if lim:ax.set_ylim(*lim)
    ax.spines[['top','right']].set_visible(False)
    ax.spines['left'].set_color('#4D5966')
    ax.spines['bottom'].set_color('#4D5966')
    ax.tick_params(labelsize=8.2,colors='#283341',direction='out',length=3)
    ax.grid(axis='y',color='#DDE3E9',lw=.65)
    ax.set_axisbelow(True)
    return n,q


def make_panel(fig, top_pos, bottom_pos, data, hit, audit, title, letter,
               alleles, min_bin_n, bin_years, max_years, text_offsets):
    roi_data,grex_data=data
    top=fig.add_axes(top_pos)
    bot=fig.add_axes(bottom_pos,sharex=top)
    n,roi_summary=draw_axis(top,roi_data,min_bin_n,bin_years)
    _,grex_summary=draw_axis(bot,grex_data,min_bin_n,bin_years,ylim_zero=True)
    x_upper=(max_years if max_years is not None else max(roi_data.time.max(),grex_data.time.max()))
    x_upper=max(1.,float(x_upper))
    top.set_xlim(0,x_upper)
    top.tick_params(labelbottom=False)
    top.set_ylabel('MRI ROI–PC1',fontsize=9.4,labelpad=8)
    bot.set_ylabel('Predicted\ntGReX(t)',fontsize=9.1,labelpad=8)
    bot.set_xlabel('Years since baseline',fontsize=9.2,labelpad=5)
    bot.set_xticks(np.arange(0,np.floor(x_upper)+.01,max(1,int(np.ceil(x_upper/6)))))
    # A compact width below 0.6 needs a real two-line title, a two-line
    # annotation and a two-row legend. We deliberately leave all plotted
    # means, axes coordinates, colours and statistical estimates unchanged.
    x0,y0,w,h=top_pos
    scale=fig.get_size_inches()[0] / 7.4
    text_x = 0.12 if scale < 0.60 else x0
    compact=scale<0.60
    very_compact=scale<0.40
    title_y=y0+h+text_offsets[0]
    subtitle_y=y0+h+text_offsets[1]
    stat_y=y0+h+text_offsets[2]

    if compact:
        # The first row identifies the IV, the second names the ROI.
        family=f'{letter}  {audit["gene"]} | {audit["lead_snp"]}\n{title}'
        if very_compact: family=family.replace(' | ROI ', '\nROI ')
        title_fs=7.15 if not very_compact else 6.45
        title_y=y0+h+(0.082 if very_compact else 0.075)
    else:
        family=f'{letter}  {audit["gene"]}  |  {audit["lead_snp"]}  |  {title}'
        title_fs=min(10.3,8.75*(fig.get_size_inches()[0]/5.18))

    fig.text(text_x,title_y,family,ha='left',va='center',fontsize=title_fs,
             weight='semibold',color='#152231',linespacing=1.08)
    maf=audit['plotted_sample_maf']
    maftxt=f'MAF={maf:.3f}' if np.isfinite(maf) else 'MAF=NA'
    ale=f'{alleles[0]}/{alleles[1]}' if alleles else 'unavailable'
    if compact:
        subtitle=(f'PPMI plotted-sample {maftxt}\n'
                  f'PLINK A1/A2={ale} | dosage allele not inferred')
        subtitle_fs=6.4 if not very_compact else 5.7
        subtitle_y=y0+h+0.036
    else:
        subtitle=f'PPMI plotted-sample {maftxt}  |  PLINK A1/A2={ale}  |  dosage allele not inferred'
        subtitle_fs=7.0
    fig.text(text_x,subtitle_y,subtitle,ha='left',va='center',fontsize=subtitle_fs,
             color='#506275',linespacing=1.05)
    stat_text=(f'Lag-1 β={float(hit.beta_grex_lag1):+.4f} | '
               f'P={float(hit.p_cluster):.2g} | BH-FDR={float(hit.q_global_BH):.3g}')
    if compact:
        stat_y=y0+h+0.011
        stat_fs=6.7 if not very_compact else 5.7
    else:
        stat_fs=7.8
    fig.text(text_x,stat_y,stat_text,ha='left',va='center',fontsize=stat_fs,color='#344357')

    handles=[Line2D([0],[0],marker='o',markersize=4,linestyle='-',lw=1.9,color=COLORS[k],
                    label=f'Dosage {k}  (n={audit[f"dosage{k}_n"]})') for k in (0,1,2)]
    legend_ncol=(1 if very_compact else 2) if compact else 3
    legend_fs=(6.1 if not very_compact else 5.4) if compact else 7.1
    top.legend(handles=handles,loc='upper left',bbox_to_anchor=(0,1.0),borderaxespad=.15,
               ncol=legend_ncol,frameon=False,fontsize=legend_fs,
               handlelength=.9 if compact else 1.12,columnspacing=.65 if compact else .8)
    return top,bot,roi_summary,grex_summary


def save_plot(stem, samples, outdir, letters, max_years,bin_years,min_bin_n,
              width_scale=0.45):
    # Two Figure-2-style stacked examples. The axes have explicitly reserved
    # title / data / annotation regions; no tight_layout on individual axes.
    both=len(samples)>1
    # Original width 7.4 in; default 0.45 => 3.33 in (55% narrower).
    # Height and original axes positions remain unchanged.
    figure_width = 7.4 * width_scale
    # Keep approximately 0.72in left margin for vertical y-axis text at
    # narrow scales; otherwise clipping may occur below width-scale=0.55.
    ax_left=max(.17, .72/figure_width) if width_scale < .60 else .17
    ax_width=.95-ax_left
    if both:
        fig=plt.figure(figsize=(figure_width,9.4),facecolor='white')
        bottom_c_y = .113 if width_scale < .60 else .083
        positions=[((ax_left,.69,ax_width,.158),(ax_left,.535,ax_width,.108)),
                   ((ax_left,.236,ax_width,.158),(ax_left,bottom_c_y,ax_width,.108))]
    else:
        fig=plt.figure(figsize=(figure_width,5.6),facecolor='white')
        positions=[((ax_left,.51,ax_width,.245),(ax_left,.165,ax_width,.21))]
    manifest=[]
    for (z,audit,hit,title,alleles),letter,(toppos,botpos) in zip(samples,letters,positions):
        cur,grex=build_current_visit_display(
            _mri_module, _mri_file, _root, _analysis_name,
            (audit['gene'],audit['lead_snp'],audit['artifact_id'],audit['roi'],title),z,max_years)
        ax,ax2,means,grexmeans=make_panel(fig,toppos,botpos,(cur,grex),hit,audit,title,letter,
                                        alleles,min_bin_n,bin_years,max_years,
                                        (.058,.037,.017) if both else (.095,.063,.030))
        audit['current_visit_MRI_rows']=len(cur)
        audit['current_visit_tGReX_rows']=len(grex)
        audit['tGReX_outside_stage1_support_rows']=int(grex['outside_stage1_support'].sum())
        audit['tGReX_outside_stage1_support_people']=int(grex.loc[grex.outside_stage1_support,'pid'].nunique())
        audit['observed_MRI_time_min']=float(cur.time.min())
        audit['display_tGReX_time_max']=float(grex.time.max())
        audit['MRI_bins_displayed']=int((means['count']>=min_bin_n).sum())
        audit['GReX_bins_displayed']=int((grexmeans['count']>=min_bin_n).sum())
        if audit['tGReX_outside_stage1_support_rows']:
            print(f'[DISPLAY EXTRAPOLATION] {audit["gene"]}: '
                  f'{audit["tGReX_outside_stage1_support_rows"]} cached GReX rows outside Stage-I '
                  'follow-up support shown for trajectory completeness; these rows remain excluded '
                  'from the unchanged lag-1 MRI fit.',flush=True)
        manifest.append(audit)
    if both:
        # Footnote wraps according to page width; no content is removed.
        if width_scale < .60:
            foot=('MRI ROI–PC1: observed group means.\n'
                  'tGReX(t): frozen genetic component at MRI visits.\n'
                  'Curves are descriptive (no CI).\n'
                  'β/P/FDR: unchanged lag-1 ARX.\n'
                  'Late tGReX may be extrapolated.')
            foot_fs=6.0 if width_scale >= .40 else 5.4
        else:
            foot=('MRI ROI–PC1: observed group means. tGReX(t): frozen genetic component\n'
                  'at observed MRI visits. Curves are descriptive (no CI); β/P/FDR from\n'
                  'unchanged lag-1 ARX. Late tGReX may be extrapolated.')
            foot_fs=6.5
        fig.text(.12 if width_scale<.60 else .17,.008,foot,fontsize=foot_fs,color='#5B6E83',ha='left',
                 va='bottom',linespacing=1.0)
    else:
        fig.text(.12 if width_scale<.60 else .17,.035,
                 'Observed MRI and frozen current-visit tGReX;\n'
                 'lag-1 inference unchanged.',
                 fontsize=6.9 if width_scale>=.60 else 5.8,
                 color='#5B6E83',ha='left',va='bottom',linespacing=1.0)
    # Preserve editable text in PDF/SVG; Cairo PNG + pathlib.Path can fail.
    for ext in ('pdf','svg','png'):
        target=str(outdir / f'{stem}.{ext}')
        backend={'pdf':'pdf','svg':'svg','png':'agg'}[ext]
        fig.savefig(target,format=ext,backend=backend,
                    dpi=450 if ext=='png' else None,facecolor='white')
    plt.close(fig)
    return manifest


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('--root',default=DEFAULT_ROOT)
    pa.add_argument('--mri-file',default=DEFAULT_MRI)
    pa.add_argument('--hit-file',default='')
    pa.add_argument('--outdir',default='')
    pa.add_argument('--analysis-name',default='mri_dbm_lag1')
    pa.add_argument('--pc-file',default='')
    pa.add_argument('--pc-manifest',default='')
    pa.add_argument('--bim',default='')
    pa.add_argument('--bin-years',type=float,default=1.0)
    pa.add_argument('--max-years',type=float,default=4.0,
                    help='Maximum current MRI followup years displayed; model stays unchanged')
    pa.add_argument('--min-bin-n',type=int,default=3)
    pa.add_argument('--width-scale',type=float,default=0.45,
                    help='Figure width relative to original 7.4 in; 0.45 is 55%% narrower (default: 0.45)')
    args=pa.parse_args()
    if not 0.30 <= args.width_scale <= 1.0:
        raise ValueError('width-scale must be between 0.30 and 1.00')
    if args.bin_years<=0 or args.max_years<=0 or args.min_bin_n<2:
        raise ValueError('bin-years and max-years must be positive; min-bin-n>=2')
    root=Path(args.root)
    code=Path(__file__).resolve().parents[2]
    source=load_mri_analysis_script(code)
    hits_path=Path(args.hit_file) if args.hit_file else (
        root/'single_iv_causal'/args.analysis_name/'pc1'/'MRI_DBM_lag1_global_FDR05.csv')
    if not hits_path.is_file():raise FileNotFoundError('The actual MRI global-FDR file is missing: '+str(hits_path))
    hits=pd.read_csv(hits_path,low_memory=False)
    outcome=Path(args.mri_file)
    if not outcome.is_file():raise FileNotFoundError('MRI outcome file missing: '+str(outcome))
    out=Path(args.outdir) if args.outdir else root/'manuscript_figures'/'tqtl_benchmark_v4'/'mri_dbm_examples'
    out.mkdir(parents=True,exist_ok=True)
    pc,pcs=source.get_pcs(root,args.pc_file,args.pc_manifest)
    base,rois=source.prepare_mri(outcome,pc,pcs,1,1e-6)
    bim=Path(args.bim) if args.bim else root/'single_iv_causal'/args.analysis_name/'genotype'/'single_iv_snps.bim'
    wanted={t[1] for t in TARGETS}
    alleles=load_bim_alleles(bim,wanted)
    for extra in [root/'single_iv_causal'/'updrs3'/'genotype'/'single_iv_snps.bim',
                  root/'data'/'genotype'/'PP'/'amp_pd_PP_qc.bim']:
        missing=wanted-set(alleles)
        if missing and extra.exists():
            alleles.update(load_bim_alleles(extra,missing))
    missing=wanted-set(alleles)
    if missing:warnings.warn(f'No actual PLINK BIM alleles found for {sorted(missing)}; not inferring from dosage')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                         'axes.linewidth':.85,'savefig.dpi':450})
    # Scoped figure sources used by save_plot for display-only current visits.
    global _mri_module, _mri_file, _root, _analysis_name
    _mri_module=source; _mri_file=outcome; _root=root; _analysis_name=args.analysis_name
    samples=[];audit=[]
    for gene,snp,aid,roi,roi_title in TARGETS:
        hit=read_exact_hit(hits,gene,snp,aid,roi)
        z,info=model_rows(source,base,pcs,(gene,snp,aid,roi,roi_title),root,args.analysis_name,args.max_years)
        a1a2=alleles.get(snp)
        info.update({'A1':a1a2[0] if a1a2 else None,'A2':a1a2[1] if a1a2 else None,
                     'beta_grex_lag1':float(hit.beta_grex_lag1),
                     'p_cluster':float(hit.p_cluster), 'q_global_BH':float(hit.q_global_BH),
                     'reported_regression_n_people':int(hit.n_people),
                     'reported_regression_n_obs':int(hit.n_obs),
                     'display_bin_years':args.bin_years,'min_bin_n':args.min_bin_n,
                     'outcome_source':str(outcome),'hits_source':str(hits_path),
                     'component_source':str(root/'single_iv_causal'/args.analysis_name/'single_iv_genetic_component'/f'{aid}.parquet')})
        if info['model_valid_people']!=info['reported_regression_n_people']:
            warnings.warn(f'{gene}: eligible {info["model_valid_people"]} participants vs main fit {info["reported_regression_n_people"]}; '
                          'model design rank/sample validation or data version may differ. Inspect audit before publication.')
        print('[EXAMPLE]',gene,roi,'model_people=',info['model_valid_people'],
              'plotted=',info['plotted_people'],'dosage n=',
              [info[f'dosage{i}_n'] for i in (0,1,2)],
              'MAF=',round(info['plotted_sample_maf'],3),
              'A1/A2=',a1a2,'beta=',info['beta_grex_lag1'],flush=True)
        samples.append((z,info,hit,roi_title,a1a2));audit.append(info)
    save_plot('Figure_MRI_DBM_FLOT1_C7orf25',samples,out,['b','c'],
              args.max_years,args.bin_years,args.min_bin_n,width_scale=args.width_scale)
    for item,label in zip(samples,['b','c']):
        z,info,hit,roi_title,a1a2=item
        save_plot(f'{info["gene"]}_{info["lead_snp"]}_{info["roi"]}',[item],out,[label],
                  args.max_years,args.bin_years,args.min_bin_n,width_scale=args.width_scale)
    pd.DataFrame(audit).to_csv(out/'MRI_DBM_example_genotype_and_model_audit.csv',index=False)
    (out/'MRI_example_figure_manifest.json').write_text(json.dumps({
        'description':'Two descriptive MRI lag-1 IV-gene examples; full frozen prediction cache for visualization only',
        'model':'DBM_current ~ GReX_previous + DBM_previous + age_cubic + sex + genotype_PCs + IQR',
        'notes':['Outcome and supported GReX plotted at current visits starting at year 0; regression is still lag-1',
                 'Binned participant means are descriptive; no SEM bands or adjusted model effects',
                 'For display only, already-cached frozen GReX outside Stage-I followup support is shown as extrapolation to match MRI time range; unchanged MRI inference excludes it',
                 'ROI-PC1 sign does not identify atrophy versus hypertrophy',
                 'Dosage 0/1/2 orientation cannot be inferred from BIM A1/A2'],
        'result_table':str(hits_path), 'figure_width_scale':args.width_scale,
        'figure_width_inches':7.4*args.width_scale, 'examples':audit},indent=2,ensure_ascii=False))
    print('[DONE] Figure/individual panels and audit saved:',out,flush=True)

if __name__=='__main__':main()
