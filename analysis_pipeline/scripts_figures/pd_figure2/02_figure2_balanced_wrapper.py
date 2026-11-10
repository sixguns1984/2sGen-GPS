#!/usr/bin/env python3
"""Run the original 2sGenGPS Figure 2 maker, then render a non-stretched 2x2 main figure.

This is a non-invasive overlay: the installed 01_make_pd_figure2.py is not edited.
Handles Matplotlib's Figure.savefig / pyplot.savefig, and saves an additional file
with suffix '_balanced' in the same output directory.
"""
import argparse
import math
import os
from pathlib import Path
import re
import runpy
import sys
import traceback
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure

MAIN_STEM = 'Figure2_PD_temporal_association_main'
VARIANT_RX = re.compile(r'(rs\d+|chr(?:[0-9XYMT]+):\d+:[ACGT]+:[ACGT]+)', re.I)
COUNT_RX = re.compile(r'(?:dosage|genotype)\s*([012])\s*\(\s*n\s*=\s*([\d,]+)', re.I)


def count_maf(labels):
    """MAF from *plotted* genotype count labels, NOT population-wide WGS MAF."""
    found = {}
    for s in labels:
        m = COUNT_RX.search(str(s))
        if m:
            found[int(m.group(1))] = int(m.group(2).replace(',', ''))
    if not found or 0 not in found or 1 not in found or 2 not in found:
        return None
    n0, n1, n2 = [found[x] for x in (0, 1, 2)]
    N = n0 + n1 + n2
    if N == 0:
        return None
    af = (n1 + 2 * n2) / (2 * N)
    return {'maf': min(af, 1-af), 'n': N, 'n0': n0, 'n1': n1, 'n2': n2}


def read_bim_alleles(prefix, variants):
    """Stream a large BIM once. Do NOT assume its A1 corresponds to plotted dosage-1."""
    if not prefix or not variants:
        return {}
    bim = Path(str(prefix)).with_suffix('.bim')
    if not bim.exists():
        return {}
    wanted = set(variants)
    alleles = {}
    with bim.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6 and parts[1] in wanted:
                alleles[parts[1]] = (parts[4], parts[5])
                if len(alleles) == len(wanted):
                    break
    return alleles


def read_freq_mafs(freq_path, variants):
    """Read PLINK --freq output when supplied. Values are PPMI/PLINK MAF, not plotted subset."""
    if not freq_path:
        return {}
    p = Path(freq_path)
    if not p.exists():
        warnings.warn(f'Frequency file absent: {p}')
        return {}
    import pandas as pd
    data = pd.read_csv(p, sep=r'\s+', dtype=str)
    id_col = next((x for x in ['ID','SNP','Variant_ID'] if x in data),None)
    maf_col = next((x for x in ['MAF','ALT_FREQS','ALT_FREQ','A1_FREQ'] if x in data),None)
    out = {}
    if not id_col or not maf_col:
        warnings.warn('Could not find ID and frequency fields in --freq file.')
        return out
    for row in data.loc[data[id_col].isin(set(variants)), [id_col,maf_col]].itertuples(index=False, name=None):
        try:
            f = float(str(row[1]).split(',')[0])
            if math.isfinite(f) and 0<=f<=1:
                out[str(row[0])] = min(f,1-f)
        except (TypeError,ValueError):
            pass
    return out


def discover_main_axes(fig):
    axes = [ax for ax in fig.axes if ax.get_visible()]
    if len(axes) < 10:
        raise RuntimeError(f'Expected >=10 visible axes (a + b/c/d x 3), found {len(axes)}')
    # A occupies most of the left column in the original figure.
    left = sorted(axes, key=lambda a: (a.get_position().x0, -a.get_position().height))
    axis_a = next((a for a in left if '-log' in a.get_ylabel().lower().replace(' ', '')),
                  min(left, key=lambda a: a.get_position().x0))
    others = [ax for ax in axes if ax is not axis_a]
    # Ignore optional tiny colorbar or inset axes by matching original right-side width.
    right = [a for a in others if a.get_position().x0 >= axis_a.get_position().x1 - 0.06]
    if len(right) != 9:
        # If there are exactly nine non-A axes, use those; otherwise reject before touching the plot.
        if len(others) == 9:
            right = others
        else:
            raise RuntimeError(f'Could not identify 9 example axes; axes={len(axes)}, right={len(right)}')
    right = sorted(right, key=lambda a: a.get_position().y0 + a.get_position().height/2, reverse=True)
    groups = [sorted(right[k:k+3], key=lambda a:a.get_position().y0, reverse=True) for k in (0,3,6)]
    return axis_a, groups


def max_observed_time(ax):
    result=[]
    for line in ax.lines:
        try:
            x=np.asarray(line.get_xdata(),dtype=float)
            y=np.asarray(line.get_ydata(),dtype=float)
            if len(x)==len(y) and len(x)>=2:
                ok=np.isfinite(x)&np.isfinite(y)
                if np.any(ok):result.extend(x[ok].tolist())
        except (ValueError,TypeError):
            continue
    return max(result) if result else None


def extract_title_and_variant(fig, ax, group_index, requested_examples):
    source = ' '.join(t.get_text() for t in fig.texts)
    source += ' '.join(a.get_title() for a in fig.axes)
    # Try a local panel title first, then nearby figure-level title.
    labels = []
    for item in fig.texts:
        txt=item.get_text()
        if txt and ((chr(ord('b')+group_index)+' ') in txt[:3] or (chr(ord('b')+group_index)+' |') in txt[:5]):
            labels.append(txt)
    if not labels:
        labels=[ax.get_title()]
    label=labels[0] if labels and labels[0] else ''
    snp=VARIANT_RX.search(label)
    if not snp:
        # find by gene in global text, or avoid inventing a variant
        gene=requested_examples[group_index].split(':')[0] if group_index<len(requested_examples) else ''
        for t in fig.texts:
            if gene and gene.lower() in t.get_text().lower():
                m=VARIANT_RX.search(t.get_text())
                if m:
                    snp=m;break
    return label, snp.group(0) if snp else None


def find_legend_labels(ax):
    labels=[]
    if ax.get_legend() is not None:
        labels += [t.get_text() for t in ax.get_legend().get_texts()]
    labels += [l.get_label() for l in ax.lines]
    return labels


def balanced_layout(fig, bim_prefix=None, freq_file=None, requested_examples=None):
    requested_examples=requested_examples or []
    axis_a, groups = discover_main_axes(fig)
    original_texts=[t.get_text() for t in fig.texts]
    metas=[]
    for gi,g in enumerate(groups):
        title, var = extract_title_and_variant(fig, g[0], gi, requested_examples)
        cnt=count_maf(find_legend_labels(g[0]))
        metas.append({'label':title,'variant':var,'count':cnt})
    varids=[m['variant'] for m in metas if m['variant']]
    alleles=read_bim_alleles(bim_prefix,varids)
    mafs=read_freq_mafs(freq_file,varids)

    # Clip only the displayed GReX panel; no extrapolated expression beyond available Y follow-up.
    time_logs=[]
    for meta,group in zip(metas,groups):
        phen_ax, grex_ax, lag_ax=group
        tmax=max_observed_time(phen_ax)
        if tmax is not None:
            xmin=min(phen_ax.get_xlim()[0], grex_ax.get_xlim()[0], 0)
            phen_ax.set_xlim(xmin,tmax+0.12)
            grex_ax.set_xlim(xmin,tmax+0.12)
        time_logs.append(tmax)

    # 2x2 arrangement: fixed short panel a, three balanced example tiles.
    fig.set_size_inches(17.2,10.8,forward=True)
    axis_a.set_position([.075,.595,.390,.305])
    axes_xy=[(.565,.55),(.075,.085),(.565,.085)]
    width,height=.395,.375
    for group,(x,y) in zip(groups,axes_xy):
        top, middle, bottom=group
        top.set_position([x,y+height*.53,width,height*.38])
        middle.set_position([x,y+height*.235,width,height*.19])
        bottom.set_position([x,y+height*.025,width,height*.115])
        # Avoid deep upper x tick labels on stacked panel axes.
        # Keep the original axes tick visibility/styles unchanged.

    # Hide previous figure-positioned labels without destroying them, so plotting can
    # be restored before original Figure.save calls for PDF/SVG/PNG.
    for artist in list(fig.texts):
        artist.set_visible(False)
    fig.text(.52,.978,'Temporal IV–gene–phenotype associations',ha='center',va='top',fontsize=15,fontweight='bold')
    axis_a.set_title('a   A-tier associations',fontsize=11,fontweight='bold',loc='left',pad=12)
    for gi,(group,meta,(x,y)) in enumerate(zip(groups,metas,axes_xy)):
        title=meta['label']
        if not title and gi<len(requested_examples):
            gene,phenotype=requested_examples[gi].split(':',1)
            title=f'{gene} | {phenotype.upper()}'
        title=re.sub(r'^[bcd]\s+', '',title).strip()
        snp=meta['variant']
        if snp and snp not in title:
            title+=f' | {snp}'
        annotations=[]
        if snp in alleles:
            a1,a2=alleles[snp]
            annotations.append(f'alleles {a1}/{a2}')
        elif snp:
            annotations.append('alleles: unavailable')
        if snp in mafs:
            annotations.append(f'PPMI MAF={mafs[snp]:.3f}')
        elif meta['count']:
            annotations.append(f'plotted-subset MAF={meta["count"]["maf"]:.3f}')
        else:
            annotations.append('MAF: unavailable')
        if meta['count'] and meta['count']['n2']<5:
            annotations.append('minor-homozygote n<5')
        group[0].set_title(chr(ord('b')+gi)+'   '+title+'\n'+ ' | '.join(annotations),fontsize=9.1,fontweight='medium',loc='left',pad=8)
        print(f'[BALANCED PANEL {chr(ord("b")+gi)}] {title}; '+', '.join(annotations)+f'; phenotype Tmax={time_logs[gi]}')
    fig.text(.50,.031,'Outcome and GReX trajectories are restricted to observed outcome follow-up; allele labels from PLINK BIM (orientation not presumed).',ha='center',va='center',fontsize=8,color='#59687A')
    return {'panels':metas,'time_max':time_logs,'original_texts':original_texts}


def parse_wrapper_args(argv):
    p=argparse.ArgumentParser(description=__doc__,add_help=True)
    p.add_argument('--original-script', default=None)
    p.add_argument('--genotype-bfile',default=None,help='PLINK prefix for allele A1/A2 lookup')
    p.add_argument('--freq-file',default=None,help='Existing PLINK .afreq or .frq for PPMI MAF')
    p.add_argument('--story-examples',nargs='+',default=['PHLPP1:updrs3','INTS6:updrs3','CCZ1B:moca'])
    p.add_argument('--output-suffix',default='_balanced')
    p.add_argument('--ranked-candidates',default=None,help='Use top validated high-MAF A-tier candidate per phenotype from 03_rank_high_maf_examples.py')
    p.add_argument('--require-alleles',action='store_true',help='Raise if allele metadata is unavailable (publication safety)')
    args,unknown=p.parse_known_args(argv)
    return args,unknown


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    args,remaining=parse_wrapper_args(argv)
    expected_snps=[]
    if args.ranked_candidates:
        import pandas as pd
        ranking=pd.read_csv(args.ranked_candidates)
        chosen=[]
        chosen_rows=[]
        seen=set()
        for ph in ('updrs3','updrs2','moca'):
            z=ranking.loc[ranking.pheno_norm.astype(str).eq(ph)].sort_values('rank_score',ascending=False)
            for _,r in z.iterrows():
                key=(str(r['gene']),str(r['lead_snp']))
                if key in seen:continue
                chosen.append(str(r['story_example']))
                chosen_rows.append(r)
                seen.add(key)
                break
        if len(chosen)!=3:
            raise SystemExit(f'Only {len(chosen)} non-overlapping high-MAF examples found for updrs3/updrs2/moca; inspect ranking CSV before plotting')
        args.story_examples=chosen
        expected_snps=[str(r['lead_snp']) for r in chosen_rows]
        print('[FIG2 SELECTED REAL EXAMPLES]',' '.join(chosen))
    folder=Path(__file__).resolve().parent
    original=Path(args.original_script) if args.original_script else folder/'01_make_pd_figure2.py'
    if not original.is_file():
        raise SystemExit(f'Original plot script not found: {original}; install beside 01_make_pd_figure2.py')
    if '01_make_pd_figure2.py' not in original.name:
        warnings.warn(f'Plot entrypoint differs from expected: {original}')
    if args.genotype_bfile is None:
        root=None
        if '--root' in remaining:
            try:root=Path(remaining[remaining.index('--root')+1])
            except (IndexError, ValueError):pass
        if root:
            prefix=root/'data/genotype/PP/amp_pd_PP_qc'
            if prefix.with_suffix('.bim').exists():args.genotype_bfile=str(prefix)
    original_save=Figure.savefig
    intercepted={'count':0}
    def intercept(self, fname,*aa,**kk):
        # Leave all original outputs unmodified.
        result=original_save(self,fname,*aa,**kk)
        if not isinstance(fname,(str,os.PathLike)) or MAIN_STEM not in Path(fname).stem or '_balanced' in Path(fname).stem:
            return result
        path=Path(fname)
        if path.suffix.lower() not in ('.pdf','.png','.svg'):
            return result
        try:
            # Snapshot all geometry/text, make the balanced variant and restore,
            # so that the *original* PDF and SVG will not accidentally inherit the
            # adjusted geometry when saved after the PNG.
            original_size=self.get_size_inches().copy()
            old_axes=[(ax,ax.get_position().frozen(),ax.get_xlim(),
                       {loc:(ax.get_title(loc=loc), title.get_fontsize(), title.get_fontweight(),title.get_color())
                        for loc,title in (('left',ax._left_title),('center',ax.title),('right',ax._right_title))})
                       for ax in self.axes]
            old_texts=list(self.texts)
            old_visible=[t.get_visible() for t in old_texts]
            try:
                metas=balanced_layout(self,args.genotype_bfile,args.freq_file,args.story_examples)
                if expected_snps:
                    for panel,expected in zip(metas['panels'],expected_snps):
                        plotted=panel.get('variant')
                        if plotted!=expected:
                            raise RuntimeError(f'Ranked locus requires SNP {expected} but original plot selected {plotted!r} for {panel.get("label","")}. This original plot script appears to select by gene/phenotype rather than exact variant: do not mislabel it. Pass a full gene+SNP selector to original plotting code or select a candidate whose SNP matches. Original figure retained.')
                if args.require_alleles and any(m['variant'] and m['variant'] not in read_bim_alleles(args.genotype_bfile,[m['variant']]) for m in metas['panels']):
                    raise RuntimeError('Missing one or more A1/A2 allele lookups')
                outfile=path.with_name(path.stem+args.output_suffix+path.suffix)
                original_save(self,outfile,*aa,**kk)
                intercepted['count']+=1
                print('[BALANCED OUTPUT]',outfile)
            finally:
                for txt in list(self.texts):
                    if txt not in old_texts:txt.remove()
                for txt,visible in zip(old_texts,old_visible):txt.set_visible(visible)
                for ax,pos,xlim,titles in old_axes:
                    ax.set_position(pos)
                    ax.set_xlim(xlim)
                    for loc,(title,fontsize,weight,color) in titles.items():
                        ax.set_title(title,loc=loc,fontsize=fontsize,fontweight=weight,color=color)
                self.set_size_inches(original_size,forward=True)
        except Exception:
            traceback.print_exc()
            raise
        return result
    Figure.savefig=intercept
    orig_argv=[str(original)]+remaining+['--story-examples']+list(args.story_examples)
    sys.argv=orig_argv
    # The previous script expects story-examples flags, but genotype-bfile is not added
    # automatically unless it came from original `remaining`.
    try:
        runpy.run_path(str(original),run_name='__main__')
    finally:
        Figure.savefig=original_save
    if not intercepted['count']:
        raise SystemExit('No main Figure2 file was intercepted; inspect the original script save calls.')

if __name__=='__main__':main()
