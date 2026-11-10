#!/usr/bin/env python3
"""2sGenGPS postmortem Fig.3F: edgeR-QLF ALT-carrier DEG-count heatmaps.

Read only the finalized significant-edgeR CSV; never re-run differential expression.
If available, use edgeR fit_status.csv or group summary to discriminate a tested
comparison with 0 significant genes from a skipped/unfinished comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use('cairo')  # Use Cairo backend for high-quality output
matplotlib.rcParams.update({
    'pdf.fonttype': 42,
    'pdf.use14corefonts': True,  # use standard Helvetica fonts to avoid Adobe Illustrator DejaVu substitution
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'text.usetex': False,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'Liberation Sans', 'DejaVu Sans'],
    'axes.unicode_minus': False,
})
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import cm
import numpy as np
import pandas as pd

LOGGER = logging.getLogger('fig3f_edger')
REGIONS = ('DMNX', 'GPI', 'PVC', 'PMC', 'PFC')
CELL_TYPES = ('Exc', 'Inh', 'Oli', 'Immune', 'OPC', 'Ast', 'Vast')
SNPS = ('rs2253588', 'rs35162690')
REQUIRED = {'snp', 'region', 'cell_type', 'stratum', 'log2FC', 'padj'}
KEYS = ('snp', 'region', 'cell_type')
COLORS = {'up':'YlOrRd_r', 'down':'YlGnBu_r'}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_significant(path: Path, stratum: str = 'all') -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Final edgeR file not found: {path}')
    data = pd.read_csv(path, low_memory=False)
    absent = REQUIRED - set(data.columns)
    if absent:
        raise ValueError(f'Significant edgeR CSV missing required columns: {sorted(absent)}')
    if 'method' in data and not data['method'].astype(str).eq('edger_qlf').all():
        raise ValueError('Non-edgeR-QLF rows detected. Refusing to mix methods.')
    if 'contrast' in data and not data['contrast'].astype(str).eq('VCF_ALT_carrier_vs_noncarrier').all():
        raise ValueError('Unexpected contrast: log2FC must be VCF ALT carrier minus noncarrier.')
    data['log2FC'] = pd.to_numeric(data['log2FC'], errors='raise')
    data['padj'] = pd.to_numeric(data['padj'], errors='coerce')
    data = data[data['stratum'].astype(str).eq(stratum) & data['padj'].lt(0.05)].copy()
    if data.empty:
        raise ValueError(f'No {stratum}-stratum BH-FDR < 0.05 rows in provided edgeR CSV.')
    for c in KEYS:
        data[c] = data[c].astype(str).str.strip()
    data = data[data['snp'].isin(SNPS) & data['region'].isin(REGIONS) & data['cell_type'].isin(CELL_TYPES)].copy()
    if data.empty:
        raise ValueError('No rows match expected SNPs/regions/cell types')
    if 'gene_id' not in data.columns and 'gene_symbol' not in data.columns:
        raise ValueError('Expected gene_id or gene_symbol to audit duplicate genes')
    gene_col = 'gene_id' if 'gene_id' in data else 'gene_symbol'
    dupl_key = list(KEYS) + [gene_col]
    duplicated = data.duplicated(dupl_key, keep=False)
    if duplicated.any():
        duplicates = data.loc[duplicated, dupl_key + ['log2FC', 'padj']]
        if duplicates.groupby(dupl_key)['log2FC'].nunique(dropna=False).gt(1).any():
            raise ValueError('Duplicated SNP/region/cell/gene has conflicting log2FC. Review source file.')
        LOGGER.warning('Deduplicating %d repeated gene records', int(duplicated.sum()))
        data = data.drop_duplicates(dupl_key, keep='first')
    if not np.isfinite(data['log2FC']).all():
        raise ValueError('Non-finite log2FC detected in source CSV')
    if (data['log2FC'] == 0).any():
        LOGGER.warning('%d exact-zero log2FC entries excluded from up/down counts', int((data['log2FC']==0).sum()))
    return data


def detected_comparisons(significant_path: Path, stratum: str = 'all') -> dict:
    """Look for independent evidence a comparison was successfully tested."""
    root = significant_path.parent
    coverage = {}
    status_path = root / 'fit_status.csv'
    if status_path.is_file():
        status = pd.read_csv(status_path, low_memory=False)
        required = set(KEYS) | {'stratum','status'}
        if required.issubset(status.columns):
            status = status[status['stratum'].astype(str).eq(stratum)]
            for _, r in status.iterrows():
                key = tuple(str(r[c]).strip() for c in KEYS)
                s = str(r['status']).strip().upper()
                coverage[key] = {'tested': s in {'PASS', 'SUCCESS'}, 'source': 'fit_status.csv', 'status': s,
                                 'reason': str(r.get('reason', '')),
                                 'n_genes_tested': r.get('n_genes', np.nan)}
            LOGGER.info('Loaded %s coverage from %s (%d comparisons)', stratum, status_path, len(coverage))
        else:
            LOGGER.warning('Ignored malformed fit_status.csv: %s', status_path)
    summary_path = root / 'edgeR_QLF_group_summary.csv'
    if summary_path.is_file():
        sums = pd.read_csv(summary_path, low_memory=False)
        if set(KEYS).union({'stratum'}).issubset(sums.columns):
            for _, r in sums[sums['stratum'].astype(str).eq(stratum)].iterrows():
                key = tuple(str(r[c]).strip() for c in KEYS)
                # A group with a saved gene-level results file is necessarily tested.
                if key not in coverage:
                    coverage[key] = {'tested': True, 'status': 'PASS', 'source': 'edgeR_QLF_group_summary.csv',
                                     'reason': '', 'n_genes_tested': r.get('n_genes', np.nan)}
            LOGGER.info('Loaded coverage from %s', summary_path)
    return coverage


def create_counts(source: pd.DataFrame, coverage: dict, assume_complete: bool = False, stratum: str = 'all') -> pd.DataFrame:
    rows = []
    grp = source.groupby(list(KEYS), sort=False)
    count_lookup = {}
    for key, df in grp:
        count_lookup[key] = (int((df['log2FC'] > 0).sum()), int((df['log2FC'] < 0).sum()),
                             int((df['log2FC'] == 0).sum()),
                             df['n_carrier'].iloc[0] if 'n_carrier' in df else np.nan,
                             df['n_noncarrier'].iloc[0] if 'n_noncarrier' in df else np.nan,
                             df['n_genes_tested'].iloc[0] if 'n_genes_tested' in df else np.nan)
    for snp in SNPS:
        for cell in CELL_TYPES:
            for region in REGIONS:
                key = snp, region, cell
                info = coverage.get(key)
                sig = count_lookup.get(key)
                if info is not None:
                    tested = bool(info['tested'])
                    status = info['status']
                    reason = info.get('reason', '')
                    source_kind = info['source']
                    if not tested and sig is not None:
                        raise ValueError(f'{key}: FDR-significant genes exist but fit_status reports {status}.')
                elif sig is not None:
                    tested = True
                    status = 'SIGNIFICANT_ROWS_PRESENT'
                    reason = ''
                    source_kind = 'significant_edgeR_CSV'
                else:
                    tested = bool(assume_complete)
                    status = 'ASSUMED_PASS' if assume_complete else 'UNVERIFIED'
                    reason = 'No independent tested-comparison record' if not assume_complete else 'User explicitly assumed full coverage'
                    source_kind = 'manual_override' if assume_complete else 'none'
                up, down, zero, carriers, noncarriers, ntested = sig if sig is not None else (0, 0, 0, np.nan, np.nan, np.nan)
                if info is not None and pd.notna(info.get('n_genes_tested', np.nan)):
                    ntested = info['n_genes_tested']
                rows.append({
                    'snp': snp,'region': region,'cell_type': cell, 'stratum': stratum,
                    'contrast': 'VCF_ALT_carrier_vs_noncarrier', 'method': 'edger_qlf',
                    'n_up': up if tested else np.nan,
                    'n_down': down if tested else np.nan,
                    'n_total': up+down if tested else np.nan,
                    'n_zero_log2fc': zero if tested else np.nan,
                    'fit_status': status, 'fit_verified': bool(tested and status != 'ASSUMED_PASS'),
                    'source_of_fit_status': source_kind,
                    'reason': reason,
                    'n_carrier': carriers, 'n_noncarrier': noncarriers,
                    'n_genes_tested': ntested,
                })
    return pd.DataFrame(rows)


def mats(audit, snp):
    arr = {}
    for direction, key in (('up', 'n_up'), ('down', 'n_down')):
        z = audit[audit['snp'].eq(snp)].pivot(index='cell_type', columns='region', values=key)
        arr[direction] = z.reindex(index=CELL_TYPES, columns=REGIONS).to_numpy(dtype=float)
    return arr


def draw_heat(ax, counts, cmap, vmax):
    values = np.log2(counts + 1)
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad('#e7e9ed')
    img = ax.imshow(values, cmap=cmap, vmin=0, vmax=vmax, aspect='auto', interpolation='none')
    for row in range(len(CELL_TYPES)):
        for col in range(len(REGIONS)):
            value = counts[row,col]
            if np.isfinite(value):
                rgba = img.cmap(img.norm(math.log2(value + 1)))
                lum = float(0.2126*rgba[0] + 0.7152*rgba[1] + 0.0722*rgba[2])
                color = '#ffffff' if lum < 0.39 else '#111111'
                label = str(int(value))
            else:
                color = '#667085'
                label = '-'
            ax.text(col,row,label,ha='center',va='center',color=color,fontsize=8.8, zorder=10)
    ax.set_xticks(range(len(REGIONS)),REGIONS,fontsize=9.2)
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x',length=0,pad=5)
    ax.set_yticks([])
    ax.tick_params(left=False)
    ax.set_xticks(np.arange(-.5,len(REGIONS),1),minor=True)
    ax.set_yticks(np.arange(-.5,len(CELL_TYPES),1),minor=True)
    ax.grid(which='minor', color='white', linewidth=0.9)
    ax.tick_params(which='minor',left=False,bottom=False)
    for spine in ax.spines.values(): spine.set_visible(False)
    return img


def plot_one(audit: pd.DataFrame, snps: list[str], path: Path, dpi=300, stratum='all'):
    nrows = len(snps)
    fig = plt.figure(figsize=(12.0, 5.05 if nrows==1 else 8.7), facecolor='white')
    fig.suptitle('Postmortem SNP-associated differential expression',
                 x=.07, y=.988 if nrows==1 else .986, ha='left', fontsize=12, fontweight='bold')
    fig.text(.047, .987 if nrows==1 else .985, 'f', ha='left', va='top',fontsize=13.5,fontweight='bold')
    gs = fig.add_gridspec(nrows=nrows, ncols=3, width_ratios=[5,.69,5],
                          left=.067,right=.97,top=.71 if nrows==1 else .82,
                          bottom=.19 if nrows==1 else .115,
                          wspace=.04,hspace=.49 if nrows>1 else .15)
    maxlog = {}
    for direction,col in [('up','n_up'),('down','n_down')]:
        values = pd.to_numeric(audit[col],errors='coerce')
        maxlog[direction] = max(1, math.ceil(np.log2(values.max()+1))) if values.notna().any() else 1
    left_ax = right_ax = None
    missing = False
    for i,snp in enumerate(snps):
        m=mats(audit,snp)
        ax_l=fig.add_subplot(gs[i,0])
        ax_c=fig.add_subplot(gs[i,1])
        ax_r=fig.add_subplot(gs[i,2])
        draw_heat(ax_l,m['up'],COLORS['up'],maxlog['up'])
        draw_heat(ax_r,m['down'],COLORS['down'],maxlog['down'])
        ax_c.set_xlim(0,1)
        ax_c.set_ylim(len(CELL_TYPES)-.5,-.5)
        ax_c.axis('off')
        for j,ct in enumerate(CELL_TYPES):
            ax_c.text(.5,j,ct,ha='center',va='center',fontsize=8.9)
        ax_l.text(-.006,1.115,snp,transform=ax_l.transAxes,ha='left',va='bottom',
                  color='#2b3440',fontweight='bold',fontsize=10.5)
        missing |= np.isnan(m['up']).any() or np.isnan(m['down']).any()
        if i==0: left_ax,right_ax=ax_l,ax_r
    for ax,direction,caption in [(left_ax,'up','Up DEG count + 1'),
                                 (right_ax,'down','Down DEG count + 1')]:
        bbox=ax.get_position()
        cax=fig.add_axes([bbox.x0+.07*bbox.width,
                          .885 if nrows==2 else .812,
                          bbox.width*.86,.017])
        cmap=plt.get_cmap(COLORS[direction])
        cb=fig.colorbar(cm.ScalarMappable(norm=mcolors.Normalize(0,maxlog[direction]),cmap=cmap),
                        cax=cax,orientation='horizontal')
        cb.ax.invert_xaxis()
        cb.set_ticks(range(maxlog[direction]+1))
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.tick_params(labelsize=8.6, length=0, pad=2)
        cb.outline.set_visible(False)
        # Draw the subscript 2 as an independent Helvetica text object,
        # avoiding mathtext's embedded DejaVu Sans in the Illustrator PDF.
        x0 = bbox.x0 + .07*bbox.width
        y0 = .946 if nrows==2 else .910
        fig.text(x0, y0, 'Log', fontsize=9.3, ha='left')
        fig.text(x0 + .0195, y0 - .0055, '2', fontsize=6.8, ha='left')
        fig.text(x0 + .0285, y0, f'({caption})', fontsize=9.3, ha='left')
    stratum_caption = 'All donors' if stratum == 'all' else 'PD cases only'
    footer = f'VCF ALT carrier vs noncarrier  |  edgeR-QLF, BH-FDR < 0.05  |  {stratum_caption}  |  Cell values: DEG counts'
    if missing:
        footer += '  |  Gray (-): not verified as fitted'
    elif (not audit.loc[audit.snp.isin(snps), 'fit_verified'].all()):
        footer += '  |  WARNING: unverified zeros assumed'
    fig.text(.068,.053 if nrows==2 else .075,footer,fontsize=8.1,color='#576478')
    path.parent.mkdir(parents=True,exist_ok=True)
    # Adobe Illustrator: Helvetica is a macOS standard font; use PDF base-14
    # fonts rather than subsetting DejaVu Sans (which AI may flag as missing).
    # For SVG explicitly request macOS Arial (no text outlines) in editable
    # <text> objects. PNG remains Agg for Cairo compatibility.
    fig.savefig(str(path.with_suffix('.pdf')), format='pdf', backend='pdf',
                bbox_inches='tight', pad_inches=.11, facecolor='white')
    svg_path = path.with_suffix('.svg')
    fig.savefig(str(svg_path), format='svg', backend='svg',
                bbox_inches='tight', pad_inches=.11, facecolor='white')
    svg_markup = svg_path.read_text(encoding='utf-8')
    # Matplotlib may resolve a Linux font family at export time. Rebind the
    # SVG text objects to Arial, which is present on macOS, without outlining.
    svg_markup = re.sub(r'font-family:\s*[^;\"]+', 'font-family: Arial', svg_markup)
    svg_path.write_text(svg_markup, encoding='utf-8')
    # PNG is a preview image; force Agg for stable PNG encoding. Some Cairo
    # versions pass Path objects to cairocffi.write_to_png(), which expects str.
    # Explicit Agg also avoids the Cairo PNG writer entirely on older servers.
    fig.savefig(str(path.with_suffix('.png')), format='png', backend='agg', dpi=dpi,
                bbox_inches='tight', pad_inches=.11, facecolor='white')
    plt.close(fig)
    LOGGER.info('Saved figure: %s.[pdf/svg/png]',path)


def main(argv=None):
    parser=argparse.ArgumentParser(description='Plot publication Fig.3F from final edgeR QLF significant CSV, no DE reruns.')
    parser.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    parser.add_argument('--input',default=None,help='FDR05_significant_edgeR_QLF_genes.csv')
    parser.add_argument('--stratum', choices=['all','case_only'], default='all', help='Select edgeR analysis stratum; case_only = PD cases only')
    parser.add_argument('--outdir',default=None,help='Output directory (separate from edgeR input)')
    parser.add_argument('--assume-complete',action='store_true',help='For missing fit status only: assume absent-significant rows mean 0 DEGs. NOT recommended for final publication.')
    parser.add_argument('--dpi',type=int,default=350)
    args=parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    logging.getLogger('fontTools').setLevel(logging.WARNING)
    root=Path(args.root)
    inputfile=Path(args.input) if args.input else root/'single_iv_causal/postmortem_snp_scDE/carrier_vs_noncarrier_v6_vcf_verified/edger/FDR05_significant_edgeR_QLF_genes.csv'
    default_dir = 'figure3_postmortem_edger_qlf' if args.stratum == 'all' else 'figure3_postmortem_edger_qlf_case_only'
    out=Path(args.outdir) if args.outdir else root/'manuscript_figures/tqtl_benchmark_v4'/default_dir
    data=load_significant(inputfile, stratum=args.stratum)
    coverage=detected_comparisons(inputfile, stratum=args.stratum)
    audit=create_counts(data,coverage,assume_complete=args.assume_complete,stratum=args.stratum)
    out.mkdir(parents=True,exist_ok=True)
    audit.to_csv(out/f'Figure3F_edgeR_QLF_{args.stratum}_DEG_count_audit.csv',index=False)
    input_sha=sha256_file(inputfile)
    (out/f'Figure3F_{args.stratum}_source_sha256.txt').write_text(f'{input_sha}  {inputfile}\n')
    unverified=audit[~audit.fit_verified]
    if not unverified.empty:
        LOGGER.warning('%d / %d SNP-region-cell comparisons are unverified (includes assumed complete). Check audit CSV.',len(unverified),len(audit))
        if args.assume_complete:
            LOGGER.warning('--assume-complete active; zeros are assumed for comparisons absent in both fit_status and significant table.')
    else:
        LOGGER.info('All %d SNP-region-cell comparisons verified as fitted',len(audit))
    for snp in SNPS:
        sub=audit[audit.snp.eq(snp)]
        LOGGER.info('%s  Up=%s  Down=%s  Tested=%d/35',snp,sub.n_up.sum(),sub.n_down.sum(),int(sub.fit_verified.sum()))
    plot_one(audit,list(SNPS),out/f'Figure3F_postmortem_edgeR_QLF_{args.stratum}',dpi=args.dpi,stratum=args.stratum)
    for snp in SNPS:
        plot_one(audit,[snp],out/f'Figure3F_{snp}_edgeR_QLF_{args.stratum}',dpi=args.dpi,stratum=args.stratum)
    LOGGER.info('Finished: %s',out)

if __name__=='__main__':
    main()
