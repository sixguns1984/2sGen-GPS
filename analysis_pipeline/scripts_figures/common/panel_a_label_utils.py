"""Shared, provenance-preserving Panel-a gene label selection for PD Figure 2 / AD Figure 4.

This module DOES NOT select A-tier evidence candidates, change order, re-rank the
scientific source table or alter panels b/c.  Pass rows already selected and
ordered for panel a; it returns flags identifying which existing points to label.

Examples:
    selected = select_panel_a_labels(df, gene_col='gene', x_col='plot_x',
        y_col='neglog10_replication_p', phenotype_col='phenotype',
        source_tier_col='descriptive_evidence_tier', y_cap=3.5)
    draw_panel_a_labels(ax, selected, x_col='plot_x', y_plot_col='plot_y', ...)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

# The final PD master supplied in the project uses this exact field and values
# such as A_crosscohort_AR_supported, B_AR_discoveryFDR_repNominal.
TIER_COLUMNS = (
    'descriptive_evidence_tier',
    'causal_evidence_grade', 'causal_evidence_tier',
    'evidence_grade', 'evidence_tier', 'tier',
)
MULTI_Q_COLUMNS = (
    'multiIV_p_q_BH', 'multi_iv_bh_fdr', 'multiIV_BH_FDR',
    'multi_iv_fdr', 'multiIV_q_BH',
)
MULTI_FLAG_COLUMNS = ('multiIV_FDR_support', 'multi_iv_FDR_support', 'multi_iv_fdr_support')
CROSS_COLUMNS = ('cross_phenotype_gene', 'is_cross_phenotype_gene', 'cross_phenotype')
_TRUE = {'true', 't', '1', 'yes', 'y'}
_FALSE = {'false', 'f', '0', 'no', 'n', '', 'nan', 'none', 'na', '<na>'}


def _column(df, explicit, candidates, required=False, purpose='column'):
    if explicit is not None:
        if explicit not in df.columns:
            raise KeyError(f'{purpose} column {explicit!r} absent; available: {list(df.columns)}')
        return explicit
    col = next((c for c in candidates if c in df.columns), None)
    if col is None and required:
        raise KeyError(f'No {purpose} column. Checked {candidates}; available: {list(df.columns)}')
    return col


def _bool_series(values, name):
    """Convert only unequivocal boolean values, not arbitrary nonempty strings."""
    out = []
    for v in values:
        if pd.isna(v):
            out.append(False)
        elif isinstance(v, (bool, np.bool_)):
            out.append(bool(v))
        elif isinstance(v, (int, np.integer, float, np.floating)) and float(v) in (0., 1.):
            out.append(bool(v))
        else:
            text = str(v).strip().lower()
            if text in _TRUE:
                out.append(True)
            elif text in _FALSE:
                out.append(False)
            else:
                raise ValueError(f'{name}: unsupported boolean value {v!r}; specify the correct flag column')
    return pd.Series(out, index=values.index, dtype=bool)


def select_panel_a_labels(
    panel_df: pd.DataFrame,
    *,
    gene_col: str = 'gene',
    x_col: str = 'plot_x',
    y_col: str = 'y_raw',
    phenotype_col: Optional[str] = None,
    y_cap: Optional[float] = None,
    top_n_per_group: int = 8,
    source_tier_col: Optional[str] = None,
    multi_iv_q_col: Optional[str] = None,
    multi_iv_flag_col: Optional[str] = None,
    cross_phenotype_col: Optional[str] = None,
    top_group_order: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Mark only gene labels in already plotted Panel-a rows.

    Force: capped point OR multi-IV BH-FDR < 0.05 OR cross-phenotype.
    Add: top 8 otherwise unlabelled genes per phenotype (including MMSE).
    De-duplicate: the same gene is annotated once per full panel at its largest
    true (unclipped) -log10 P, with deterministic tie-breaking.

    IMPORTANT: y_col must be the SAME P-derived score used to DRAW panel a.
    Do NOT calculate it from another cohort/replication P value.
    """
    if not isinstance(panel_df, pd.DataFrame):
        raise TypeError('panel_df must be a pandas DataFrame')
    if not isinstance(top_n_per_group, int) or top_n_per_group < 0:
        raise ValueError('top_n_per_group must be a nonnegative integer')
    if not panel_df.index.is_unique:
        raise ValueError('panel_df must have a unique index, to retain original row positions')
    for c in (gene_col, x_col, y_col):
        if c not in panel_df.columns:
            raise KeyError(f'Missing plotted panel-a column {c!r}')
    if phenotype_col is not None and phenotype_col not in panel_df.columns:
        raise KeyError(f'Missing phenotype column {phenotype_col!r}')
    if y_cap is not None and (not np.isfinite(y_cap) or y_cap <= 0):
        raise ValueError('y_cap must be finite and positive')

    df = panel_df.copy()
    df['_panel_a_gene'] = df[gene_col].astype('string').str.strip()
    if df['_panel_a_gene'].isna().any() or df['_panel_a_gene'].eq('').any():
        raise ValueError('Panel-a contains blank or missing gene names')
    df['_panel_a_group'] = df[phenotype_col].astype(str) if phenotype_col else 'MMSE'
    df['_panel_a_y_raw'] = pd.to_numeric(df[y_col], errors='coerce')
    df['_panel_a_x'] = pd.to_numeric(df[x_col], errors='coerce')
    if not np.isfinite(df['_panel_a_y_raw'].to_numpy()).all() or not np.isfinite(df['_panel_a_x'].to_numpy()).all():
        raise ValueError('Panel-a x / true y contains missing or non-finite values')

    tier = _column(df, source_tier_col, TIER_COLUMNS, required=True, purpose='original evidence grade')
    if df[tier].isna().any():
        raise ValueError('Missing source evidence tier in the actual plotted rows; perform a strict master join')
    df['panel_a_source_evidence_tier'] = df[tier].copy()  # NEVER recode values

    q_col = _column(df, multi_iv_q_col, MULTI_Q_COLUMNS, required=False, purpose='multi-IV BH FDR')
    flag_col = _column(df, multi_iv_flag_col, MULTI_FLAG_COLUMNS, required=False, purpose='multi-IV flag')
    if q_col is not None:
        q = pd.to_numeric(df[q_col], errors='coerce')
        multi = (q < 0.05).fillna(False)
        if flag_col is not None:
            ref = _bool_series(df[flag_col], flag_col)
            comparable = q.notna()
            if (multi[comparable] != ref[comparable]).any():
                raise ValueError(f'Multi-IV BH-FDR and {flag_col} disagree. Investigate original master fields.')
    elif flag_col is not None:
        multi = _bool_series(df[flag_col], flag_col)
    else:
        multi = pd.Series(False, index=df.index)
    df['panel_a_multi_iv_bh_fdr05'] = multi.astype(bool)

    cross_col = _column(df, cross_phenotype_col, CROSS_COLUMNS, required=False, purpose='cross-phenotype flag')
    df['panel_a_cross_phenotype'] = (
        _bool_series(df[cross_col], cross_col) if cross_col else False
    )
    df['panel_a_above_cap'] = bool(False) if y_cap is None else df['_panel_a_y_raw'] > y_cap
    df['panel_a_label'] = False
    df['panel_a_label_reason'] = ''

    # One best position per gene over the entire panel, not one per IV row.
    # Preserve current plotting order on tie; row identity is preserved.
    ordered = df.sort_values('_panel_a_y_raw', ascending=False, kind='mergesort')
    representative = ordered[~ordered['_panel_a_gene'].duplicated(keep='first')]
    gene_flags = df.groupby('_panel_a_gene', sort=False)[
        ['panel_a_above_cap', 'panel_a_multi_iv_bh_fdr05', 'panel_a_cross_phenotype']
    ].any()
    for i, row in representative.iterrows():
        flags = gene_flags.loc[row['_panel_a_gene']]
        reasons = []
        if bool(flags['panel_a_above_cap']): reasons.append('above_display_range')
        if bool(flags['panel_a_multi_iv_bh_fdr05']): reasons.append('multi_IV_BH_FDR_lt_0.05')
        if bool(flags['panel_a_cross_phenotype']): reasons.append('cross_phenotype_gene')
        if reasons:
            df.at[i, 'panel_a_label'] = True
            df.at[i, 'panel_a_label_reason'] = '+'.join(reasons)

    groups = list(top_group_order) if top_group_order is not None else list(dict.fromkeys(df['_panel_a_group'].tolist()))
    unexpected = set(df['_panel_a_group'].unique()) - set(groups)
    if unexpected:
        raise ValueError(f'Phenotype groups not listed in top_group_order: {unexpected}')
    for group in groups:
        eligible = representative.loc[
            representative['_panel_a_group'].eq(group)
            & ~representative.index.isin(df.index[df['panel_a_label']])
        ]
        extra = eligible.sort_values('_panel_a_y_raw', ascending=False, kind='mergesort').head(top_n_per_group)
        for i in extra.index:
            df.at[i, 'panel_a_label'] = True
            df.at[i, 'panel_a_label_reason'] = 'top_n_per_phenotype'
    return df


def merge_original_evidence(
    plot_df: pd.DataFrame,
    source_master: pd.DataFrame,
    *,
    plot_keys=('gene', 'lead_snp'),
    source_keys=('gene', 'lead_snp'),
    source_tier_col='descriptive_evidence_tier',
    output_col='descriptive_evidence_tier',
):
    """Strictly join original-grade STRING without reconstructing A/B/C.

    Pair keys are essential: do not merge by gene alone when multiple IVs exist.
    Include phenotype in the keys when joining pooled PD I/II/III data.
    """
    keys1, keys2 = list(plot_keys), list(source_keys)
    if len(keys1) != len(keys2):
        raise ValueError('plot_keys and source_keys lengths differ')
    for c in keys1:
        if c not in plot_df: raise KeyError(f'Plot missing key: {c}')
    for c in keys2 + [source_tier_col]:
        if c not in source_master: raise KeyError(f'Original master missing column: {c}')
    source = source_master[keys2 + [source_tier_col]].copy()
    if source.duplicated(keys2).any():
        raise ValueError('Original master keys are duplicated; add phenotype/lag keys rather than guessing')
    names = {a:b for a,b in zip(keys2,keys1)}
    source = source.rename(columns=names).rename(columns={source_tier_col:'__verified_original_tier'})
    if output_col in plot_df.columns:
        raise ValueError('Grade column already present in plot data; do not silently overwrite')
    merged = plot_df.merge(source, how='left', on=keys1, validate='many_to_one', sort=False)
    if merged['__verified_original_tier'].isna().any():
        examples = merged.loc[merged['__verified_original_tier'].isna(),keys1].head(5).to_dict('records')
        raise ValueError(f'{merged["__verified_original_tier"].isna().sum()} rows missing matching original evidence grade; examples={examples}')
    return merged.rename(columns={'__verified_original_tier':output_col})


def write_label_audit(df, path, *, gene_col='gene', phenotype_col=None, extra_cols=()):
    """Save exactly WHICH points are labelled and provenance of their grade."""
    cols = [gene_col]
    if phenotype_col and phenotype_col in df: cols.append(phenotype_col)
    for c in ('_panel_a_x','_panel_a_y_raw','panel_a_source_evidence_tier',
              'panel_a_multi_iv_bh_fdr05','panel_a_cross_phenotype',
              'panel_a_above_cap','panel_a_label','panel_a_label_reason'):
        if c in df: cols.append(c)
    cols.extend(c for c in extra_cols if c in df and c not in cols)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(path, index=False)
    return path


def draw_panel_a_labels(ax, df, *, x_col='plot_x', y_plot_col='plot_y', gene_col='gene',
                        y_offset=0.05, rotation=60, fontsize=8, color='black',
                        ha='left', va='bottom', zorder=20, **kwargs):
    """Render SELECTED labels only; replace existing ax.text/annotate label loop.

    Labels follow the same plotted x and CLIPPED y as the original scatter.
    No axes limits, positioning of other panels, point styling or legends changed.
    """
    if y_plot_col not in df.columns:
        raise KeyError(f'Expected actual plotted y column {y_plot_col!r}; supply the exact plotted coordinate')
    text_artists = []
    for _, row in df.loc[df['panel_a_label']].iterrows():
        val = float(row[y_plot_col])
        if not np.isfinite(val): raise ValueError('Nonfinite plotted y')
        text_artists.append(ax.text(float(row[x_col]), val + y_offset,
                      str(row[gene_col]), rotation=rotation, fontsize=fontsize,
                      color=color, ha=ha, va=va, zorder=zorder, **kwargs))
    return text_artists


def save_editable_svg(fig, path):
    """Export genuine SVG <text> elements while original backend remains Cairo.

    Cairo's own SVG backend converts text to outlines even if svg.fonttype='none'.
    This calls Matplotlib's SVG renderer *only for the SVG export* and restores
    the original Cairo canvas afterwards. The main required backend stays Cairo.
    """
    import matplotlib as mpl
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    canvas = fig.canvas
    try:
        with mpl.rc_context({'svg.fonttype':'none'}):
            FigureCanvasSVG(fig).print_svg(str(path))
    finally:
        fig.set_canvas(canvas)
    return path
