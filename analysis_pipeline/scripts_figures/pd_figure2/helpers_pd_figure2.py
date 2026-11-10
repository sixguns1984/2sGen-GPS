import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.lines import Line2D

PHENO_ORDER = ['updrs1','updrs2','updrs3']
PHENO_LABEL = {
    'updrs1':'MDS-UPDRS I',
    'updrs2':'MDS-UPDRS II',
    'updrs3':'MDS-UPDRS III',
    'moca':'MoCA',
}
OUTCOME_COL = {
    'updrs1':'mds_updrs_part_i_summary_score',
    'updrs2':'mds_updrs_part_ii_summary_score',
    'updrs3':'mds_updrs_part_iii_summary_score',
    'moca':'moca_total_score',
}
PHENO_COLOR = {
    'updrs1':'#7655AA',
    'updrs2':'#168D79',
    'updrs3':'#D5583A',
    'moca':'#2D69AF',
}
GENO_COLORS = ['#D8A031','#2A6DB3','#D95F02']


def setup_rcparams():
    mpl.rcParams.update({
        'font.family':'sans-serif',
        'font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
        'font.size':9,
        'axes.titlesize':12,
        'axes.labelsize':10,
        'axes.linewidth':0.8,
        'xtick.labelsize':8,
        'ytick.labelsize':8,
        'xtick.major.width':0.8,
        'ytick.major.width':0.8,
        'xtick.major.size':3,
        'ytick.major.size':3,
        'legend.fontsize':8,
        'legend.title_fontsize':8,
        'savefig.dpi':300,
        'pdf.fonttype':42,
        'ps.fonttype':42,
        'svg.fonttype':'none',
    })


def read_publication_table(path):
    df = pd.read_csv(path, low_memory=False)
    if 'phenotype' not in df.columns:
        raise KeyError('publication table missing phenotype column')
    df['phenotype'] = df['phenotype'].astype(str).str.lower()
    return df


def is_a_tier(s):
    return isinstance(s, str) and s.startswith('A_')


def tier_rank(s):
    order = {
        'A_both_AR_and_multiIV_supported':0,
        'A_multiIV_stable':1,
        'A_crosscohort_AR_supported':2,
        'B_multiIV_FDR':3,
        'B_AR_discoveryFDR_repNominal':4,
        'C_primary_replicated':5,
    }
    return order.get(s, 9)


def safe_neglog10(x, floor=1e-300):
    x = np.maximum(np.asarray(x, dtype=float), floor)
    return -np.log10(x)


def attach_artifact_ids(df, artifact_table):
    if 'artifact_id' in df.columns and df['artifact_id'].notna().any():
        return df
    if not artifact_table:
        return df
    add = pd.read_csv(artifact_table, low_memory=False)
    need = [c for c in ['gene','lead_snp','artifact_id'] if c in add.columns]
    if need != ['gene','lead_snp','artifact_id']:
        return df
    add = add[['gene','lead_snp','artifact_id']].drop_duplicates()
    out = df.merge(add, on=['gene','lead_snp'], how='left', suffixes=('','_art'))
    if 'artifact_id_art' in out.columns:
        if 'artifact_id' in out.columns:
            out['artifact_id'] = out['artifact_id'].where(out['artifact_id'].notna(), out['artifact_id_art'])
            out = out.drop(columns=['artifact_id_art'])
        else:
            out = out.rename(columns={'artifact_id_art':'artifact_id'})
    return out


def detect_story_examples(df, requested=None, n_examples=2):
    a = df[df['descriptive_evidence_tier'].map(is_a_tier)].copy()
    if a.empty:
        a = df.copy()  # No A-level examples: use real power-qualified candidates transparently.
    # New biologically interpretable defaults after power filtering and MoCA removal.
    pref = [
        ('COQ8A','updrs3'),  # mitochondrial CoQ metabolism
        ('ATG10','updrs2'),  # autophagy, activity of daily living
        ('FADS2','updrs1'),  # lipid metabolism, non-motor symptoms
        ('FLOT1','updrs3'),  # DAT trafficking / striatal biology
        ('SQSTM1','updrs3'), # selective autophagy
        ('TIRAP','updrs3'),  # innate immune signaling
        ('BACE2','updrs3'),  # APP proteolysis
    ]
    if requested:
        out=[]
        for item in requested:
            parts=item.split(':')
            gene=parts[0]
            phenotype=parts[1].lower() if len(parts)>1 else None
            sub=a[a['gene'].eq(gene)]
            if phenotype is not None:
                sub=sub[sub['phenotype'].eq(phenotype)]
            if sub.empty:
                raise ValueError(f'Requested story example not found among A-tier rows: {item}')
            out.append(sub.sort_values(['PD_PLUS_SU_pvalue','PPMI_pvalue']).iloc[0].to_dict())
        return pd.DataFrame(out)
    picked=[]; used=set()
    for gene, phenotype in pref:
        sub=a[(a['gene'].eq(gene)) & (a['phenotype'].eq(phenotype))]
        if not sub.empty:
            row=sub.sort_values(['PD_PLUS_SU_pvalue','PPMI_pvalue']).iloc[0]
            key=(row['gene'], row['phenotype'])
            if key not in used:
                picked.append(row.to_dict()); used.add(key)
            if len(picked)>=n_examples:
                return pd.DataFrame(picked)
    for phen in PHENO_ORDER:
        sub=a[a['phenotype'].eq(phen)].sort_values(['PD_PLUS_SU_pvalue','PPMI_pvalue'])
        if not sub.empty:
            row=sub.iloc[0]
            key=(row['gene'], row['phenotype'])
            if key not in used:
                picked.append(row.to_dict()); used.add(key)
            if len(picked)>=n_examples:
                return pd.DataFrame(picked)
    if len(picked)<n_examples:
        for _,row in a.sort_values(['PD_PLUS_SU_pvalue','PPMI_pvalue']).iterrows():
            key=(row['gene'], row['phenotype'])
            if key not in used:
                picked.append(row.to_dict()); used.add(key)
            if len(picked)>=n_examples:
                break
    return pd.DataFrame(picked)


def panel_a_order(df, top_n_labels=6):
    plot_df=df.copy()
    plot_df['selected_replication_pvalue'] = pd.to_numeric(plot_df.get('PD_PLUS_SU_pvalue'), errors='coerce')
    plot_df['p_plot']=plot_df['selected_replication_pvalue'].fillna(pd.to_numeric(plot_df.get('PPMI_pvalue'), errors='coerce'))
    plot_df['neglog10p']=safe_neglog10(plot_df['p_plot'])
    eff1 = pd.to_numeric(plot_df.get('PPMI_coef_sum'), errors='coerce')
    eff2 = pd.to_numeric(plot_df.get('PD_PLUS_SU_coef_sum'), errors='coerce')
    plot_df['effect']=eff1.where(eff1.notna(), eff2)
    plot_df['sign']=np.where(plot_df['effect'].fillna(0)>=0,'positive','negative')
    plot_df['evidence_rank']=plot_df['descriptive_evidence_tier'].map(tier_rank)
    plot_df['shared_gene']=plot_df.groupby('gene')['phenotype'].transform('nunique')>1
    plot_df['multiIV_FDR_support'] = pd.to_numeric(plot_df.get('multiIV_p_q_BH'), errors='coerce').lt(0.05)
    chunks=[]; offset=0
    for phen in PHENO_ORDER:
        sub=plot_df[plot_df['phenotype'].eq(phen)].copy()
        if sub.empty: continue
        sub=sub.sort_values(['evidence_rank','selected_replication_pvalue','PPMI_pvalue','gene']).reset_index(drop=True)
        sub['x']=np.arange(offset+1, offset+1+len(sub))
        offset += len(sub)+3
        # Labels are assigned by the shared PD/AD Panel-a selector in the entrypoint.
        sub['label']=''; sub['label_reason']=''
        chunks.append(sub)
    return pd.concat(chunks, ignore_index=True) if chunks else plot_df


def draw_panel_a(ax, plot_df, show_legend=True):
    for phen in PHENO_ORDER:
        sub=plot_df[plot_df['phenotype'].eq(phen)]
        if sub.empty: continue
        color=PHENO_COLOR[phen]
        ymax=ax.get_ylim()[1]
        cap=ymax*0.89
        y=np.minimum(sub['neglog10p'].to_numpy(float),cap)
        x=sub['x'].to_numpy(float)
        ax.vlines(x, 0, y, color=color, alpha=0.34, lw=0.7, zorder=1)
        for sign in ('positive','negative'):
            d=sub[sub['sign'].eq(sign)]
            if d.empty: continue
            face=(color if sign=='positive' else 'white')
            mi_flag=d['multiIV_FDR_support'].fillna(False).astype(bool)
            edge=['#C83CB4' if bool(f) else '#333333' for f in mi_flag]
            dy=np.minimum(d['neglog10p'].to_numpy(float),cap)
            ax.scatter(d['x'], dy, s=np.where(d['shared_gene'], 46, 29),
                       facecolors=face, edgecolors=edge,
                       linewidths=np.where(d['shared_gene'],1.2,0.8), marker='o', zorder=3)
            extreme=d[d['neglog10p']>cap]
            if not extreme.empty:
                ax.scatter(extreme['x'],np.full(len(extreme),cap+0.10),marker='^',s=36,
                           color=color,edgecolor='black',linewidth=.5,zorder=5)
        xm=(sub['x'].min()+sub['x'].max())/2
        ax.text(xm, ax.get_ylim()[1]*0.985, PHENO_LABEL[phen], color=color, ha='center', va='top', fontsize=10, fontweight='bold')
    y_top=ax.get_ylim()[1]; y_cap=y_top*.89
    for phen in PHENO_ORDER:
        sub=plot_df[(plot_df['phenotype']==phen)&(plot_df['label'].ne(''))].sort_values('x')
        for j,(_,r) in enumerate(sub.iterrows()):
            yy=min(float(r['neglog10p']),y_cap)
            offset=.18+((j%3)*.14)
            ax.text(r['x'], min(yy+offset,y_top*.97), r['label'], rotation=63,
                    ha='left',va='bottom',fontsize=6.7,color='#2E3440')
    ax.set_ylabel(r'$-\log_{10}$(replication $P$)')
    ax.set_xticks([])
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', color='#D9D9D9', lw=0.6, alpha=0.8)
    if not show_legend:
        return
    handles = [
        *[Line2D([0],[0], marker='o', color='none', label=PHENO_LABEL[p],
                 markerfacecolor=PHENO_COLOR[p], markeredgecolor=PHENO_COLOR[p], markersize=5.2)
          for p in PHENO_ORDER if p in plot_df['phenotype'].unique()],
        Line2D([0],[0], marker='o', color='none', label='Positive summed-lag coefficient',
               markerfacecolor='#666666', markeredgecolor='#666666', markersize=5.0),
        Line2D([0],[0], marker='o', color='none', label='Negative summed-lag coefficient',
               markerfacecolor='white', markeredgecolor='#666666', markersize=5.0),
        Line2D([0],[0], marker='o', color='none', label='Cross-phenotype gene',
               markerfacecolor='white', markeredgecolor='#333333', markeredgewidth=1.0, markersize=7.0),
        Line2D([0],[0], marker='o', color='none', label='Multi-IV BH-FDR < 0.05',
               markerfacecolor='white', markeredgecolor='#C83CB4', markeredgewidth=1.2, markersize=5.5),
        Line2D([0],[0], marker='^', color='none', label='P above display range',
               markerfacecolor='#666666', markeredgecolor='#333333', markersize=5.2),
    ]
    ax.legend(handles=handles, ncol=4, loc='lower left', bbox_to_anchor=(0.0,1.17),
              frameon=False, fontsize=6.4, columnspacing=1.0, handletextpad=0.35,
              labelspacing=0.35, borderaxespad=0.0)