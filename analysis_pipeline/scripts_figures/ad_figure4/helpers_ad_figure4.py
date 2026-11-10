import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('cairo')  # Use Cairo backend for high-quality output
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PHENO_LABEL={'ad_mmse':'MMSE'}
OUTCOME_ALIASES=['mmse','MMSE','mmse_total','MMSE_total','mmse_score','MMSE_score','mmse_total_score','MMSE_total_score']
GENO_COLORS=['#D8A031','#2A6DB3','#D95F02']
TIER_OUTLINE={
    'A_crosscohort_AR_supported':'#2F2F2F',
    'A_both_AR_and_multiIV_supported':'#B641C4',
    'A_multiIV_stable':'#B641C4',
}

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
    df=pd.read_csv(path,low_memory=False)
    df['phenotype']='ad_mmse'
    return df


def is_a_tier(s):
    return isinstance(s,str) and s.startswith('A_')


def safe_neglog10(x,floor=1e-300):
    x=np.maximum(np.asarray(x,dtype=float),floor)
    return -np.log10(x)


def rank_for_panel_a(df, top_n=30):
    d=df[df['descriptive_evidence_tier'].map(is_a_tier)].copy()
    if d.empty: raise ValueError('No A-tier rows found in AD MMSE table')
    d['rep_p']=pd.to_numeric(d['ADNI_pvalue'],errors='coerce')
    d['disc_p']=pd.to_numeric(d['MAP_pvalue'],errors='coerce')
    d['neglog10p']=safe_neglog10(d['rep_p'])
    d['direction']=np.where(pd.to_numeric(d['MAP_coef_sum'],errors='coerce')>=0,'positive','negative')
    d['display_label']=d['gene'].astype(str)
    d=d.sort_values(['rep_p','disc_p','gene']).reset_index(drop=True)
    d=d.head(int(top_n)).copy()
    d['x']=np.arange(len(d))
    return d


def detect_story_examples(df, requested=None, n_examples=2):
    a=df[df['descriptive_evidence_tier'].map(is_a_tier)].copy()
    if a.empty: raise ValueError('No A-tier rows found in publication table.')
    # discussion-friendly preference list
    pref=['TRPC6','TIRAP','SEMA3C','PLD2','GDPD5','FAM19A2','NDUFA3','CARS2','DYNLL1','CLCN7','RAB11FIP4']
    if requested:
        out=[]
        for gene in requested:
            sub=a[a['gene'].eq(gene)]
            if sub.empty: raise ValueError(f'Requested story example not found among A-tier rows: {gene}')
            out.append(sub.sort_values(['ADNI_pvalue','MAP_pvalue']).iloc[0].to_dict())
        return pd.DataFrame(out)
    picked=[]; used=set()
    for gene in pref:
        sub=a[a['gene'].eq(gene)]
        if not sub.empty:
            row=sub.sort_values(['ADNI_pvalue','MAP_pvalue']).iloc[0]
            if row['gene'] not in used:
                picked.append(row.to_dict()); used.add(row['gene'])
            if len(picked)>=n_examples: return pd.DataFrame(picked)
    for _,row in a.sort_values(['ADNI_pvalue','MAP_pvalue']).iterrows():
        if row['gene'] not in used:
            picked.append(row.to_dict()); used.add(row['gene'])
        if len(picked)>=n_examples: break
    return pd.DataFrame(picked)


def draw_panel_a(ax,pdata):
    fill_map={'positive':'#CB6B50','negative':'#39799C'}
    for _,r in pdata.iterrows():
        outline=TIER_OUTLINE.get(r['descriptive_evidence_tier'],'#2F2F2F')
        ax.scatter(r['x'],r['neglog10p'],s=36,c=fill_map[r['direction']],edgecolors=outline,linewidths=0.85,zorder=3)
    ax.set_xticks(pdata['x'])
    ax.set_xticklabels(pdata['display_label'],rotation=70,ha='right',fontsize=7)
    ax.set_ylabel(r'$-\log_{10}(P_{ADNI})$')
    ax.set_title('a  A-tier MMSE-associated IV–genes (replication-ranked)',loc='left',weight='bold',fontsize=11)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#E6EAEE',lw=.65)
    leg=[
        Line2D([0],[0],marker='o',color='none',label='MAP coef_sum > 0',markerfacecolor='#CB6B50',markeredgecolor='#2F2F2F',markersize=6),
        Line2D([0],[0],marker='o',color='none',label='MAP coef_sum < 0',markerfacecolor='#39799C',markeredgecolor='#2F2F2F',markersize=6),
        Line2D([0],[0],marker='o',color='none',label='A: crosscohort AR',markerfacecolor='white',markeredgecolor='#2F2F2F',markersize=6),
        Line2D([0],[0],marker='o',color='none',label='A: AR + multi-IV',markerfacecolor='white',markeredgecolor='#B641C4',markersize=6),
    ]
    ax.legend(handles=leg,frameon=False,ncol=2,loc='upper left',fontsize=7,handlelength=1.0,columnspacing=0.8)
    # annotate top points sparsely
    top=pdata.sort_values(['rep_p','disc_p']).head(min(12,len(pdata)))
    for _,r in top.iterrows():
        ax.text(r['x'],r['neglog10p']+0.08,r['gene'],fontsize=6.5,rotation=55,ha='left',va='bottom')

