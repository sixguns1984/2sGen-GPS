"""Shared, dependency-light plotting and audit helpers for tQTL benchmark figures."""
from __future__ import annotations
import json
import re
import warnings
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CURRENT='submitted_followup_ctar_mptga'
OLD='submitted_followup_mptga'
BAD='lin2018_regression'  # unadjusted linear regression; removed everywhere
CURRENT_BASELINE='lin2018_regression_cov'
MAIN_METHODS=[CURRENT,CURRENT_BASELINE,'lin2018_mptga','lin2018_ar','union','fisher','manova']
REVIEW_METHODS=[CURRENT,OLD,CURRENT_BASELINE,'lin2018_mptga','lin2018_ar','union','fisher','manova']
LABELS={CURRENT:'CTAR-MPTGA',OLD:'MPTGA (original)',CURRENT_BASELINE:'Linear regression + covariates',
        'lin2018_mptga':'Lin2018-MPTGA','lin2018_ar':'AR(1) regression',
        'union':'Union','fisher':'Fisher','manova':'MANOVA'}
COLORS={CURRENT:'#B83B45', OLD:'#C47772', CURRENT_BASELINE:'#245A81',
        'lin2018_mptga':'#34989D','lin2018_ar':'#73568B',
        'union':'#AD813E','fisher':'#626C78','manova':'#3F8565'}
MARKERS={CURRENT:'o',OLD:'X',CURRENT_BASELINE:'s','lin2018_mptga':'D',
         'lin2018_ar':'^','union':'v','fisher':'P','manova':'h'}

REGRESSION_METHODS=[CURRENT,OLD,CURRENT_BASELINE,'lin2018_mptga','lin2018_ar']
COMBINATION_METHODS=['union','fisher','manova']
GROUP_TEXT_COLORS={'regression':'#173A5E','combination':'#6A5A38'}


def set_style():
    plt.rcParams.update({
        'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
        'font.size':7.7,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7.3,
        'ytick.labelsize':7.3,'legend.fontsize':7.4,'axes.linewidth':0.75,
        'xtick.major.width':0.7,'ytick.major.width':0.7,'xtick.major.size':3,
        'ytick.major.size':3,'lines.linewidth':1.4,'pdf.fonttype':42,
        'ps.fonttype':42,'svg.fonttype':'none','figure.dpi':160,
        'savefig.dpi':600,'axes.unicode_minus':False})


def tidy(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out', length=3, pad=2)
    ax.grid(axis='y', color='#ECEFF2', lw=0.5, zorder=0)


def tag(ax,letter):
    ax.text(-0.13,1.075,letter,transform=ax.transAxes,weight='bold',fontsize=10.5,
            va='top',ha='right',clip_on=False)


def save(fig,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    for fmt in ('pdf','svg','png'):
        fig.savefig(p.with_suffix('.'+fmt),dpi=600 if fmt=='png' else None,
                    bbox_inches='tight',pad_inches=0.075)
    plt.close(fig)


def write_json(p,data):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str),encoding='utf-8')


def clean_methods(df,only_main=True):
    z=df.copy()
    if 'method' not in z:raise ValueError('Input missing method column')
    m=z['method'].astype(str).str.strip()
    z=z.loc[(~m.str.endswith('__nocov')) & (m!=BAD)].copy()
    permitted=MAIN_METHODS if only_main else REVIEW_METHODS
    z=z[z.method.isin(permitted)].copy()
    if CURRENT not in set(z.method):
        raise ValueError(f'Missing primary {CURRENT}; available methods={sorted(set(df.method.astype(str)))}')
    return z


def parse_simulation(raw):
    """Convert genuine simulation_summary.csv (aggregated) to plotted format.

    Actual aggregate_replicates writes rho_true_mean (NOT rho_true), and does not
    preserve the nonnumeric confounding_status, so infer status from 'scenario'.
    """
    z=raw.copy()
    req={'method','scenario','n_replicates'}
    if not req.issubset(z.columns):raise ValueError(f'Simulation summary missing {sorted(req-set(z.columns))}')
    if 'rho_true_mean' in z.columns:
        z['rho']=pd.to_numeric(z['rho_true_mean'],errors='coerce')
    elif 'rho_true' in z.columns:
        z['rho']=pd.to_numeric(z['rho_true'],errors='coerce')
    else:
        z['rho']=z.scenario.astype(str).str.extract(r'rho_([m\d]+p\d+)')[0].map(
            lambda s:float(str(s).replace('m','-').replace('p','.')) if pd.notna(s) else np.nan)
    if 'confounding_status' not in z.columns:
        z['confounding_status']=np.where(z.scenario.astype(str).str.contains(r'__unconfounded$'),
                                          'unconfounded','confounded')
    else:
        z['confounding_status']=z.confounding_status.astype(str)
    if z.rho.isna().any():raise ValueError('Cannot determine rho for all simulation scenarios')
    valid={'unconfounded','confounded'}
    if not set(z.confounding_status).issubset(valid):
        raise ValueError(f'Unknown confounding scenario names: {set(z.confounding_status)-valid}')
    if z.duplicated(['method','rho','confounding_status']).any():
        raise ValueError('Duplicate method/rho/confounding groups in simulation_summary')
    return z


def ci(z,metric):
    name=metric+'_mean'; lo=metric+'_ci95_low'; hi=metric+'_ci95_high'
    if name not in z:raise KeyError(f'Missing simulation metric {name}')
    mean=pd.to_numeric(z[name],errors='coerce').to_numpy(dtype=float)
    if lo in z and hi in z:
        lower=pd.to_numeric(z[lo],errors='coerce').to_numpy(dtype=float)
        upper=pd.to_numeric(z[hi],errors='coerce').to_numpy(dtype=float)
    elif metric+'_sd' in z:
        sd=pd.to_numeric(z[metric+'_sd'],errors='coerce').to_numpy(dtype=float)
        n=pd.to_numeric(z['n_replicates'],errors='coerce').to_numpy(dtype=float)
        lower=mean-1.96*sd/np.sqrt(n);upper=mean+1.96*sd/np.sqrt(n)
    else:raise KeyError(f'Missing 95% CI for {metric}')
    return mean,lower,upper


def audit_methods(df,src,target,kind):
    src=Path(src)
    report={'input':str(src),'input_sha256':_sha256(src),'n_input_rows':len(df),
            'source_methods':df.method.astype(str).value_counts().to_dict(),
            'excluded_unadjusted_linear_regression':int((df.method==BAD).sum()),
            'excluded_nocov_ablation':int(df.method.astype(str).str.endswith('__nocov').sum()),
            'main_methods_present':[m for m in MAIN_METHODS if m in set(df.method)],
            'review_methods_present':[m for m in REVIEW_METHODS if m in set(df.method)],
            'kind':kind}
    write_json(target,report)
    return report


def _sha256(src):
    import hashlib
    h=hashlib.sha256()
    with Path(src).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()
