"""Read actual L=1 exposure coefficient from the all-lag models (not higher-lag coef_sum)."""
from pathlib import Path
import pandas as pd
import numpy as np

KEY=['gene','lead_snp']
def add_lag1_direction(master, fast_dir, replication_cohorts=('PD','SU')):
    x=master.copy();fast_dir=Path(fast_dir)
    for cohort,label in [('PP','PPMI')]+[(c,c.replace('+','_PLUS_')) for c in replication_cohorts]:
        f=(fast_dir/'discovery_all_IV_gene_lag_models.csv.gz' if cohort=='PP' else fast_dir/f'replication_{label}_all_lags_annotated.csv.gz')
        if not f.is_file():
            raise FileNotFoundError('Required all-lag result missing: '+str(f))
        t=pd.read_csv(f,low_memory=False)
        needed=KEY+['lag_order','coef_sum','ok']
        if any(c not in t for c in needed):raise ValueError(f'{f}: missing {set(needed)-set(t.columns)}')
        t=t.loc[pd.to_numeric(t.lag_order,errors='coerce').eq(1)].copy()
        if t.duplicated(KEY).any():raise ValueError(f'{f}: duplicate first-lag pair')
        val=pd.to_numeric(t.coef_sum,errors='coerce').where(t.ok.fillna(False).astype(bool))
        t[f'{label}_lag1_beta']=val
        x=x.merge(t[KEY+[f'{label}_lag1_beta']],on=KEY,how='left',validate='one_to_one')
        x[f'{label}_lag1_direction']=np.where(x[f'{label}_lag1_beta']>0,'positive',np.where(x[f'{label}_lag1_beta']<0,'negative','not_estimable'))
    flags=[]
    for c in replication_cohorts:
        lab=c.replace('+','_PLUS_');prefix=f'PPMI_{lab}_lag1_'
        pp=pd.to_numeric(x.PPMI_lag1_beta,errors='coerce');q=pd.to_numeric(x[f'{lab}_lag1_beta'],errors='coerce')
        good=np.isfinite(pp)&np.isfinite(q)&pp.ne(0)&q.ne(0)
        x[prefix+'evaluable']=good
        x[prefix+'direction_concordant']=pd.Series(np.where(good,np.sign(pp).eq(np.sign(q)),pd.NA),index=x.index,dtype='boolean')
        flags.append(x[prefix+'direction_concordant'])
    if flags:
        consensus=flags[0].copy()
        for s in flags[1:]:consensus=consensus & s
        x['all_replications_lag1_direction_concordant']=consensus
        x['all_replications_lag1_direction_evaluable']=pd.concat(flags,axis=1).notna().all(axis=1)
    else:
        x['all_replications_lag1_direction_concordant']=pd.Series(pd.NA,index=x.index,dtype='boolean')
        x['all_replications_lag1_direction_evaluable']=False
    x['lag1_direction_definition']='Sign of first-lag grex coefficient from a separately fitted L=1 model; independent of selected optimal lag'
    return x
