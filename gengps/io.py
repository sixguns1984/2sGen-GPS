from __future__ import annotations
from pathlib import Path
import pandas as pd

def read_table(path,index_col=None):
    p=str(path)
    if p.endswith(".parquet"):return pd.read_parquet(p)
    if p.endswith(".feather"):return pd.read_feather(p)
    return pd.read_csv(p,index_col=index_col)

def write_table(df,path):
    p=str(path);Path(p).parent.mkdir(parents=True,exist_ok=True)
    if p.endswith(".parquet"):df.to_parquet(p,index=False)
    else:df.to_csv(p,index=False)
