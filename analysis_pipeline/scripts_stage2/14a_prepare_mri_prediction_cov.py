#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mri-file',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    d=pd.read_csv(a.mri_file,usecols=['PATNO','time','age','sex','IQR'],low_memory=False)
    d['participant_id']=d.PATNO.astype(str).str.strip()
    d['cohort']='PP'
    for x in ['time','age','IQR']:d[x]=pd.to_numeric(d[x],errors='coerce')
    d=d.dropna(subset=['participant_id','time']).copy()
    if d.duplicated(['cohort','participant_id','time']).any():raise ValueError('Duplicate MRI participant/time rows; resolve before prediction')
    z=d[['cohort','participant_id','time','age','sex','IQR']]
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    z.to_csv(a.out,index=False)
    print('[MRI_COV]',len(z),'rows',z.participant_id.nunique(),'people')

if __name__=='__main__':main()
