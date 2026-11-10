#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent audit of two postmortem SNP ALT-carrier groups at every data stage.

Read-only: checks actual PLINK --recode A output and precomputed pseudobulk samples.
Never guesses ALT from a1/a2; uses verified REF:ALT in variant ID.
"""
import argparse
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = {
    'rs2253588': ('6','30931600','C','G'),
    'rs35162690': ('7','43178314','T','TC'),
}
REGIONS = ('GPI','DMNX','PFC','PMC','PVC')
CELL_TYPES = ('Oli','Immune','Inh','OPC','Ast','Vast','Exc')


def normalized_donor(s):
    return re.sub(r'-(?:PVC|GPI|DMNX|PFC|PMC)[-_](?:D?WGS)$','',str(s).strip(),flags=re.I)


def checked_donors(df, label):
    if 'donor' not in df.columns:
        if 'IID' not in df.columns:
            raise ValueError(f'{label}: neither donor nor IID exists')
        df['donor'] = df['IID'].map(normalized_donor)
    else:
        df['donor'] = df['donor'].map(normalized_donor)
    if df['donor'].duplicated().any():
        d = df.loc[df.donor.duplicated(),'donor'].head(8).tolist()
        raise ValueError(f'{label}: duplicated donor IDs: {d}')
    return df


def read_definition(allele_file):
    doc=json.loads(allele_file.read_text())
    defs={}
    for rsid, expected in TARGETS.items():
        if rsid not in doc: raise KeyError(f'{rsid} absent in {allele_file}')
        r=doc[rsid]
        vid=str(r.get('bim_id','')).strip()
        counted=str(r.get('counted_allele','')).strip().upper()
        parts=tuple(v.upper() for v in vid.split(':'))
        if len(parts)!=4 or parts!=expected:
            raise ValueError(f'{rsid}: unexpected variant ID {vid!r}; expected {expected} from verified tQTL/postmortem audit')
        if counted not in (parts[2],parts[3]):
            raise ValueError(f'{rsid}: recorded PLINK counted allele {counted!r} is neither REF={parts[2]} nor ALT={parts[3]}')
        defs[rsid]={'id':vid,'ref':parts[2],'alt':parts[3],
                    'metadata_counted':counted,'raw_column':str(r.get('plink_raw_column',''))}
    return defs


def allele_convert(x, counted, ref, alt):
    """Count copies of ALT, using exact hardcalls only; no interpolation/imputation."""
    a=pd.to_numeric(x,errors='coerce').to_numpy(dtype=float)
    valid=np.isfinite(a)&(a>=0)&(a<=2)&np.isclose(a,np.rint(a),atol=1e-6)
    c=np.full(a.shape,np.nan,dtype=float)
    if counted==alt:c[valid]=a[valid]
    elif counted==ref:c[valid]=2-a[valid]
    else:raise ValueError(f'Invalid counted allele: {counted}')
    return c


def values_to_frame(df, source, counted_by_snp, definitions, status=None):
    result=pd.DataFrame({'donor':df['donor'].astype(str)})
    if status and status in df.columns:
        result['status']=df[status].astype(str).str.strip().str.title()
    elif 'status' in df.columns:
        result['status']=df['status'].astype(str).str.strip().str.title()
    else:result['status']='All'
    for rsid, d in definitions.items():
        if rsid not in df: raise KeyError(f'{source}: column {rsid} absent')
        result[rsid+'_raw']=pd.to_numeric(df[rsid],errors='coerce')
        result[rsid+'_alt']=allele_convert(df[rsid],counted_by_snp[rsid],d['ref'],d['alt'])
    return result


def stats(frame, snp, stage, region='ALL', cell='ALL', subgroup='All'):
    alt=frame[snp+'_alt'].to_numpy(float)
    eligible=(frame['status'].isin(['Case','Control']).to_numpy() if subgroup=='Case+Control' else
              frame['status'].eq(subgroup).to_numpy() if subgroup in ('Case','Control') else
              np.ones(len(frame),dtype=bool))
    a=alt[eligible]
    n0=int(np.sum(a==0));n1=int(np.sum(a==1));n2=int(np.sum(a==2))
    missing=int(np.sum(~np.isfinite(a)))
    nvalid=n0+n1+n2
    freq=float((n1+2*n2)/(2*nvalid)) if nvalid else np.nan
    return dict(stage=stage,region=region,cell_type=cell,subset=subgroup,snp=snp,
                n_donors=len(a),n_valid=nvalid,n_missing=missing,
                n_alt0=n0,n_alt1=n1,n_alt2=n2,n_noncarrier=n0,n_carrier=n1+n2,
                alt_af=freq)


def tally(rows, frame, stage, region='ALL', cell='ALL'):
    for snp in TARGETS:
        rows.append(stats(frame,snp,stage,region,cell,'All'))
        if frame.status.isin(['Case','Control']).any():
            rows.append(stats(frame,snp,stage,region,cell,'Case'))
            rows.append(stats(frame,snp,stage,region,cell,'Control'))
            rows.append(stats(frame,snp,stage,region,cell,'Case+Control'))


def styled_html(main, defs, flags, donor_trace, output):
    esc=html.escape
    head='''<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Postmortem SNP Carrier Audit</title><style>
body{font-family:Arial,"Microsoft YaHei",sans-serif;color:#243248;margin:2.1rem auto;padding:0 1.3rem;max-width:1220px;background:#fff}
h1{font-size:25px;letter-spacing:-.3px}p{line-height:1.55;color:#4c5c70}h2{font-size:18px;margin-top:2rem}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px;border-bottom:1px solid #e2e6ed;text-align:right;white-space:nowrap}th{background:#f6f8fa;position:sticky;top:0}th:first-child,td:first-child,td:nth-child(2),td:nth-child(3),td:nth-child(4){text-align:left}
.table-scroll{overflow-x:auto;max-height:620px;overflow-y:auto;border:1px solid #e4e8ef;border-radius:9px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:22px 0}.kpi{padding:15px;background:#f7f9fc;border-radius:10px;border:1px solid #e7ecf4}.kpi strong{font-size:22px;display:block;margin:6px 0}.small{font-size:12px;color:#67778c}.flag{border-left:3px solid #a06621;padding:8px 10px;margin:8px 0;background:#fffaf3}.bar{display:inline-flex;min-width:105px;width:125px;height:12px;border-radius:3px;overflow:hidden;background:#edf1f5;vertical-align:middle}.carrier{background:#3c83b4}.non{background:#c3cad6}.legend{margin:12px 0;font-size:12px}.dot{display:inline-block;height:9px;width:9px;margin:0 5px 0 10px;border-radius:2px}
</style></head><body>'''
    parts=[head,'<h1>rs2253588 / rs35162690 — Carrier 核查</h1>',
           '<p>每个数字都是<strong>独立供体数</strong>，不是细胞数。Carrier = 至少有 1 个 ALT 等位基因。原始 PLINK RAW 与筛选后的 pseudobulk 逐级独立核对。</p>']
    for snp,d in defs.items():
        rows=main[(main.snp==snp)&(main.subset=='All')&(main.stage=='PLINK_RAW')]
        last=rows.iloc[0] if len(rows) else None
        carrier='?' if last is None else int(last.n_carrier)
        non='?' if last is None else int(last.n_noncarrier)
        parts.append('<div class="kpis"><div class="kpi">'+esc(snp)+' <span class="small">ALT '+esc(d['alt'])+'</span><strong>'+str(carrier)+' / '+str(non)+'</strong><span class="small">PLINK 原始 Carrier / NonCarrier</span></div>'
                     +'<div class="kpi">REF / ALT<strong>'+esc(d['ref'])+' / '+esc(d['alt'])+'</strong><span class="small">RAW 计数等位基因 '+esc(d['raw_counted'])+'</span></div></div>')
    parts.append('<h2>逐阶段人数</h2><div class="legend"><i class="dot non" style="background:#c3cad6"></i>NonCarrier <i class="dot carrier" style="background:#3c83b4"></i>Carrier</div>')
    view=main[main.subset.isin(['All'])].copy()
    view=view[['stage','region','cell_type','snp','n_donors','n_valid','n_missing','n_alt0','n_alt1','n_alt2','n_noncarrier','n_carrier','alt_af']]
    parts.append('<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in view.columns)+'<th>两组人数</th></tr></thead><tbody>')
    for row in view.itertuples(index=False):
        values=list(row);nc=int(row.n_noncarrier);cc=int(row.n_carrier);total=max(nc+cc,1)
        bar=f'<span class="bar"><i class="non" style="width:{100*nc/total:.1f}%"></i><i class="carrier" style="width:{100*cc/total:.1f}%"></i></span>'
        cells=''.join('<td>'+esc(f'{float(v):.4f}' if isinstance(v,(np.floating,float)) and np.isfinite(v) else str(v))+'</td>' for v in values)
        parts.append('<tr>'+cells+'<td>'+bar+'</td></tr>')
    parts.append('</tbody></table></div>')
    if flags:
        parts.append('<h2>需要注意的情况</h2>')
        for s in flags:parts.append('<div class="flag">'+esc(s)+'</div>')
    parts.append('<h2>逐供体溯源</h2><p>完整个体级对照请打开 <code>carrier_audit_per_donor.csv</code>；Case/Control 及各脑区、细胞类型分布见 <code>carrier_audit_counts.csv</code>。</p>')
    parts.append('<p class="small">本页面只执行基因型计数与匹配审计，不进行差异表达拟合，也不修改任何输入文件。</p></body></html>')
    output.write_text(''.join(parts),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description='Independent ALT-carrier audit from PLINK RAW through five-region pseudobulk')
    p.add_argument('--outdir',required=True,help='Existing single_iv_causal/postmortem_snp_scDE root')
    p.add_argument('--bfile',default='',help='PLINK BED/BIM/FAM prefix, for provenance only')
    p.add_argument('--write-to',default='',help='Audit folder, defaults to OUTDIR/carrier_audit_v6')
    a=p.parse_args()
    root=Path(a.outdir); output=Path(a.write_to) if a.write_to else root/'carrier_audit_v6'
    output.mkdir(parents=True,exist_ok=True)
    cfgfile=root/'dosage_alleles.json'
    rawfile=root/'genotypes.raw'
    if not cfgfile.exists() or not rawfile.exists():
        raise FileNotFoundError(f'Need existing {cfgfile} and {rawfile}; run prepare only if these files do not exist')
    defs=read_definition(cfgfile)
    raw=pd.read_csv(rawfile,sep=r'\s+',low_memory=False)
    raw=checked_donors(raw,'genotypes.raw')
    raw_vals=pd.DataFrame({'donor':raw['donor']})
    for snp,d in defs.items():
        matches=[c for c in raw.columns if c.startswith(d['id']+'_')]
        if len(matches)!=1:
            raise ValueError(f'{snp}: expected exactly 1 RAW column for {d["id"]}: {matches}')
        col=matches[0]
        counted=col[len(d['id'])+1:].upper()
        if counted not in (d['ref'],d['alt']):
            raise ValueError(f'{snp}: RAW counted allele {counted!r} incompatible with REF/ALT')
        d['raw_counted']=counted;d['raw_column_actual']=col
        raw_vals[snp]=pd.to_numeric(raw[col],errors='coerce')
    frames={}
    frames['PLINK_RAW']=values_to_frame(raw_vals,'PLINK_RAW',{s:d['raw_counted'] for s,d in defs.items()},defs)
    flags=[]
    for snp,d in defs.items():
        if d['raw_counted']!=d['metadata_counted']:
            flags.append(f'{snp}: RAW 文件实际计数 {d["raw_counted"]}，dosage_alleles.json 声称计数 {d["metadata_counted"]}；v5 的携带者转换方向可能错误！')
    saved=root/'snp_dosages.csv'
    if saved.exists():
        pre=checked_donors(pd.read_csv(saved,low_memory=False),'snp_dosages.csv')
        frames['PREPARED_GENOTYPE']=values_to_frame(pre,'PREPARED_GENOTYPE',
                {s:d['metadata_counted'] for s,d in defs.items()},defs)
    else:flags.append('Missing snp_dosages.csv: cannot inspect prepared genotype stage')
    clin=root/'clinical_genotype_matched.csv'
    if clin.exists():
        cm=checked_donors(pd.read_csv(clin,low_memory=False),'clinical_genotype_matched.csv')
        if 'status' not in cm and 'case_control_other_latest' in cm:cm['status']=cm['case_control_other_latest']
        frames['CLINICAL_MATCHED']=values_to_frame(cm,'CLINICAL_MATCHED',
                {s:d['metadata_counted'] for s,d in defs.items()},defs)
    else:flags.append('Missing clinical_genotype_matched.csv: cannot inspect clinical matching')
    # Compare raw independent ALT dosages to previous preparation, by normalized donor IDs.
    trace=frames['PLINK_RAW'][['donor']+[s+'_raw' for s in TARGETS]+[s+'_alt' for s in TARGETS]].copy()
    trace=trace.rename(columns={s+'_raw':s+'_plink_count' for s in TARGETS})
    trace=trace.rename(columns={s+'_alt':s+'_alt_from_raw' for s in TARGETS})
    for label in ('PREPARED_GENOTYPE','CLINICAL_MATCHED'):
        if label in frames:
            f=frames[label][['donor']+[s+'_raw' for s in TARGETS]+[s+'_alt' for s in TARGETS]].copy()
            f=f.rename(columns={s+'_raw':s+('_prepared_count' if label=='PREPARED_GENOTYPE' else '_clinical_count') for s in TARGETS})
            f=f.rename(columns={s+'_alt':s+('_alt_from_saved' if label=='PREPARED_GENOTYPE' else '_alt_clinical') for s in TARGETS})
            trace=trace.merge(f,on='donor',how='left',validate='one_to_one')
    if 'CLINICAL_MATCHED' in frames:
        trace=trace.merge(frames['CLINICAL_MATCHED'][['donor','status']],on='donor',how='left',validate='one_to_one')
    for snp in TARGETS:
        for dest in ('_alt_from_saved','_alt_clinical'):
            if snp+dest in trace:
                left=trace[snp+'_alt_from_raw'];right=trace[snp+dest]
                mismatch=left.notna()&right.notna()&(~np.isclose(left.fillna(-1),right.fillna(-1)))
                if mismatch.any():flags.append(f'{snp}: PLINK_RAW vs {dest} differ in {int(mismatch.sum())} donors! Check allele orientation')
        for stage in ('PLINK_RAW','PREPARED_GENOTYPE','CLINICAL_MATCHED'):
            if stage in frames:
                x=frames[stage][snp+'_alt'].to_numpy(float)
                if np.isfinite(x).any() and not np.any(x>=1):
                    flags.append(f'{snp}: {stage} has 0 ALT carriers among {int(np.isfinite(x).sum())} valid genotypes')
                if not np.isfinite(x).any():flags.append(f'{snp}: {stage} has no valid hardcall genotypes')
    stage_rows=[]
    for stage,frame in frames.items():tally(stage_rows,frame,stage)
    for region in REGIONS:
        fpath=root/'pseudobulk'/region/'samples.csv'
        if not fpath.exists():
            flags.append(f'{region}: missing pseudobulk samples.csv (region not prepared)');continue
        x=pd.read_csv(fpath,low_memory=False)
        if 'donor' not in x or 'cell_type' not in x:raise ValueError(f'{fpath}: donor/cell_type missing')
        x['donor']=x.donor.map(normalized_donor)
        if x.duplicated(['donor','cell_type']).any():raise ValueError(f'{region}: duplicated donor/cell_type')
        if 'status' not in x:x['status']='All'
        # Data shown in final fit MUST use exact pseudobulk rows, not all 100 FAM samples.
        ff=values_to_frame(x,'pseudobulk '+region,
                {s:d['metadata_counted'] for s,d in defs.items()},defs)
        ff['cell_type']=x['cell_type'].astype(str).to_numpy()
        if 'n_cells' in x:ff['n_cells']=pd.to_numeric(x['n_cells'],errors='coerce').to_numpy()
        tally(stage_rows,ff.drop_duplicates('donor'),f'PSEUDOBULK_{region}',region)
        for cell in CELL_TYPES:
            sel=ff[ff.cell_type==cell].copy()
            tally(stage_rows,sel,f'PSEUDOBULK_{region}',region,cell)
            if len(sel)>0 and all(sel[s+'_alt'].notna().sum()>0 and (sel[s+'_alt']<1).all() for s in TARGETS):
                flags.append(f'{region}/{cell}: BOTH SNPs show 0 carriers in this cell type')
        sdon=ff.groupby('donor')['cell_type'].apply(lambda s:','.join(sorted(set(s)))).rename('celltypes_'+region)
        trace=trace.merge(sdon.reset_index(),on='donor',how='left',validate='one_to_one')
    counts=pd.DataFrame(stage_rows)
    # Other stages may contain excluded Other, so subset 'Case+Control' is authoritative for the model.
    summary=counts[(counts.subset=='Case+Control') | ((counts.subset=='All')&counts.stage.eq('PLINK_RAW'))]
    print('\n=== ORIGINAL PLINK RAW: ALL GENOTYPED DONORS ===')
    print(counts[(counts.stage=='PLINK_RAW')&(counts.subset=='All')][['snp','n_donors','n_valid','n_missing','n_alt0','n_alt1','n_alt2','n_carrier','n_noncarrier','alt_af']].to_string(index=False))
    print('\n=== CLINICAL-MATCHED / PSEUDOBULK: Case + Control ===')
    short=summary[(summary.stage!='PLINK_RAW') & (summary.cell_type=='ALL')]
    print(short[['stage','region','snp','n_valid','n_alt0','n_alt1','n_alt2','n_carrier','n_noncarrier']].to_string(index=False))
    print('\n=== FINAL PER CELL-TYPE COUNTS: Case + Control ===')
    detailed=counts[(counts.stage.str.startswith('PSEUDOBULK_'))&(counts.cell_type!='ALL')&(counts.subset=='Case+Control')]
    print(detailed[['region','cell_type','snp','n_valid','n_carrier','n_noncarrier','n_alt0','n_alt1','n_alt2']].to_string(index=False))
    if flags:
        print('\n=== AUDIT FLAGS ===')
        for f in dict.fromkeys(flags):print('[!] ',f)
    counts.to_csv(output/'carrier_audit_counts.csv',index=False)
    trace.to_csv(output/'carrier_audit_per_donor.csv',index=False)
    pd.DataFrame([{**{'snp':snp},**d} for snp,d in defs.items()]).to_csv(output/'carrier_audit_alleles.csv',index=False)
    (output/'carrier_audit_flags.txt').write_text('\n'.join(dict.fromkeys(flags))+'\n',encoding='utf-8')
    styled_html(counts,defs,list(dict.fromkeys(flags)),trace,output/'carrier_audit_dashboard.html')
    print('\n[OUTPUT]',output)
    print('[HTML]',output/'carrier_audit_dashboard.html')
    print('[CSV]',output/'carrier_audit_counts.csv')
    print('[DONOR]',output/'carrier_audit_per_donor.csv')


if __name__=='__main__':main()
