#!/usr/bin/env python3
"""Postmortem SNP carrier-vs-noncarrier differential expression, v6.3.

Read-only source inputs; deliberately independent of Stage-I/II models.
VCF ALT carrier is the exposed comparison, NOT assumed to be Stage-I effect A1.
"""
import argparse
import gzip
import inspect
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats

ROOT_DEFAULT = '/public2/data/luojunfeng/2sGenGPS_revised_chatgpt'
SC_DIR = '/public2/data/duwenxuan/AMP_PD_anno/try'
VCF_DEFAULT = '/public/labdata/public_data/AMP_PD_Postmortem_Brain/vcf/merged.100s.vcf'
PLINK_DEFAULT = '/public/labdata/public_data/AMP_PD_Postmortem_Brain/vcf/merged.100s.new.fixed'
DONOR_METADATA_DEFAULT = '/public2/data/luojunfeng/2sTGCS_binary/scQTL/AMP_PD_Postmortem_Brain_2024_8_30_dwx.csv'
REGIONS = ('GPI','DMNX','PFC','PMC','PVC')
CELLTYPES = ('Oli','Immune','Inh','OPC','Ast','Vast','Exc')
TARGETS = {'rs2253588':('6',30931600,'G','C'), 'rs35162690':('7',43178314,'TC','T')}


def die(message):
    raise RuntimeError(message)


def norm_id(value):
    """Exactly normalize WGS sample to postmortem donor IID; no fuzzy ID truncation."""
    v = str(value).strip()
    v = re.sub(r'^0_', '', v)
    v = re.sub(r'-(?:GPI|DMNX|PFC|PMC|PVC)-DWGS$', '', v, flags=re.I)
    return v


def smart_col(columns, names, required=False):
    lookup = {str(x).lower():x for x in columns}
    for name in names:
        if name.lower() in lookup: return lookup[name.lower()]
    if required: die(f'Cannot find column among {names}; available={list(columns)[:100]}')
    return None


def open_vcf(path):
    return gzip.open(path,'rt') if str(path).endswith('.gz') else open(path,'rt')


def read_genotypes(vcf):
    """Scan raw VCF once, resolve exactly two genomic coordinates, verify REF/ALT."""
    by_pos = {(v[0],v[1]):name for name,v in TARGETS.items()}
    hits = defaultdict(list)
    samples = []
    scanned = 0
    with open_vcf(vcf) as f:
        for line in f:
            if line.startswith('##'): continue
            if line.startswith('#CHROM'):
                samples = line.rstrip('\n').split('\t')[9:]
                if len(samples) != len(set(samples)): die('Duplicate VCF sample IDs')
                continue
            if line.startswith('#'): continue
            a = line.rstrip('\n').split('\t')
            if len(a)<10: continue
            scanned += 1
            key = (a[0].removeprefix('chr'),int(a[1]))
            if key in by_pos: hits[by_pos[key]].append(a)
    if not samples: die('No VCF sample header found')
    norm = [norm_id(s) for s in samples]
    if len(norm)!=len(set(norm)): die('VCF donor normalization creates duplicate IDs')
    rows=[]
    for name,(chrom,pos,ref,alt) in TARGETS.items():
        records=hits[name]
        if len(records)!=1: die(f'{name}: expected 1 record at {chrom}:{pos}, found {len(records)}')
        a=records[0]
        if a[3].upper()!=ref or a[4].upper()!=alt:
            die(f'{name}: expected REF/ALT={ref}/{alt}, observed {a[3]}/{a[4]}; stop')
        fmt=a[8].split(':')
        if 'GT' not in fmt: die(f'{name}: VCF GT field absent')
        gi=fmt.index('GT')
        if len(a[9:])!=len(samples): die(f'{name}: sample columns mismatch')
        for raw,donor,rawfield in zip(samples,norm,a[9:]):
            gt = rawfield.split(':')[gi] if len(rawfield.split(':'))>gi else '.'
            alleles=re.split(r'[/|]',gt)
            if len(alleles)==2 and set(alleles)<=set('0123456789') and all(x in ('0','1') for x in alleles):
                dose=sum(int(x) for x in alleles)
                status=str(dose)
                gt_l='/'.join(sorted(alleles))
            else:
                dose=np.nan; status='missing'; gt_l=gt
            rows.append(dict(snp=name,chr=chrom,pos=pos,ref=ref,alt=alt,
                             vcf_sample=raw, donor=donor,gt=gt_l,alt_dose=dose,
                             carrier=(int(dose>0) if np.isfinite(dose) else np.nan),gt_status=status))
    d=pd.DataFrame(rows)
    for name,z in d.groupby('snp'):
        counts=z['gt'].value_counts(dropna=False).to_dict()
        logging.info('%s: %s (ALT carriers=%s, ALT noncarriers=%s, missing=%s)', name,counts,
                     int((z.carrier==1).sum()),int((z.carrier==0).sum()),int(z.carrier.isna().sum()))
    logging.info('VCF %s records scanned, %s sample names',scanned,len(samples))
    return d


def read_stage1_bims(root):
    rows=[]
    for lab,p in [('PPMI',Path(root)/'data/genotype/PP/amp_pd_PP_qc.bim'),
                  ('PDBP',Path(root)/'data/genotype/REP/amp_pd_REP_qc.bim'),
                  ('POSTMORTEM',Path(PLINK_DEFAULT+'.bim'))]:
        if not p.is_file():
            logging.warning('BIM missing: %s',p); continue
        with open(p) as f:
            for line in f:
                x=line.split()
                if len(x)<6: continue
                for name,(chrom,pos,ref,alt) in TARGETS.items():
                    if (lab!='POSTMORTEM' and x[1]==name) or (lab=='POSTMORTEM' and x[0].removeprefix('chr')==chrom and x[3]==str(pos)):
                        if lab!='POSTMORTEM' and (x[0].removeprefix('chr')!=chrom or x[3]!=str(pos)):
                            die(f'{name}: Stage-I {lab} BIM has unexpected chromosome/position')
                        rows.append(dict(cohort=lab,snp=name,variant_id=x[1],chrom=x[0],pos=x[3],a1=x[4],a2=x[5],
                                         pair_matches=sorted([x[4].upper(),x[5].upper()])==sorted([ref,alt]),
                                         a1_is_alt=x[4].upper()==alt))
    d=pd.DataFrame(rows)
    if len(d):
        if (~d.pair_matches).any(): die('BIM reference allele conflict; see Stage-I BIM audit')
        for (lab,name),grp in d.groupby(['cohort','snp']):
            if len(grp)!=1: die(f'Duplicate BIM hits {lab}/{name}')
    return d


def read_bed_one(prefix,idx,n,a1,a2):
    width=(n+3)//4
    with open(prefix+'.bed','rb') as f:
        if f.read(3)!=b'\x6c\x1b\x01':die('BED is not SNP-major PLINK1')
        f.seek(3+idx*width); chunk=f.read(width)
    if len(chunk)!=width:die('Truncated PLINK BED')
    decoder={0:(a1,a1),1:None,2:(a1,a2),3:(a2,a2)}
    result=[]
    for b in chunk:
        for shift in (0,2,4,6): result.append(decoder[(b>>shift)&3])
    return result[:n]


def audit_bed(g, prefix, out):
    """Verify BED against VCF by allele strings, not A1 dosage orientation."""
    if not all(Path(prefix+x).exists() for x in ('.bed','.bim','.fam')):
        logging.warning('Postmortem BED/BIM/FAM absent. VCF remains authoritative.');return
    fam=pd.read_csv(prefix+'.fam',sep=r'\s+',header=None,dtype=str)
    if fam.shape[1]<2:die('Invalid FAM')
    fids=fam.iloc[:,0].astype(str).tolist(); iids=fam.iloc[:,1].astype(str).tolist()
    lookup=defaultdict(set)
    for i,(fid,iid) in enumerate(zip(fids,iids)):
        for x in (iid,f'{fid}_{iid}'):
            lookup[norm_id(x)].add(i)
    matches={}; ambiguous=[]
    for donor in g.donor.unique():
        indices=lookup.get(donor,set())
        if len(indices)==1:matches[donor]=next(iter(indices))
        elif len(indices)>1:ambiguous.append(donor)
    if len(set(matches.values()))!=len(matches):die('Non-one-to-one FAM / VCF matching')
    if ambiguous:die(f'Ambiguous FAM sample matching: {ambiguous[:10]}')
    if len(matches)!=g.donor.nunique():
        logging.warning('VCF to PLINK FAM: matched %s/%s; remaining samples not genotype-validated',len(matches),g.donor.nunique())
    results=[]
    with open(prefix+'.bim') as f:
        for i,line in enumerate(f):
            a=line.split()
            if len(a)<6:continue
            for name,(chrom,pos,ref,alt) in TARGETS.items():
                if a[0].removeprefix('chr')!=chrom or a[3]!=str(pos):continue
                if sorted([a[4].upper(),a[5].upper()])!=sorted([ref,alt]):die(f'{name}: PLINK alleles disagree')
                bed=read_bed_one(prefix,i,len(fam),a[4].upper(),a[5].upper())
                for row in g[g.snp==name].itertuples(index=False):
                    if row.donor not in matches:continue
                    got=bed[matches[row.donor]]
                    expected=None if not np.isfinite(row.alt_dose) else tuple([ref]*(2-int(row.alt_dose))+[alt]*int(row.alt_dose))
                    eq=(got is None and expected is None) or (got is not None and expected is not None and Counter(got)==Counter(expected))
                    results.append(dict(snp=name,donor=row.donor,vcf_gt=row.gt,plink_gt='/'.join(got) if got else 'missing',agree=eq))
    audit=pd.DataFrame(results)
    audit.to_csv(out/'vcf_plink_concordance.csv',index=False)
    if audit.empty:logging.warning('No BED genotype pairs comparable; BED concordance not established')
    else:
        for name,grp in audit.groupby('snp'):
            logging.info('%s: BED concordance %s/%s incl missing',name,int(grp.agree.sum()),len(grp))
        if (~audit.agree).any():die('VCF vs BED genotype discordance; see vcf_plink_concordance.csv')


def summarize_gt(g, out):
    rows=[]
    for name,z in g.groupby('snp'):
        valid=z.alt_dose.notna().sum()
        alt=z.alt_dose.sum()
        rows.append(dict(snp=name,vcf_n=len(z),n_valid=int(valid),n_missing=int(z.alt_dose.isna().sum()),
                         gt_00=int((z.alt_dose==0).sum()),gt_01=int((z.alt_dose==1).sum()),
                         gt_11=int((z.alt_dose==2).sum()),alt_allele_frequency=alt/(2*valid) if valid else np.nan,
                         n_carrier=int((z.carrier==1).sum()),n_noncarrier=int((z.carrier==0).sum()),
                         n_carrier_expected_reference={'rs2253588':40,'rs35162690':47}[name]))
    d=pd.DataFrame(rows);d.to_csv(out/'vcf_gt_summary.csv',index=False)
    for r in d.itertuples(index=False):
        if r.n_carrier!=r.n_carrier_expected_reference:
            die(f'Unexpected genotype carrier count for {r.snp}; verify VCF and reference build')
    return d


def stage1_pair_links(root,out):
    path=Path(root)/'single_iv_causal/iv_validation/single_iv_primary_eligible.csv'
    if not path.exists():
        logging.warning('Stage-II eligible IV-gene table absent: %s',path);return
    d=pd.read_csv(path,low_memory=False)
    variant=smart_col(d.columns,['lead_snp','Variant_ID','variant','snp','rsid'])
    gene=smart_col(d.columns,['gene','Ensembl_ID','gene_id','gene_name'])
    if variant is None or gene is None:
        logging.warning('Eligible IV table columns not recognized: %s',list(d.columns));return
    z=d[d[variant].astype(str).isin(TARGETS)].copy()
    z.to_csv(out/'stage1_stage2_target_snp_IV_genes.csv',index=False)
    logging.info('Stage-I/II eligible IV-gene pair matches for two SNPs: %s',len(z))


def command_audit(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    g=read_genotypes(a.vcf)
    g.to_csv(out/'vcf_genotypes.csv',index=False)
    summarize_gt(g,out)
    b=read_stage1_bims(a.root)
    if not b.empty:b.to_csv(out/'stage1_postmortem_BIM_audit.csv',index=False)
    stage1_pair_links(a.root,out)
    audit_bed(g,a.plink_prefix,out)
    logging.info('SNP audit completed: %s',out)


def case_label(x):
    s=str(x).strip().lower()
    if s in ('case','pd','parkinson disease','parkinson\'s disease','parkinsons disease','patient','1','true'):
        return 'Case'
    if s in ('control','ct','hc','healthy','healthy control','non-pd','non_pd','0','false'):
        return 'Control'
    return np.nan


def sex_label(x):
    s=str(x).strip().lower()
    if s in ('m','male','1'):return 'M'
    if s in ('f','female','2'):return 'F'
    return np.nan


def get_covariates(obs, a):
    col_iid=smart_col(obs.columns,[a.id_col,'IID','iid','donor_id'],required=True)
    col_ct=smart_col(obs.columns,[a.cell_col,'cell_type','celltype'],required=True)
    col_age=smart_col(obs.columns,[a.age_col,'age_at_baseline','age','Age','age_at_death','age_death'])
    col_sex=smart_col(obs.columns,[a.sex_col,'sex','Sex','gender','Gender'])
    col_case=smart_col(obs.columns,[a.case_col,'case_control_other_latest','Case/Control','case_control','disease','Disease','diagnosis','status','condition','PD_status'])
    d=pd.DataFrame({'donor':obs[col_iid].astype(str).map(norm_id).values,
                    'cell_type':obs[col_ct].astype(str).values})
    d['age']=pd.to_numeric(obs[col_age],errors='coerce').to_numpy() if col_age else np.nan
    d['sex']=obs[col_sex].astype(str).map(sex_label).to_numpy() if col_sex else np.nan
    d['case']=obs[col_case].astype(str).map(case_label).to_numpy() if col_case else np.nan
    d['cell_idx']=np.arange(len(d))
    logging.info('obs columns used: IID=%s cell=%s age=%s sex=%s case=%s',col_iid,col_ct,col_age,col_sex,col_case)
    return d


def gene_map(var, with_collapse=False):
    """Aggregate repeated gene_name features by summing RAW counts, preserving Ensembl provenance.

    Each published gene has one output row, with all contributing source Ensembl IDs.
    Missing/invalid symbols are retained as individual normalized Ensembl IDs.
    The returned sparse projection maps input H5AD features to unique output genes.
    """
    col=smart_col(var.columns,['gene_name','gene_names','gene_symbol','gene_symbols','feature_name','symbol','feature','genename','hgnc_symbol'])
    raw_id=var.index.astype(str).to_numpy()
    ensembl=pd.Series(raw_id).str.replace(r'\.\d+$','',regex=True).to_numpy()
    symbols=var[col].astype(str).str.strip().to_numpy() if col else np.repeat('',len(raw_id))
    invalid={'','nan','none','na','null','n/a'}
    names=np.array([v if v.lower() not in invalid else eid for eid,v in zip(ensembl,symbols)],dtype=object)
    # Use gene symbols as gene keys for unambiguous aggregation across duplicated symbols.
    # Retain all original IDs in an additional column for Stage-I Ensembl crosswalk.
    dat=pd.DataFrame({'gene_id':names,'gene_symbol':symbols,'source_ensembl_id':ensembl,'raw_var_id':raw_id})
    codes,uniques=pd.factorize(dat.gene_id,sort=False)
    grouping=dat.groupby('gene_id',sort=False)
    genes=grouping.agg(gene_symbol=('gene_symbol','first'),
                       ensembl_ids=('source_ensembl_id',lambda x:'|'.join(dict.fromkeys(x.astype(str)))),
                       raw_var_ids=('raw_var_id',lambda x:'|'.join(dict.fromkeys(x.astype(str)))),
                       n_merged_features=('raw_var_id','size')).reset_index()
    n_merged=int(len(dat)-len(genes))
    logging.info('Gene symbol column=%s; annotated=%d/%d; unique genes=%d; merged duplicate features=%d',
                 col, sum(v.lower() not in invalid for v in symbols),len(dat),len(genes),n_merged)
    if not with_collapse:return genes
    projection=sparse.csr_matrix((np.ones(len(codes),dtype=np.int64),
                                  (np.arange(len(codes)),codes)),
                                 shape=(len(codes),len(uniques)))
    if genes.gene_id.astype(str).tolist()!=uniques.astype(str).tolist():
        die('Gene aggregation ordering mismatch')
    return genes,projection


def check_raw(mat):
    data=mat.data if sparse.issparse(mat) else np.asarray(mat).ravel()
    if len(data):
        if not np.all(np.isfinite(data)) or data.min() < 0 or not np.allclose(data,np.rint(data),atol=1e-5,rtol=0):
            die('Matrix is not nonnegative integer raw counts; NO pseudobulk on normalized expression!')


def donor_cov(meta):
    rows=[]
    for donor,z in meta.groupby('donor',sort=True):
        row={'donor':donor}
        for field in ('age','sex','case'):
            v=z[field].dropna().unique()
            if len(v)>1:die(f'Conflicting {field} for donor {donor}: {v}')
            row[field]=v[0] if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def apply_external_cov(meta, csv):
    if not csv:return meta
    x=pd.read_csv(csv)
    logging.info('Using actual postmortem donor metadata %s: %s rows, columns=%s',csv,len(x),list(x.columns))
    c=smart_col(x.columns,['donor','IID','iid','donor_id'],required=True)
    x['donor']=x[c].astype(str).map(norm_id)
    if x.donor.duplicated().any():die('Duplicate donor metadata ID')
    for field in ('sex','case'):
        meta[field]=meta[field].astype(object)
    logging.info('Clinical case counts (raw): %s', x[smart_col(x.columns,['case_control_other_latest','case','Case/Control'],required=True)].astype(str).value_counts().to_dict())
    for dst,src in [('age',['age_at_baseline','age','Age']),('sex',['SEX','sex','gender']),('case',['case_control_other_latest','case','Case/Control','diagnosis','status','disease'])]:
        actual=smart_col(x.columns,src)
        if not actual:die(f'Postmortem donor clinical CSV {csv} lacks required {dst} column; columns={list(x.columns)}')
        val=pd.to_numeric(x[actual],errors='coerce') if dst=='age' else x[actual].map(sex_label if dst=='sex' else case_label)
        lut=dict(zip(x.donor,val))
        observed=meta.donor.map(lut)
        nonmissing=observed.notna()
        overlap=nonmissing & meta[dst].notna()
        if overlap.any() and not (meta.loc[overlap,dst].astype(str).values == observed.loc[overlap].astype(str).values).all():
            die(f'Conflicting obs and external metadata for {dst}')
        meta.loc[nonmissing,dst]=observed[nonmissing]
    matched=meta[['donor','age','sex','case']].drop_duplicates('donor')
    logging.info('Postmortem clinical merge: %s donors; age=%s sex=%s case/control=%s', len(matched),int(matched.age.notna().sum()),int(matched.sex.notna().sum()),int(matched['case'].isin(['Case','Control']).sum()))
    return meta


def prepared_region_state(regdir):
    """Check complete prepare artifacts, also accepting complete v6 legacy regions.

    A region directory alone is NOT proof that all seven cell types were saved.
    """
    regdir=Path(regdir)
    required=[regdir/'gene_mapping.csv',regdir/'donor_covariates.csv',
              regdir/'input_columns.json',regdir/'unmatched_sc_donors.csv']
    for ct in CELLTYPES:
        required.extend([regdir/ct/'donors.csv',regdir/ct/'counts.npz'])
    missing=[str(x.relative_to(regdir)) for x in required if not x.is_file() or x.stat().st_size==0]
    if missing:return False,missing
    # Force rebuilding pre-v6.2 pseudobulk because gene_name aggregation and donor
    # clinical metadata are now explicit, unlike incomplete earlier versions.
    marker=regdir/'.prepared_complete.json'
    if not marker.is_file():return False,['.prepared_complete.json (v6.2)']
    try:
        info=json.loads(marker.read_text())
        if info.get('schema_version')!='v6.2':return False,['v6.2 schema/version mismatch']
        head=pd.read_csv(regdir/'gene_mapping.csv',nrows=2)
        if not {'gene_id','gene_symbol','ensembl_ids','n_merged_features'}.issubset(head.columns):
            return False,['gene_mapping.csv missing v6.2 merge columns']
    except Exception as ex:return False,[f'v6.2 metadata check failed: {ex}']
    return True,[]


def resolve_covariate_source(a, genotype_donors):
    """Resolve to real donor metadata; never infer clinical labels from SNPs."""
    if a.donor_metadata and a.donor_metadata.lower() != 'auto':
        path=Path(a.donor_metadata)
        if not path.is_file():die(f'Explicit donor metadata not found: {path}')
        return str(path)
    audit=Path(a.out)/'metadata_audit'
    cached=audit/'canonical_donor_metadata.csv'
    if cached.is_file() and not getattr(a,'force_metadata_scan',False):
        check=pd.read_csv(cached)
        if set(('donor','age','sex','case')).issubset(check.columns) and \
           set(genotype_donors).issubset(set(check.donor.astype(str))):
            logging.info('Reusing verified clinical metadata: %s', cached)
            return str(cached)
    from donor_metadata_resolver import discover_metadata
    roots=[Path(a.h5ad_dir),Path(a.h5ad_dir).parent,
           Path(VCF_DEFAULT).parent.parent,
           Path(a.root)/'data/metadata',
           Path(a.root)/'single_iv_causal/postmortem_snp_scDE']
    roots += list(map(Path,getattr(a,'metadata_root',[]) or []))
    logging.warning('Clinical covariates absent from H5AD.obs. Searching actual donor-level tables under: %s',roots)
    try:
        path,summary=discover_metadata(genotype_donors,roots,audit)
    except Exception as ex:
        die('Clinical metadata discovery failed: '+str(ex)+
            '. Review metadata_audit/candidate_table_audit.csv and provide '
            '--donor-metadata /actual/file.csv or --metadata-root /real/clinical/data/dir')
    logging.info('Real donor metadata matched: %s (complete donors=%s)',path,summary['n_full_covariates'])
    return str(path)


def prepare_region(region,a,g):
    import anndata as ad
    scpath=Path(a.h5ad_dir)/f'{region}_subtype_count_try.h5ad'
    if not scpath.exists():die(f'Missing H5AD {scpath}')
    logging.info('Opening backed %s',scpath)
    obj=ad.read_h5ad(scpath,backed='r')
    try:
        meta=get_covariates(obj.obs,a)
        if any(not meta.loc[meta.cell_type.isin(CELLTYPES),f].notna().any() for f in ('age','sex','case')):
            source=resolve_covariate_source(a,g.donor.dropna().unique())
            meta=apply_external_cov(meta,source)
        elif a.donor_metadata and a.donor_metadata.lower()!='auto':
            meta=apply_external_cov(meta,a.donor_metadata)
        genes,projection=gene_map(obj.var,with_collapse=True)
        regdir=Path(a.out)/'pseudobulk'/region
        regdir.mkdir(parents=True,exist_ok=True)
        marker=regdir/'.prepared_complete.json'
        # Invalidate previous completion on every re-preparation, including interrupted runs.
        marker.unlink(missing_ok=True)
        genes.to_csv(regdir/'gene_mapping.csv',index=False)
        (regdir/'input_columns.json').write_text(json.dumps({'obs_columns':list(obj.obs.columns),
            'var_columns':list(obj.var.columns),'counts_layer':a.counts_layer or ('counts' if 'counts' in obj.layers else 'X'),
            'clinical_metadata':a.donor_metadata, 'clinical_columns':{'id':'IID','age':'age_at_baseline','sex':'SEX','case':'case_control_other_latest'},
            'other_excluded':True},indent=2))
        matched=set(g.donor)
        meta['is_matched']=meta.donor.isin(matched)
        meta['included']=meta.is_matched & meta.cell_type.isin(CELLTYPES) & meta['case'].isin(['Case','Control'])
        for field in ('age','sex','case'):
            if not meta.loc[meta.included,field].notna().any():
                die(f'{region}: donor covariate {field} absent/unrecognized among matched cells. Specify --donor-metadata CSV (IID,age,sex,case) or inspect obs columns; no unadjusted model will be run.')
        donor_summary=donor_cov(meta.loc[meta.included])
        donor_summary.to_csv(regdir/'donor_covariates.csv',index=False)
        all_donors=sorted(meta.loc[meta.included,'donor'].unique())
        logging.info('%s: %s nuclei; %s retained donor IDs; %s unmatched nuclei; %s Other/unlisted nuclei',
            region,len(meta),len(all_donors),int((~meta.is_matched).sum()),
            int((meta.is_matched & ~meta.cell_type.isin(CELLTYPES)).sum()))
        unmatched=meta.loc[~meta.is_matched,'donor'].value_counts().rename_axis('donor').reset_index(name='n_cells')
        unmatched.to_csv(regdir/'unmatched_sc_donors.csv',index=False)
        if not all_donors:die(f'{region}: no matched donors; inspect IDs')
        donor_index={x:i for i,x in enumerate(all_donors)}
        M={ct:np.zeros((len(all_donors),len(genes)),dtype=np.int64) for ct in CELLTYPES}
        cells={ct:Counter() for ct in CELLTYPES}
        layer=a.counts_layer or ('counts' if 'counts' in obj.layers else 'X')
        matrix=obj.X if layer=='X' else obj.layers[layer]
        if a.counts_layer and layer!='X' and layer not in obj.layers:die(f'Unknown counts layer {layer}')
        for lo in range(0,obj.n_obs,a.chunk_size):
            hi=min(obj.n_obs,lo+a.chunk_size)
            zz=meta.iloc[lo:hi]
            active=zz.included.to_numpy()
            if not active.any():continue
            # Backed rows are read once; sparse multiplication aggregates many cells at once.
            chunk=matrix[lo:hi]
            check_raw(chunk)
            if not sparse.issparse(chunk):chunk=sparse.csr_matrix(chunk)
            else:chunk=chunk.tocsr()
            chunk=(chunk@projection).tocsr()
            for ct in CELLTYPES:
                mask=active & zz.cell_type.eq(ct).to_numpy()
                loc=np.flatnonzero(mask)
                if not len(loc):continue
                indices=np.array([donor_index[v] for v in zz.donor.to_numpy()[loc]],dtype=np.int32)
                unique,inv=np.unique(indices,return_inverse=True)
                ind=sparse.csr_matrix((np.ones(len(loc),dtype=np.int64),(inv,loc)),
                                      shape=(len(unique),hi-lo))
                summed=(ind @ chunk).toarray()
                M[ct][unique,:]+=summed.astype(np.int64)
                for ix,count in zip(*np.unique(indices,return_counts=True)):
                    cells[ct][all_donors[ix]]+=int(count)
            if (lo//a.chunk_size+1)%20==0:
                logging.info('%s %s/%s nuclei processed',region,hi,obj.n_obs)
        cov=donor_summary.set_index('donor')
        for ct in CELLTYPES:
            dest=regdir/ct
            dest.mkdir(exist_ok=True)
            ncell=np.array([cells[ct][d] for d in all_donors])
            take=np.flatnonzero(ncell>=a.min_cells)
            donor_df=cov.reindex(np.array(all_donors)[take]).reset_index()
            donor_df['n_cells']=ncell[take]
            donor_df.to_csv(dest/'donors.csv',index=False)
            sparse.save_npz(dest/'counts.npz',sparse.csr_matrix(M[ct][take,:]))
            logging.info('%s/%s: %s donors at min_cells=%s',region,ct,len(take),a.min_cells)
        del M
        # Verify saved artifacts directly before atomically committing the v6.2 marker.
        required=[regdir/'gene_mapping.csv',regdir/'donor_covariates.csv',regdir/'input_columns.json',regdir/'unmatched_sc_donors.csv']
        for _ct in CELLTYPES:required.extend([regdir/_ct/'donors.csv',regdir/_ct/'counts.npz'])
        missing=[str(x.relative_to(regdir)) for x in required if not x.is_file() or x.stat().st_size==0]
        if missing:die(f'{region}: pseudobulk output incomplete: {missing}')
        marker_tmp=regdir/'.prepared_complete.json.tmp'
        marker_tmp.write_text(json.dumps({'schema_version':'v6.2','region':region,'n_genes':len(genes),
                            'n_cells_input':int(obj.n_obs),'min_cells':int(a.min_cells),
                            'h5ad_file':str(scpath),'donor_metadata':a.donor_metadata,'merged_gene_name':True,'complete':True},indent=2))
        marker_tmp.replace(marker)
        logging.info('%s: preparation COMPLETE; %s features',region,len(genes))
    finally:
        obj.file.close()


def command_prepare(a):
    out=Path(a.out)
    p=out/'vcf_genotypes.csv'
    if not p.exists():die('Run audit before prepare')
    g=pd.read_csv(p)
    for region in a.regions.split(','):
        region=region.strip()
        if region not in REGIONS:die(f'Unknown region {region}')
        regdir=out/'pseudobulk'/region
        ready,_=prepared_region_state(regdir)
        if getattr(a,'skip_prepared',False) and ready:
            logging.info('%s already prepared: skip (--skip-prepared)',region)
            continue
        prepare_region(region,a,g)
    write_group_audit(a)


def write_group_audit(a):
    out=Path(a.out)
    g=pd.read_csv(out/'vcf_genotypes.csv')
    rows=[]
    for region in REGIONS:
        for ct in CELLTYPES:
            p=out/'pseudobulk'/region/ct/'donors.csv'
            if not p.exists():continue
            m=pd.read_csv(p).merge(g[['snp','donor','alt_dose','carrier']],on='donor')
            for snp,z in m.groupby('snp'):
                for stratum,zs in [('all',z),('case_only',z[z['case']=='Case'])]:
                    valid=zs[zs.carrier.notna()]
                    rows.append(dict(region=region,cell_type=ct,snp=snp,stratum=stratum,
                        donors_with_cells=len(zs),n_carrier=int((valid.carrier==1).sum()),
                        n_noncarrier=int((valid.carrier==0).sum()),n_gt_missing=int(zs.carrier.isna().sum()),
                        age_nonmissing=int(valid.age.notna().sum()),sex_nonmissing=int(valid.sex.notna().sum()),
                        case_nonmissing=int(valid['case'].notna().sum())))
    pd.DataFrame(rows).to_csv(out/'matched_group_counts.csv',index=False)


def bh(p):
    p=np.asarray(p,dtype=float);out=np.full_like(p,np.nan)
    keep=np.flatnonzero(np.isfinite(p))
    if not len(keep):return out
    order=keep[np.argsort(p[keep])]
    rank=np.arange(1,len(order)+1)
    out[order]=np.minimum(1,np.minimum.accumulate((p[order]*len(order)/rank)[::-1])[::-1])
    return out


def design_for(meta,stratum,min_group):
    m=meta.copy()
    if stratum=='case_only':m=m[m['case']=='Case'].copy()
    else:m=m[m['case'].isin(['Case','Control'])].copy()
    m=m.dropna(subset=['carrier','age','sex'])
    m=m[m['sex'].isin(['M','F'])].copy()
    m['carrier']=m.carrier.astype(int)
    n1=int((m.carrier==1).sum());n0=int((m.carrier==0).sum())
    if min(n1,n0)<min_group:return None,f'group_too_small: carrier={n1} noncarrier={n0}'
    X=pd.DataFrame({'intercept':1.,'carrier':m.carrier.astype(float).values,
                    'age_z':((m.age.astype(float)-m.age.astype(float).mean()) / (m.age.astype(float).std(ddof=0) or 1.)).values},index=m.index)
    if m.sex.nunique()>1:X['sex_M']=(m.sex=='M').astype(float)
    if stratum=='all' and m['case'].nunique()>1:X['case_PD']=(m['case']=='Case').astype(float)
    if len(X)<=X.shape[1]+2:return None,'insufficient_residual_df'
    if np.linalg.matrix_rank(X.to_numpy())<X.shape[1]:return None,'rank_deficient_design'
    return (m,X),None


def hc3(counts,genes,X):
    counts=np.asarray(counts,dtype=np.float64)
    libs=counts.sum(axis=1)
    if np.any(libs<=0):die('Zero-library pseudobulk retained; filter at preparation')
    y=np.log2((counts+0.5)/(libs[:,None]+1.0)*1e6)
    x=X.to_numpy(dtype=float);n,p=x.shape
    inv=np.linalg.inv(x.T@x)
    coef=inv@x.T@y
    residual=y-x@coef
    leverage=np.einsum('ij,jk,ik->i',x,inv,x)
    w=(x@inv[:,1])/(1-leverage)
    se=np.sqrt(np.sum((w[:,None]*residual)**2,axis=0))
    beta=coef[1]
    stat=np.divide(beta,se,out=np.full_like(beta,np.nan),where=se>1e-12)
    pv=2*stats.t.sf(np.abs(stat),df=n-p)
    return pd.DataFrame({'gene_id':genes,'log2FC':beta,'SE_HC3':se,'stat':stat,'pvalue':pv,'padj':bh(pv)})


def run_deseq2(counts,genes,metadata,X,n_cpus):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    meta=pd.DataFrame(index=metadata.index)
    meta['carrier']=metadata.carrier.map({0:'0',1:'1'}).astype(str)
    # Keep the same standardized age covariate but avoid factor-name underscores
    # (PyDESeq2 0.4.x silently rewrites these and emits a warning).
    meta['ageScaled']=X['age_z'].astype(float)
    factors=['carrier','ageScaled']
    if 'sex_M' in X:
        meta['sex']=metadata.sex.astype(str)
        factors.append('sex')
    if 'case_PD' in X:
        meta['case']=metadata['case'].astype(str)
        factors.append('case')
    cnt=pd.DataFrame(np.asarray(counts,dtype=np.int64),index=meta.index,columns=genes)
    dds=DeseqDataSet(counts=cnt,metadata=meta,design_factors=factors,
                     continuous_factors=['ageScaled'],ref_level=['carrier','0'],
                     refit_cooks=True,n_cpus=n_cpus,quiet=True)
    dds.deseq2()
    # PyDESeq2 0.4.11/0.4.12: DeseqStats accepts `inference` but NOT `n_cpus`.
    # Some other versions accept n_cpus. Inspect the installed signature rather
    # than assuming a version string or retrying a full DESeq2 fit on TypeError.
    ds_sig=inspect.signature(DeseqStats).parameters
    ds_kwargs={'contrast':['carrier','1','0'],'quiet':True}
    if 'inference' in ds_sig and getattr(dds,'inference',None) is not None:
        ds_kwargs['inference']=dds.inference
    elif 'n_cpus' in ds_sig:
        ds_kwargs['n_cpus']=n_cpus
    ds=DeseqStats(dds,**ds_kwargs)
    ds.summary()
    res=ds.results_df.copy();res['gene_id']=res.index.astype(str)
    return res[['gene_id','baseMean','log2FoldChange','lfcSE','stat','pvalue','padj']].rename(
        columns={'log2FoldChange':'log2FC','lfcSE':'SE_DESeq2'})


def validated_existing_de(file, marker, gene_ids, prefix, method, n_carrier, n_noncarrier):
    """Return number of FDR hits only when file belongs to CURRENT prepare.

    Historical CSVs from before v6.2 must not be reused after regeneration.
    All expected gene IDs and sample-count/contrast metadata are checked.
    """
    if not file.is_file() or file.stat().st_size==0 or not marker.is_file():
        return None
    # If prepare was regenerated after this fit, the old DE is no longer valid.
    if file.stat().st_mtime < marker.stat().st_mtime:
        return None
    required=['gene_id','method','stratum','snp','region','cell_type',
              'n_carrier','n_noncarrier','n_genes_tested','contrast','padj']
    try:
        z=pd.read_csv(file,usecols=required,low_memory=False)
        if len(z)!=len(gene_ids) or not np.array_equal(z.gene_id.astype(str).to_numpy(),gene_ids):
            return None
        expected=dict(prefix,method=method,n_carrier=n_carrier,n_noncarrier=n_noncarrier,
                      n_genes_tested=len(gene_ids),contrast='VCF_ALT_carrier_vs_noncarrier')
        for key,value in expected.items():
            if not bool((z[key].astype(str)==str(value)).all()):
                return None
        return int((pd.to_numeric(z.padj,errors='coerce')<0.05).sum())
    except (OSError,ValueError,KeyError,pd.errors.ParserError):
        return None


def persist_fit_status(out, rows):
    """Write status incrementally so Ctrl+C preserves the completed groups."""
    path=out/'fit_status.csv'
    temp=out/'fit_status.csv.tmp'
    pd.DataFrame(rows).to_csv(temp,index=False)
    temp.replace(path)


def command_fit(a):
    out=Path(a.out)
    g=pd.read_csv(out/'vcf_genotypes.csv')
    methods=['hc3','deseq2'] if a.methods=='both' else [a.methods]
    # Validate/repair ALL requested regions before fitting anything. Never silently
    # fit a partial region set and label it as a five-region analysis.
    regions=[r.strip() for r in a.regions.split(',') if r.strip()]
    for region in regions:
        if region not in REGIONS:die(f'Unknown region {region}')
        regdir=out/'pseudobulk'/region
        ready, missing=prepared_region_state(regdir)
        if not ready:
            if getattr(a,'auto_prepare_missing',False):
                logging.warning('%s incomplete (%s); preparing automatically',region,', '.join(missing[:12]))
                prepare_region(region,a,g)
                ready, missing=prepared_region_state(regdir)
                if not ready:die(f'{region}: preparation still incomplete: {missing}')
            else:
                die(f'{region}: pseudobulk preparation MISSING/INCOMPLETE: {missing[:12]}. '
                    f'Run: bash scripts_snp_scDE/run_snp_scDE.sh prepare --regions {region} '
                    f'--out {out} ; or rerun fit with --auto-prepare-missing. '
                    'The same --out and --min-cells must be used throughout.')
    audit=[]
    for region in regions:
        regdir=out/'pseudobulk'/region
        genes=pd.read_csv(regdir/'gene_mapping.csv').fillna({'gene_symbol':''})
        for ct in CELLTYPES:
            dest=regdir/ct
            if not (dest/'donors.csv').exists():continue
            m0=pd.read_csv(dest/'donors.csv')
            counts=sparse.load_npz(dest/'counts.npz').toarray()
            if len(counts)!=len(m0):die('Pseudobulk metadata/counts sample mismatch')
            if len(counts)==0:continue
            for name in TARGETS:
                gg=g[g.snp==name][['donor','carrier','alt_dose']]
                m=m0.merge(gg,on='donor',how='left',validate='one_to_one')
                m.index=pd.Index(np.arange(len(m)))
                for stratum in ('all','case_only'):
                    prepared,why=design_for(m,stratum,a.min_group)
                    prefix=dict(region=region,cell_type=ct,snp=name,stratum=stratum)
                    if prepared is None:
                        audit.append(dict(**prefix,status='SKIPPED',reason=why));continue
                    m2,X=prepared
                    raw=counts[m2.index,:]
                    good=(raw.sum(axis=0)>=a.min_total_count)&((raw>=a.min_per_donor_count).sum(axis=0)>=a.min_detected_donors)
                    good &= np.isfinite(raw).all(axis=0)
                    if not np.any(good):
                        audit.append(dict(**prefix,status='SKIPPED',reason='no_expressed_genes'));continue
                    gene_ids=genes.gene_id.astype(str).to_numpy()[good]
                    sub_counts=raw[:,good]
                    md=m2.copy();md.index=md.donor.astype(str)
                    X.index=md.index
                    for method in methods:
                        file=out/'de'/stratum/name/region/ct/f'{method}.csv'
                        file.parent.mkdir(parents=True,exist_ok=True)
                        n_carrier=int((md.carrier==1).sum())
                        n_noncarrier=int((md.carrier==0).sum())
                        if getattr(a,'resume',False):
                            previous_hits=validated_existing_de(
                                file,regdir/'.prepared_complete.json',gene_ids,
                                prefix,method,n_carrier,n_noncarrier)
                            if previous_hits is not None:
                                audit.append(dict(**prefix,method=method,status='PASS',
                                    reason='RESUMED_EXISTING_CSV',n_genes=int(good.sum()),
                                    n_carrier=n_carrier,n_noncarrier=n_noncarrier,n_fdr05=previous_hits))
                                persist_fit_status(out,audit)
                                logging.info('Reused %s/%s/%s/%s/%s: %d genes',
                                             stratum,name,region,ct,method,int(good.sum()))
                                continue
                        try:
                            if method=='hc3':result=hc3(sub_counts,gene_ids,X)
                            else:result=run_deseq2(sub_counts,gene_ids,md,X,a.n_cpus)
                            result['gene_id']=result.gene_id.astype(str)
                            result=result.merge(genes[['gene_id','gene_symbol']],on='gene_id',how='left',validate='one_to_one')
                            result.insert(0,'method',method)
                            for key,value in reversed(list(prefix.items())):result.insert(0,key,value)
                            result['n_carrier']=int((md.carrier==1).sum())
                            result['n_noncarrier']=int((md.carrier==0).sum())
                            result['n_genes_tested']=int(good.sum())
                            result['contrast']='VCF_ALT_carrier_vs_noncarrier'
                            temp_file=file.with_suffix('.csv.tmp')
                            result.to_csv(temp_file,index=False)
                            temp_file.replace(file)
                            audit.append(dict(**prefix,method=method,status='PASS',reason='',n_genes=int(good.sum()),
                                n_carrier=int((md.carrier==1).sum()),n_noncarrier=int((md.carrier==0).sum()),
                                n_fdr05=int((result.padj<0.05).sum())))
                            persist_fit_status(out,audit)
                            logging.info('Fitted %s/%s/%s/%s/%s: %d genes',stratum,name,region,ct,method,int(good.sum()))
                        except Exception as ex:
                            audit.append(dict(**prefix,method=method,status='ERROR',reason=repr(ex)))
                            persist_fit_status(out,audit)
                            logging.exception('%s / %s %s %s %s failed',method,name,region,ct,stratum)
                            if a.fail_fast:raise
    persist_fit_status(out,audit)
    if any(x['status']=='ERROR' for x in audit):
        die('One or more fit jobs failed; see fit_status.csv and log')
    logging.info('Fit done: %s',out/'fit_status.csv')


def command_lmm(a):
    import anndata as ad
    import statsmodels.api as sm
    out=Path(a.out)
    geno=pd.read_csv(out/'vcf_genotypes.csv')
    summary=[]
    for region in a.regions.split(','):
        scpath=Path(a.h5ad_dir)/f'{region}_subtype_count_try.h5ad'
        obj=ad.read_h5ad(scpath,backed='r')
        try:
            meta=get_covariates(obj.obs,a)
            if any(not meta.loc[meta.cell_type.isin(CELLTYPES),f].notna().any() for f in ('age','sex','case')):
                source=resolve_covariate_source(a,geno.donor.dropna().unique())
                meta=apply_external_cov(meta,source)
            elif a.donor_metadata and a.donor_metadata.lower()!='auto':
                meta=apply_external_cov(meta,a.donor_metadata)
            genes_lmm,projection=gene_map(obj.var,with_collapse=True)
            geneids=genes_lmm.gene_id.astype(str).tolist()
            gene_index={name:i for i,name in enumerate(geneids)}
            layer=a.counts_layer or ('counts' if 'counts' in obj.layers else 'X')
            matrix=obj.X if layer=='X' else obj.layers[layer]
            rng=np.random.default_rng(a.seed)
            for ct in CELLTYPES:
                cm=meta[meta.cell_type==ct]
                if cm.empty:continue
                for snp in TARGETS:
                    for stratum in ('all','case_only'):
                        # Rank by donor-level HC3 if available, else PyDESeq2.
                        paths=[out/'de'/stratum/snp/region/ct/(x+'.csv') for x in ('hc3','deseq2')]
                        file=next((p for p in paths if p.exists()),None)
                        if file is None:continue
                        hits=pd.read_csv(file).sort_values('pvalue',na_position='last').head(a.top_n)
                        if hits.empty:continue
                        gm=geno[geno.snp==snp][['donor','carrier']]
                        sub=cm.merge(gm,on='donor',how='inner')
                        if stratum=='case_only':sub=sub[sub['case']=='Case']
                        else:sub=sub[sub['case'].isin(['Case','Control'])]
                        sub=sub.dropna(subset=['carrier','age','sex'])
                        if sub.empty:continue
                        # Downsample independently by donor, not by cell (reproducible).
                        chosen=[]
                        for donor,z in sub.groupby('donor'):
                            if len(z)<a.min_cells:continue
                            selected=rng.choice(z.index.to_numpy(),size=min(len(z),a.max_cells_per_donor),replace=False)
                            chosen.extend(selected.tolist())
                        sampled=sub.loc[chosen].copy()
                        if sampled.empty:continue
                        donor_meta=sampled.drop_duplicates('donor')[['donor','carrier','age','sex','case']].reset_index(drop=True)
                        design,why=design_for(donor_meta,stratum,a.min_group)
                        if design is None:
                            logging.warning('LMM %s/%s/%s/%s skipped %s',region,ct,snp,stratum,why);continue
                        donor_ok=set(design[0].donor)
                        sampled=sampled[sampled.donor.isin(donor_ok)].sort_values('cell_idx')
                        feat=[(gid,gene_index[gid]) for gid in hits.gene_id.astype(str) if gid in gene_index]
                        if not feat:continue
                        # Compute library size from whole-gene raw counts for each selected cell;
                        # full matrix chunking avoids advanced backed indexing assumptions.
                        chosen_idx=sampled.cell_idx.astype(int).to_numpy()
                        target_gene_indices=[i for _,i in feat]
                        vals=np.zeros((len(chosen_idx),len(feat)),dtype=float)
                        libs=np.zeros(len(chosen_idx),dtype=float)
                        for lo in range(0,len(obj.obs),a.chunk_size):
                            hi=min(lo+a.chunk_size,len(obj.obs))
                            i0=np.searchsorted(chosen_idx,lo);i1=np.searchsorted(chosen_idx,hi)
                            if i0==i1:continue
                            block=matrix[lo:hi]
                            check_raw(block)
                            local=chosen_idx[i0:i1]-lo
                            submat=block[local]
                            libs[i0:i1]=np.asarray(submat.sum(axis=1)).ravel()
                            selected=(submat @ projection[:,target_gene_indices])
                            vals[i0:i1]=selected.toarray() if sparse.issparse(selected) else selected
                        valid=libs>0
                        sampled=sampled.iloc[np.flatnonzero(valid)].copy()
                        vals=vals[valid];libs=libs[valid]
                        if sampled.donor.nunique()<2*a.min_group:continue
                        vals=np.log1p((vals/libs[:,None])*1e4)
                        for k,(gid,_) in enumerate(feat):
                            try:
                                df=sampled[['donor','carrier','age','sex','case']].copy()
                                df['y']=vals[:,k]
                                df['age_z']=(df.age-df.age.mean())/(df.age.std(ddof=0) or 1.)
                                df['carrier']=df.carrier.astype(float)
                                fixed=['carrier','age_z']
                                if df.sex.nunique()>1:
                                    df['sex_M']=(df.sex=='M').astype(float);fixed+=['sex_M']
                                if stratum=='all' and df['case'].nunique()>1:
                                    df['case_PD']=(df['case']=='Case').astype(float);fixed+=['case_PD']
                                exog=sm.add_constant(df[fixed],has_constant='add')
                                if np.linalg.matrix_rank(exog.to_numpy())<exog.shape[1]:continue
                                model=sm.MixedLM(df.y,exog,groups=df.donor)
                                result=model.fit(reml=False,method='lbfgs',disp=False,maxiter=200)
                                summary.append(dict(snp=snp,region=region,cell_type=ct,stratum=stratum,gene_id=gid,
                                    beta_carrier=float(result.params['carrier']),
                                    pvalue=float(result.pvalues['carrier']),n_cells=len(df),n_donors=df.donor.nunique(),
                                    converged=bool(result.converged),scale='log1p(counts_per_1e4)'))
                            except Exception as ex:
                                summary.append(dict(snp=snp,region=region,cell_type=ct,stratum=stratum,gene_id=gid,
                                                    error=repr(ex),converged=False))
        finally:
            obj.file.close()
    result=pd.DataFrame(summary)
    if not result.empty:
        result['padj_within_group']=np.nan
        for _,idx in result.groupby(['snp','region','cell_type','stratum']).groups.items():
            result.loc[idx,'padj_within_group']=bh(result.loc[idx,'pvalue'])
    result.to_csv(out/'cell_LMM_top_genes.csv',index=False)
    logging.info('Cell LMM finished: %d rows',len(result))


def summarize_stage1_gene_coverage(a):
    out=Path(a.out)
    candidates=out/'stage1_stage2_target_snp_IV_genes.csv'
    if not candidates.exists():return
    pairs=pd.read_csv(candidates)
    if pairs.empty:return
    gene_col=smart_col(pairs.columns,['gene','Ensembl_ID','gene_id','gene_name'])
    variant_col=smart_col(pairs.columns,['lead_snp','Variant_ID','variant','snp','rsid'])
    if gene_col is None or variant_col is None:return
    gene_list=[]
    for region in REGIONS:
        p=out/'pseudobulk'/region/'gene_mapping.csv'
        if not p.exists():continue
        features=pd.read_csv(p,dtype=str).fillna('')
        index={}
        for _,f in features.iterrows():
            index[str(f['gene_id'])]=f
            for ensembl in str(f['ensembl_ids']).split('|'):
                if ensembl:index[ensembl]=f
        for snp,gid in pairs[[variant_col,gene_col]].itertuples(index=False,name=None):
            gid=re.sub(r'\.\d+$','',str(gid))
            rec=index.get(gid)
            gene_list.append(dict(snp=snp,region=region,stage1_gene_id=gid,
                                  in_sc_h5ad=rec is not None,
                                  h5ad_gene_symbol=str(rec['gene_symbol']) if rec is not None else '',
                                  merged_features=int(rec['n_merged_features']) if rec is not None else 0))
    pd.DataFrame(gene_list).drop_duplicates().to_csv(out/'stage1_target_gene_presence_in_sc.csv',index=False)


def command_summarize(a):
    out=Path(a.out)
    files=sorted((out/'de').glob('*/*/*/*/*.csv'))
    if not files:die('No DE files found; run fit')
    frames=[]
    for file in files:
        x=pd.read_csv(file)
        if x.empty:continue
        frames.append(x)
    if not frames:die('All fit outputs empty')
    z=pd.concat(frames,ignore_index=True)
    z.to_csv(out/'ALL_genes_DE_results.csv',index=False)
    z['fdr05']=z.padj<0.05
    hits=z[z.fdr05].sort_values('padj')
    hits.to_csv(out/'FDR05_significant_DE_genes.csv',index=False)
    top=z.sort_values(['snp','region','cell_type','stratum','method','pvalue']).groupby(
        ['snp','region','cell_type','stratum','method'],sort=False).head(30)
    top.to_csv(out/'Top30_DE_genes_by_group.csv',index=False)
    z.groupby(['snp','region','cell_type','stratum','method'],dropna=False).agg(
        n_genes=('gene_id','size'),n_fdr05=('fdr05','sum'),min_p=('pvalue','min'),
        min_fdr=('padj','min'),n_carrier=('n_carrier','first'),n_noncarrier=('n_noncarrier','first')).reset_index().to_csv(
            out/'DE_group_summary.csv',index=False)
    summarize_stage1_gene_coverage(a)
    print('DE results:',len(z),'rows; FDR<0.05:',len(hits),'\nFolder:',out)


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['audit','prepare','fit','cell-lmm','summarize','all'])
    p.add_argument('--root',default=os.getenv('ROOT',ROOT_DEFAULT))
    p.add_argument('--out',default=None)
    p.add_argument('--vcf',default=VCF_DEFAULT)
    p.add_argument('--plink-prefix',default=PLINK_DEFAULT)
    p.add_argument('--h5ad-dir',default=SC_DIR)
    p.add_argument('--regions',default=','.join(REGIONS))
    p.add_argument('--id-col',default='IID')
    p.add_argument('--cell-col',default='cell_type')
    p.add_argument('--age-col',default='age_at_baseline')
    p.add_argument('--sex-col',default='SEX')
    p.add_argument('--case-col',default='case_control_other_latest')
    p.add_argument('--donor-metadata',default=DONOR_METADATA_DEFAULT,help='confirmed 2026-11-02 AMP-PD postmortem clinical table (IID,age_at_baseline,SEX,case_control_other_latest)')
    p.add_argument('--metadata-root',action='append',default=[],help='additional real directory of postmortem donor-level tables; repeatable')
    p.add_argument('--force-metadata-scan',action='store_true',help='ignore existing metadata audit cache and rescan source tables')
    p.add_argument('--counts-layer',default='',help='default counts layer if present, else X')
    p.add_argument('--chunk-size',type=int,default=3000)
    p.add_argument('--min-cells',type=int,default=30)
    p.add_argument('--min-group',type=int,default=5)
    p.add_argument('--min-total-count',type=int,default=10)
    p.add_argument('--min-per-donor-count',type=int,default=2)
    p.add_argument('--min-detected-donors',type=int,default=3)
    p.add_argument('--methods',choices=['hc3','deseq2','both'],default='both')
    p.add_argument('--n-cpus',type=int,default=4)
    p.add_argument('--top-n',type=int,default=30)
    p.add_argument('--max-cells-per-donor',type=int,default=30)
    p.add_argument('--seed',type=int,default=20261103)
    p.add_argument('--fail-fast',action='store_true')
    p.add_argument('--resume',action='store_true',
                   help='fit: skip only validated DE CSVs generated after current v6.2 pseudobulk prepare')
    p.add_argument('--skip-prepared',action='store_true',
                   help='prepare: skip already completed pseudobulk regions (v6 artifacts accepted)')
    p.add_argument('--auto-prepare-missing',action='store_true',
                   help='fit: explicitly permit preparing missing/incomplete H5AD regions before DE')
    return p


def main():
    a=build_parser().parse_args()
    if a.out is None:a.out=str(Path(a.root)/'single_iv_causal/postmortem_snp_scDE/carrier_vs_noncarrier_v6_vcf_verified')
    os.environ.setdefault('OMP_NUM_THREADS','1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    commands={'audit':command_audit,'prepare':command_prepare,'fit':command_fit,
              'cell-lmm':command_lmm,'summarize':command_summarize}
    if a.command=='all':
        for fn in commands.values():fn(a)
    else:commands[a.command](a)


if __name__=='__main__':main()
