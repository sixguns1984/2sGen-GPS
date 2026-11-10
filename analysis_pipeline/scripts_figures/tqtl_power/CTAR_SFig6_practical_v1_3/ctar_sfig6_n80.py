#!/usr/bin/env python3
"""CTAR-MPTGA prospective sample-size figure (8-df overall; 6-df interaction).

Scientific definition:
* prospective: PVE anchored at MAF 0.20, not fitted from significant discoveries;
* exact OU precision for irregular visits, plug-in known rho (not profiled rho);
* noncentral chi-square large-sample likelihood-ratio power approximation;
* temporal discovery/replication independent, joint power is their product;
* discovery threshold anchored to observed scan's raw BH cutoff (conditional forecast);
* N80/80: first equal per-cohort N where >=80% loci have >=80% joint power.

This deliberately does not claim to rerun the production GPU CTAR scan. See README.
"""
from __future__ import annotations
import argparse, json, math, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2, ncx2
from numpy.polynomial.legendre import leggauss
import matplotlib
matplotlib.use('cairo')  # Use Cairo backend for high-quality output
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'svg.fonttype':'none', 'pdf.fonttype':42, 'ps.fonttype':42})

COLORS = {'navy':'#214D70','blue':'#387EAC','teal':'#188D8B','gold':'#D49A43','red':'#B95258','gray':'#6D7886','purple':'#7867A0'}

def acol(df, choices, required=True):
    names={str(c).lower().replace('_','').replace('-',''):c for c in df.columns}
    for key in choices:
        k=key.lower().replace('_','').replace('-','')
        if k in names:return names[k]
    if required:raise ValueError('Cannot determine column from '+str(choices)+'; available: '+str(list(df.columns)[:60]))
    return None

def meta_templates(path,cohort_col=None,donor_col=None,time_col=None):
    md=pd.read_parquet(path)
    cohort=cohort_col or acol(md,['cohort','cohort_name','study','dataset','source','project'])
    donor=donor_col or acol(md,['participant_id','participant','patient_id','subject_id','individual_id','donor_id','subject','IID'])
    tcol=time_col or acol(md,['followup_years','followup_time_year','followup_time_years','followup_time','time_years','time_from_baseline_years','time_from_baseline','years_from_baseline','visit_time_years','visit_time','time_since_baseline','visit_year','followup','time','months_since_baseline','visit_months','days_since_baseline','visit_days'])
    data=md[[cohort,donor,tcol]].copy();data.columns=['cohort','donor','time']
    data['cohort']=data['cohort'].astype(str).str.upper().str.strip()
    data['cohort']=data['cohort'].replace({'PP':'PPMI','PD':'PDBP','PARKINSONS PROGRESSION MARKERS INITIATIVE':'PPMI'})
    data['cohort']=data['cohort'].map(lambda x: 'PPMI' if 'PPMI' in x else ('PDBP' if 'PDBP' in x else x))
    data['time']=pd.to_numeric(data['time'],errors='coerce')
    if 'month' in tcol.lower():data['time']/=12.0
    if 'day' in tcol.lower():data['time']/=365.25
    data=data.dropna(subset=['donor','time']).drop_duplicates(['cohort','donor','time'])
    groups={}
    for key in ('PPMI','PDBP'):
        z=data[data.cohort==key]
        if z.empty:raise ValueError(f'Missing {key} from metadata; cohort values={data.cohort.value_counts().head(20).to_dict()}')
        groups[key]=[np.sort(g.time.to_numpy(dtype=float)) for _,g in z.groupby('donor')]
        if any(np.diff(t).min(initial=1.)<=0 for t in groups[key]):raise ValueError('Nonunique visit times')
    return groups

def ordered_templates(n=800,seed=3):
    rng=np.random.default_rng(seed)
    a=[];b=[]
    for _ in range(n):
        t=np.array([0.,1.,2.,3.,4.])
        t[1:]+=rng.normal(0,.10,4)
        t[1:]=np.sort(np.maximum(t[1:],0.15))
        t=t[np.r_[True,rng.random(4)>.14]]
        a.append(t)
        t2=np.array([0.,1.,2.,3.,4.]); t2[1:]+=rng.normal(0,.15,4)
        t2[1:]=np.sort(np.maximum(t2[1:],.15))
        t2=t2[np.r_[True,rng.random(4)>.20]]
        b.append(t2)
    return {'PPMI':a,'PDBP':b}

def planned_templates(m,n=1500,missing=.15,jitter=.08,seed=9,horizon=4.):
    rng=np.random.default_rng(seed+int(m*102));out=[]
    for _ in range(n):
        t=np.linspace(0,horizon,m)
        t[1:]+=rng.normal(0,jitter,m-1)
        t[1:]=np.sort(np.maximum(t[1:],.05))
        keep=np.r_[True,rng.random(m-1)>missing]
        out.append(t[keep])
    return out

def mode_values(t,pattern,horizon=4.):
    x=np.asarray(t)/horizon
    if pattern=='linear':return x
    if pattern=='late':return 3*x*x-2*x*x*x
    if pattern=='early':return 3*x-3*x*x+x*x*x
    if pattern=='static':return np.ones_like(x)
    raise ValueError(pattern)

def ou_grams(templates,rho,pattern,horizon=4.):
    """Return expected OU weighted f'f, 1'f, 1'1 per donor, and raw f^2 per visit.
    rho is correlation per ONE year; whitening is mathematically exact for OU.
    """
    qff=qf1=q11=f2=obs=0.
    for t in templates:
        t=np.asarray(t,dtype=float)
        if len(t)==0:continue
        f=mode_values(t,pattern,horizon)
        if len(t)>1:
            dt=np.diff(t)
            if np.any(dt<=0):raise ValueError('Visit times must strictly increase')
            a=np.power(rho,dt) if rho>0 else np.zeros_like(dt)
            denom=np.maximum(1.-a*a,1e-12)
            wf=np.r_[f[0],(f[1:]-a*f[:-1])/np.sqrt(denom)]
            w1=np.r_[1.,(1.-a)/np.sqrt(denom)]
        else:wf=f;w1=np.ones(1)
        qff+=float(np.dot(wf,wf));qf1+=float(np.dot(wf,w1));q11+=float(np.dot(w1,w1))
        f2+=float(np.dot(f,f));obs+=len(t)
    nd=len(templates)
    if nd==0 or f2<=0:raise ValueError('Invalid longitudinal templates / effect pattern')
    return (qff/nd,qf1/nd,q11/nd,f2/obs)

def ncps(N,maf,pve,g_disc,g_rep,retain=1.,ref_maf=.2,f2_reference=None):
    """Noncentralities of full 8-df and dynamic 6-df tests.
    PVE is defined for a five-visit reference schedule at MAF=0.20 and unit residual variance.
    """
    maf=np.asarray(maf,float)
    v=np.maximum(2*maf*(1-maf),0)/(2*ref_maf*(1-ref_maf))
    amp2=pve/(1-pve)
    f2_reference = g_disc[3] if f2_reference is None else f2_reference
    def one(g,scale=1.,n=None):
        ff, f1, i1, f2=g
        total=n*v*amp2*scale*scale*(ff/f2_reference)
        dyn=n*v*amp2*scale*scale*(max(0.,ff-f1*f1/max(i1,1e-12))/f2_reference)
        return np.maximum(total,0),np.minimum(np.maximum(dyn,0),np.maximum(total,0))
    ndisc,nrep = (N,N) if np.isscalar(N) else N
    return one(g_disc,1.,ndisc),one(g_rep,retain,nrep)

def overall_power(lam,p):
    return ncx2.sf(chi2.isf(p,8),8,lam)

_GL_X,_GL_W=leggauss(24)
def dynamic_joint_power(lam_total,lam_dynamic,p_overall,p_interaction):
    """P(chi2_8>c8 and chi2_6>c6), accounting for nested tests.
    Exact under orthogonal Gaussian model: chi2_8=independent chi2_2 + chi2_6.
    """
    a=np.asarray(lam_total,float)
    d=np.asarray(lam_dynamic,float)
    c8=float(chi2.isf(p_overall,8));c6=float(chi2.isf(p_interaction,6))
    if c6>=c8:return ncx2.sf(c6,6,d)
    static=np.maximum(a-d,0.)
    # Integral over [c6,c8] + P(D6 >= c8), 24-point Gauss-Legendre.
    mid=(c6+c8)/2;half=(c8-c6)/2
    out=ncx2.sf(c8,6,d)
    for z,w in zip(_GL_X,_GL_W):
        x=mid+half*z
        out=out+half*w*ncx2.pdf(x,6,d)*ncx2.sf(c8-x,2,static)
    return np.clip(out,0,1)

def compute_power(N,maf,pve,gd,gr,alpha_disc,alpha_repl=.05,alpha_inter=None,retain=1.,test='temporal',f2_reference=None,min_minor_hom=5):
    maf=np.asarray(maf,float)
    (td,dd),(tr,dr)=ncps(N,maf,pve,gd,gr,retain=retain,f2_reference=f2_reference)
    if test=='temporal':
        pdsc=overall_power(td,alpha_disc);prep=overall_power(tr,alpha_repl)
    elif test=='dynamic':
        if alpha_inter is None:raise ValueError('Dynamic ctAR requires stage-wise interaction threshold')
        pdsc=dynamic_joint_power(td,dd,alpha_disc,alpha_inter)
        prep=dynamic_joint_power(tr,dr,alpha_repl,alpha_repl)
    else:raise ValueError(test)
    # CTAR uses two independent hard-call genotype channels. Near-empty minor-homozygote
    # channel cannot support the nominal 8-df full-rank genetic contrast.
    if min_minor_hom>0:
        nd,nr=(N,N) if np.isscalar(N) else N
        valid_d=(nd*(maf**2)>=min_minor_hom) & (nd*2*maf*(1-maf)>=min_minor_hom)
        valid_r=(nr*(maf**2)>=min_minor_hom) & (nr*2*maf*(1-maf)>=min_minor_hom)
        pdsc=np.where(valid_d,pdsc,0.);prep=np.where(valid_r,prep,0.)
    return pdsc,prep,pdsc*prep

def find_min(ns,vals,goal=.8):
    ok=np.flatnonzero(np.asarray(vals)>=goal-1e-12)
    return float(np.asarray(ns)[ok[0]]) if len(ok) else float('nan')

# Columns are intentionally explicit for the actual CTAR Stage-I output:
# Variant_ID, Ensembl_ID, Pvalue, Interaction_Pvalue, beta_g_0..7.
_VARIANT_COLS=['Variant_ID','variant_id','snp_id','SNP','variant','rsid','ID']
_GENE_COLS=['Ensembl_ID','gene_id','gene','phenotype_id','gene_name','geneid']
_MAF_COLS=['MAF','minor_allele_frequency','minor_allele_freq','maf_pp',
           'allele_frequency','af','eaf']
_P_COLS=['Pvalue','Overall_Pvalue','Temporal_Pvalue','P_value','P',
         'tqtl_pvalue','overall_p','pvalue_overall','discovery_p']


def read_locus_candidates(path):
    """Project-exact schema mapping; retain all eligible pairs until MAF QC."""
    f=Path(path)
    if f.suffix.lower() in ('.csv','.tsv','.txt'):
        sep='\t' if f.suffix.lower() in ('.tsv','.txt') else ','
        src=pd.read_csv(f,sep=sep,low_memory=False)
        schema=src.columns.tolist()
    else:
        import pyarrow.parquet as pq
        schema=pq.ParquetFile(f).schema_arrow.names
        src=None
    sample_schema=pd.DataFrame(columns=schema)
    v=acol(sample_schema,_VARIANT_COLS)
    g=acol(sample_schema,_GENE_COLS)
    mafcol=acol(sample_schema,_MAF_COLS,False)
    pcol=acol(sample_schema,_P_COLS,False)
    keep=list(dict.fromkeys([v,g]+([mafcol] if mafcol else [])+([pcol] if pcol else [])))
    if src is None:src=pd.read_parquet(f,columns=keep)
    z=src[keep].dropna(subset=[v,g]).copy()
    z=z.rename(columns={v:'variant_id',g:'gene_id',
                       **({mafcol:'maf'} if mafcol else {}),
                       **({pcol:'discovery_p'} if pcol else {})})
    z['variant_id']=z['variant_id'].astype(str).str.strip()
    z['gene_id']=z['gene_id'].astype(str).str.strip()
    z=z.loc[(z.variant_id!='')&(z.gene_id!='')&
            (~z.variant_id.str.lower().isin(['nan','none']))&
            (~z.gene_id.str.lower().isin(['nan','none']))].copy()
    z=z.drop_duplicates(['variant_id','gene_id']).reset_index(drop=True)
    if 'maf' not in z:z['maf']=np.nan
    z['maf']=pd.to_numeric(z['maf'],errors='coerce')
    z['maf']=np.where(z.maf>.5,1.-z.maf,z.maf)
    print(f'[INPUT] PPMI CTAR: {len(z):,} unique variant-gene pairs; '
          f'variant={v}; gene={g}; rawP={pcol}; MAF={mafcol or "PLINK needed"}',flush=True)
    return z


def normalize_variant(x):
    s=str(x).lower().strip()
    m=re.match(r'^(?:chr)?(\d{1,2}|x|y|m|mt)[_:\-](\d+)(?:[_:\-]([acgt]+)[_:\-]([acgt]+))?(?:$|[:_\-].*)',s)
    if not m:return None
    chrom=m.group(1)
    if chrom=='m':chrom='mt'
    alleles=frozenset((m.group(3),m.group(4))) if m.group(3) and m.group(4) else None
    return (chrom,int(m.group(2)),alleles)


def read_plink_index(prefix,requested_variants=None):
    """Stream potentially multi-million-row WGS BIM without loading it all.

    If requested_variants is supplied, only matching SNP IDs and potentially
    matching genomic positions are indexed. Row numbers remain original BIM
    indexes, which is mandatory for correct BED column reads.
    """
    pref=Path(prefix)
    bim_path=Path(str(pref)+'.bim')
    if not bim_path.is_file():
        raise FileNotFoundError('PPMI PLINK BIM missing: '+str(bim_path))
    requested=None;wanted_pos=None
    if requested_variants is not None:
        requested={str(v).strip() for v in requested_variants}
        wanted_pos=set()
        for var in requested:
            norm=normalize_variant(var)
            if norm is not None:wanted_pos.add(norm[:2])
    direct={};ambiguous=set();positions={}
    n=0
    with bim_path.open(encoding='utf-8') as stream:
        for i,line in enumerate(stream):
            toks=line.split()
            if len(toks)<6:
                raise ValueError(f'Invalid BIM line {i+1}: expected 6 columns in {bim_path}')
            n+=1
            chrom,ident,bp=toks[0].lower().removeprefix('chr'),toks[1],toks[3]
            if chrom=='m':chrom='mt'
            try:pos=int(bp)
            except ValueError:
                raise ValueError(f'Invalid BIM position at line {i+1}: {bp}')
            use_id=requested is None or ident in requested
            use_pos=wanted_pos is None or (chrom,pos) in wanted_pos
            if use_id:
                if ident in direct:ambiguous.add(ident)
                else:direct[ident]=i
            if use_pos:
                key=(chrom,pos)
                positions.setdefault(key,[]).append((i,frozenset((toks[4].lower(),toks[5].lower()))))
    for ident in ambiguous:direct.pop(ident,None)
    if requested is not None:
        print(f'[MAF] Scanned {n:,} BIM rows; retained {len(direct):,} direct IDs '
              f'and {len(positions):,} requested positions.',flush=True)
    return direct,positions,n


def match_plink_variants(variants, direct,positions):
    """Exact SNP ID first; unambiguous chr:pos fallback with allele validation."""
    result={}
    for var in variants:
        v=str(var).strip()
        ix=direct.get(v)
        if ix is None:
            obj=normalize_variant(v)
            if obj is not None:
                chrom,bp,alleles=obj
                hits=positions.get((chrom,bp),[])
                if alleles is not None:
                    hits=[(k,a) for k,a in hits if a==alleles]
                if len(hits)==1:ix=hits[0][0]
        result[v]=ix
    return result


def add_plink_maf(z,prefix,plink_index=None):
    if not prefix:
        raise ValueError('CTAR output contains no MAF. Specify the real PPMI PLINK --plink-prefix.')
    direct,positions,bim_count=(plink_index if plink_index is not None else
                                read_plink_index(prefix,z.variant_id.drop_duplicates()))
    lookup=match_plink_variants(z.variant_id.drop_duplicates(),direct,positions)
    indices=z.variant_id.map(lookup)
    not_matched=int(indices.isna().sum())
    needed=np.flatnonzero((~np.isfinite(z.maf.to_numpy(dtype=float))) & indices.notna().to_numpy())
    print(f'[MAF] PLINK BIM {bim_count:,} variants; candidates {len(z):,}; '
          f'unmatched variant-gene pairs {not_matched:,}; genotype MAF needed {len(needed):,}',flush=True)
    if not len(needed):return z
    bed_path=Path(str(prefix)+'.bed')
    if not bed_path.is_file():raise FileNotFoundError('PPMI PLINK BED missing: '+str(bed_path))
    try:
        from bed_reader import open_bed
    except ImportError as e:
        raise ImportError('Install bed-reader using pip install -r requirements.txt to calculate real PPMI MAF') from e
    bed=open_bed(str(bed_path))
    unique=np.unique(indices.iloc[needed].to_numpy(dtype=int))
    frequencies={}
    for block in np.array_split(unique,max(1,math.ceil(len(unique)/128))):
        d=bed.read(index=(slice(None),block),dtype=np.float32)
        if d.ndim==1:d=d[:,None]
        for col,k in enumerate(block):
            val=d[:,col]
            val=val[np.isfinite(val)]
            if val.size:
                af=float(val.mean()/2)
                frequencies[int(k)]=min(af,1.-af)
    # Align MAF back by original row position, never by DataFrame index label.
    z=z.reset_index(drop=True).copy()
    for row in needed:
        ix=indices.iat[row]
        if int(ix) in frequencies:z.iat[row,z.columns.get_loc('maf')]=frequencies[int(ix)]
    print(f'[MAF] Retrieved frequencies for {len(frequencies):,} distinct PLINK variants; '
          f'usable candidate pairs: {int(z.maf.between(.005,.5).sum()):,}',flush=True)
    return z


def sample_loci(path,seed,n_pairs=1500,minimum=1000,plink_prefix=None):
    """Always sample *after* SNP ID mapping and MAF QC, not before.

    Sampling before MAF recovery can mistakenly turn valid >1500 pairs into a
    <1000 usable-pair failure. For massive result sets, recover MAF from a
    seeded candidate pool, expanding deterministically if necessary.
    """
    if n_pairs<minimum:raise ValueError('Requested n_pairs must be >= min_pairs')
    z=read_locus_candidates(path)
    if len(z)<minimum:
        raise ValueError(f'Only {len(z)} unique CTAR loci; need >= {minimum}')
    rng=np.random.default_rng(seed)
    z=z.iloc[rng.permutation(len(z))].reset_index(drop=True)
    # With a computed MAF column, sample immediately among valid loci.
    if z.maf.notna().sum()==len(z):
        eligible=z[z.maf.between(.005,.5)]
    else:
        if not plink_prefix:
            raise ValueError('MAF unavailable: supply original PPMI PLINK --plink-prefix')
        direct,positions,bim_count=read_plink_index(plink_prefix,z.variant_id.drop_duplicates())
        plink_index=(direct,positions,bim_count)
        matched=match_plink_variants(z.variant_id.drop_duplicates(),direct,positions)
        can_match=z.variant_id.map(matched).notna()
        z=z.loc[can_match | z.maf.notna()].reset_index(drop=True)
        print(f'[INPUT] PPMI PLINK BIM: {bim_count:,}; candidate QTL pairs with matching SNP IDs '
              f'or existing MAF: {len(z):,}',flush=True)
        if len(z)<minimum:
            raise ValueError(f'Only {len(z)} unique CTAR SNP-gene pairs match PPMI PLINK BIM; '
                             f'need >= {minimum}. Inspect Variant_ID convention vs BIM IDs, '
                             'chromosome/position/alleles and the chosen genotype prefix.')
        # Use increasingly large candidate pools; never accept a small random
        # pool as evidence that the full QTL universe has insufficient MAF.
        target=min(len(z),max(n_pairs*3,minimum*3))
        while True:
            eligible=add_plink_maf(z.iloc[:target].copy(),plink_prefix,plink_index=plink_index)
            eligible=eligible[eligible.maf.between(.005,.5)]
            if len(eligible)>=n_pairs or target==len(z):break
            target=min(len(z),max(target+2000,target*2))
    if len(eligible)<minimum:
        raise ValueError(f'Only {len(eligible)} QTL pairs have verified real PPMI MAF 0.005–0.5; '
                         f'minimum required {minimum}. Review the [INPUT]/[MAF] audit above.')
    # Rows are randomly permuted before filtering, giving seeded reproducibility.
    loci=eligible.head(n_pairs).copy().reset_index(drop=True)
    loci.attrs['loci_audit']={'requested_loci':int(n_pairs),
                              'usable_candidates_checked':int(len(eligible)),
                              'retained_loci':int(len(loci)),
                              'distinct_genes':int(loci.gene_id.nunique()),
                              'maf_min':float(loci.maf.min()),
                              'maf_max':float(loci.maf.max())}
    print(f'[INPUT] Final sampled loci: {len(loci):,}; unique genes: '
          f'{loci.gene_id.nunique():,}; MAF median: {loci.maf.median():.4f}',flush=True)
    return loci


def load_input_loci(path,seed,n,minimum,plink_prefix):
    return sample_loci(path,seed,n,minimum,plink_prefix)

def _flat_json_numbers(obj):
    """Find named numeric metadata without assuming a particular JSON nesting."""
    out = {}
    def walk(value):
        if isinstance(value, dict):
            for key, val in value.items():
                if isinstance(val, (dict, list)):
                    walk(val)
                else:
                    canonical = re.sub(r"[^a-z0-9]", "", str(key).lower())
                    if canonical not in out:
                        try:
                            v = float(val)
                            if np.isfinite(v):out[canonical] = v
                        except (TypeError, ValueError):
                            pass
        elif isinstance(value, list):
            for entry in value:walk(entry)
    walk(obj)
    return out


def _first_numeric(flat, names):
    for name in names:
        canonical = re.sub(r"[^a-z0-9]", "", name.lower())
        if canonical in flat:return flat[canonical]
    return None


def find_threshold_json(directory):
    """Return raw BH cutoff, stagewise interaction alpha, G and R from CTAR finalization."""
    path = Path(directory) / 'fdr_finalize_summary.json'
    if not path.is_file():return None, None, None, None, None
    j = json.loads(path.read_text(encoding='utf-8'))
    flat = _flat_json_numbers(j)
    G = _first_numeric(flat, ['hierarchical_screening_n_tests_G', 'G', 'n_finite', 'n_finite_tests', 'n_finite_overall_tests',
                              'n_tested','n_total_tests','total_tests','n_total_finite_tests'])
    R = _first_numeric(flat, ['hierarchical_screening_n_rejected_R', 'R', 'n_screened', 'screened_count', 'n_significant_tqtl',
                              'n_screened_tqtl','n_bh_discoveries','n_discovered'])
    ai = _first_numeric(flat, ['hierarchical_confirmation_alpha','alpha_ii', 'alpha_stage2','alpha_interaction',
                               'interaction_alpha','confirmation_alpha', 'stagewise_alpha_ii'])
    cutoff = _first_numeric(flat, ['overall_bh_p_cutoff', 'bh_raw_p_threshold','bh_raw_p_cutoff',
                                   'bh_p_cutoff', 'overall_p_threshold', 'screening_p_cutoff',
                                   'p_cutoff', 'bh_cutoff', 'p_bh_cutoff',
                                   'bh_threshold', 'bh_raw_cutoff', 'raw_p_cutoff'])
    if ai is None and G and R:ai = 0.05*R/G
    if cutoff is not None and not (0 < cutoff <= 0.05):
        raise ValueError(f'Invalid raw BH P cutoff={cutoff} in {path}; do not substitute FDR 0.05')
    if ai is not None and not (0 < ai <= 0.05):
        raise ValueError(f'Invalid alpha_II={ai} in {path}')
    if G is not None and R is not None and not (0 < R <= G):
        raise ValueError(f'Invalid G/R in {path}: G={G}, R={R}')
    return cutoff, ai, G, R, str(path)


def _parquet_pvals(path):
    """Read only the overall raw P column from the complete CTAR discovery Parquet."""
    import pyarrow.dataset as ds
    dset=ds.dataset(str(path),format='parquet')
    candidates=['Overall_Pvalue','Temporal_Pvalue','Pvalue','P_value','P',
                'tqtl_pvalue','overall_p','pvalue_overall']
    col=acol(pd.DataFrame(columns=dset.schema.names),candidates,False)
    if col is None:
        raise ValueError(f'Cannot locate raw overall P column in {path}; '
                         f'columns={dset.schema.names[:30]}')
    blocks=[]
    for batch in dset.to_batches(columns=[col],batch_size=500000):
        v=pd.to_numeric(batch.column(0).to_pandas(),errors='coerce').to_numpy(dtype=float)
        v=v[np.isfinite(v) & (v>=0) & (v<=1)]
        if len(v):blocks.append(v)
    if not blocks:
        raise ValueError('No finite raw overall P values in '+str(path))
    return np.concatenate(blocks)


def _bh_raw_cutoff(p, q=0.05):
    p=np.sort(np.asarray(p,float))
    g=len(p)
    ok=np.flatnonzero(p <= q * np.arange(1,g+1)/g)
    if not len(ok):raise ValueError('No BH discoveries at q=0.05 in full CTAR scan')
    r=int(ok[-1]+1)
    return float(p[r-1]), int(g), r


def resolve_thresholds(qtl, explicit_disc=None, explicit_inter=None):
    """Resolve both CTAR thresholds, requiring verifiable primary-scan evidence.

    Priority: explicit user value > finalization metadata > exact full-scan BH.
    A significant-only Parquet is used ONLY if its size matches finalization R;
    this prevents a dynamic/replicated subset being mistaken for full discoveries.
    """
    qtl=Path(qtl)
    cutoff, stagewise, G, R, source=find_threshold_json(qtl.parent)
    sources=[]
    if explicit_disc is not None:
        cutoff=float(explicit_disc);sources.append('user --alpha-disc')
    elif cutoff is not None:
        sources.append(source+':BH raw cutoff')
    if explicit_inter is not None:
        stagewise=float(explicit_inter);sources.append('user --alpha-inter')
    elif stagewise is not None:
        sources.append(source+':alpha_II or 0.05*R/G')
    # For the current Stage-I schema, significant_tqtl.parquet is typically the
    # COMPLETE BH-screened R set. When and only when it has exactly R unique
    # finite overall tests, its maximum raw P is the exact BH discovery cutoff.
    # This avoids sorting >70 million scan P values unnecessarily.
    if cutoff is None and R is not None and qtl.is_file():
        try:
            import pyarrow.parquet as pq
            schema=pq.ParquetFile(qtl).schema_arrow.names
            pcol=acol(pd.DataFrame(columns=schema),_P_COLS,False)
            if pcol is not None:
                vals=pd.read_parquet(qtl,columns=[pcol])[pcol]
                vals=pd.to_numeric(vals,errors='coerce').to_numpy(dtype=float)
                if len(vals)==int(R) and np.all(np.isfinite(vals) & (vals>=0) & (vals<=.05)):
                    cutoff=float(np.max(vals))
                    sources.append(str(qtl)+':verified complete R BH discoveries')
        except (ImportError,ValueError,OSError):
            pass
    if cutoff is None or stagewise is None:
        full_candidates=[qtl.parent/'all_results.parquet',
                         qtl.parent.parent/'all_results.parquet',
                         qtl.parent/'all_results',
                         qtl.parent.parent/'all_results']
        full=next((x for x in full_candidates if x.exists()),None)
        if full is not None:
            bh,Gcalc,Rcalc=_bh_raw_cutoff(_parquet_pvals(full))
            if G is not None and int(G)!=Gcalc:
                raise ValueError(f'Full-scan G differs from finalization: parquet={Gcalc}, JSON={G}. '
                                 'Check whether all_results has been filtered or G includes other tests.')
            if R is not None and int(R)!=Rcalc:
                raise ValueError(f'Full-scan R differs from finalization: parquet={Rcalc}, JSON={R}. '
                                 'Check FDR strategy, scan version and input files.')
            if cutoff is None:
                cutoff=bh;sources.append(str(full)+':recomputed global BH at 0.05')
            if stagewise is None:
                stagewise=0.05*Rcalc/Gcalc;sources.append(str(full)+':0.05R/G')
        elif cutoff is None and R is not None:
            # Only a provably complete set of BH-screened hits permits the boundary.
            try:
                selected=_parquet_pvals(qtl)
                if len(selected)==int(R) and np.all(selected<=0.05):
                    cutoff=float(selected.max());sources.append(str(qtl)+':complete R BH discoveries')
            except (ValueError,ImportError):pass
    missing=[]
    if cutoff is None:missing.append('raw BH discovery P cutoff (alpha-disc)')
    if stagewise is None:missing.append('stagewise interaction alpha_II (alpha-inter)')
    if missing:
        raise ValueError('Cannot reconstruct '+', '.join(missing)+' from the real CTAR scan. '
                         'Check '+str(qtl.parent/'fdr_finalize_summary.json')+
                         ' and full-scan all_results.parquet. '
                         'If necessary, supply NUMERIC --alpha-disc and --alpha-inter '
                         'from actual PPMI finalization (not the FDR q=0.05). '
                         'Never insert estimated thresholds.')
    if not (0<float(cutoff)<=0.05 and 0<float(stagewise)<=0.05):
        raise ValueError(f'Invalid real-data thresholds: alpha_disc={cutoff}, alpha_inter={stagewise}')
    print(f'[CTAR threshold audit] alpha_disc={cutoff:.12g}; alpha_inter={stagewise:.12g}')
    print('[CTAR threshold audit] sources: '+' | '.join(sources))
    return float(cutoff),float(stagewise),' | '.join(sources)


def find_real_threshold(loci,path,explicit=None):
    """Legacy public API retained for callers; prefer resolve_thresholds for both."""
    cutoff,_,source=resolve_thresholds(path,explicit_disc=explicit)
    return cutoff,source


def find_interaction_alpha(path,explicit=None):
    """Legacy public API retained for callers; prefer resolve_thresholds for both."""
    _,stagewise,source=resolve_thresholds(path,explicit_inter=explicit)
    return stagewise,source

def wilson(k,n):
    if not n:return (np.nan,np.nan)
    p=k/n;z=1.95996398454;d=1+z*z/n
    center=(p+z*z/(2*n))/d
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return center-half,center+half

def empirical_replication(loci,reppath):
    if not reppath.exists():return {'note':'PDBP all_results.parquet absent: empirical replication panel is not calculated'}
    import pyarrow.parquet as pq
    schema=pq.ParquetFile(reppath).schema_arrow.names
    schema_df=pd.DataFrame(columns=schema)
    v=acol(schema_df,_VARIANT_COLS)
    g=acol(schema_df,_GENE_COLS)
    p=acol(schema_df,_P_COLS)
    df=pd.read_parquet(reppath,columns=list(dict.fromkeys([v,g,p])))
    z=df[[v,g,p]].rename(columns={v:'variant_id',g:'gene_id',p:'rep_p'}).drop_duplicates(['variant_id','gene_id'])
    hit=loci[['variant_id','gene_id']].merge(z,on=['variant_id','gene_id'],how='left')
    ps=pd.to_numeric(hit.rep_p,errors='coerce')
    tested=int(ps.notna().sum());rep=int((ps<.05).sum())
    if tested==0:return {'note':'No exact variant/gene pairs matched; check gene IDs or variant naming, do not report zero replication'}
    return {'matched':tested,'n_loci':len(hit),'replicated':rep,'rate':rep/tested,'wilson95':wilson(rep,tested),'unmatched':len(hit)-tested}

def primary_grid(args,mafs,groups):
    ns=np.arange(args.min_n,args.max_n+1,args.step_n,dtype=int)
    rows=[];loci_cache={}
    gd=ou_grams(groups['PPMI'],args.rho,args.pattern,args.horizon)
    gr=ou_grams(groups['PDBP'],args.rho,args.pattern,args.horizon)
    for pve in args.pves:
        matrix=[]
        for n in ns:
            pdsc,prep,j=compute_power(n,mafs,pve,gd,gr,args.alpha_disc,args.alpha_repl,retain=args.retain,f2_reference=args.pve_ref_f2,min_minor_hom=args.min_minor_hom)
            ok=(j>=args.locus_power)
            matrix.append(ok)
            rows.append({'pve':pve,'n':n,'fraction_ge80':ok.mean(),
                'median_joint_power':float(np.median(j)),'mean_joint_power':float(j.mean()),
                'fraction_dis_ge80':float(np.mean(pdsc>=args.locus_power)),
                'fraction_rep_ge80':float(np.mean(prep>=args.locus_power))})
        loci_cache[pve]=np.vstack(matrix)
    return pd.DataFrame(rows),ns,loci_cache

def block_bootstrap_n80(mat,ns,genes,nboot,seed,target=.8):
    rng=np.random.default_rng(seed)
    gn,idx=np.unique(np.asarray(genes).astype(str),return_inverse=True)
    if len(gn)<5:return np.nan,np.nan
    pools=[np.flatnonzero(idx==i) for i in range(len(gn))]
    if nboot<=0:return np.nan,np.nan
    sizes=[]
    for _ in range(nboot):
        pick=rng.integers(0,len(pools),size=len(pools))
        ix=np.concatenate([pools[k] for k in pick])
        sizes.append(find_min(ns,mat[:,ix].mean(axis=1),target))
    good=np.array([s for s in sizes if np.isfinite(s)])
    if len(good)<.80*nboot:return np.nan,np.nan
    return tuple(np.percentile(good,[2.5,97.5]))

def compute_secondary(args,mafs,groups,ns):
    records=[]
    for rho in args.rhos:
        for m in args.visits:
            t=planned_templates(m,missing=args.missing,jitter=args.jitter,seed=args.seed,horizon=args.horizon)
            gd=ou_grams(t,rho,args.pattern,args.horizon)
            gr=gd
            vals=[]
            for N in ns:
                _,_,j=compute_power(N,mafs,args.focus_pve,gd,gr,args.alpha_disc,args.alpha_repl,retain=args.retain,f2_reference=args.pve_ref_f2,min_minor_hom=args.min_minor_hom)
                vals.append(np.mean(j>=args.locus_power))
            records.append({'visits':m,'rho':rho,'N80_80':find_min(ns,vals,args.coverage),'mode':'prospective_planned_visits'})
    rows=[]
    gd=ou_grams(groups['PPMI'],args.rho,args.pattern,args.horizon)
    gr=ou_grams(groups['PDBP'],args.rho,args.pattern,args.horizon)
    for pve in args.sensitivity_pves:
        for ret in args.retains:
            vals=[]
            for n in ns:
                _,_,j=compute_power(n,mafs,pve,gd,gr,args.alpha_disc,args.alpha_repl,retain=ret,f2_reference=args.pve_ref_f2,min_minor_hom=args.min_minor_hom)
                vals.append(float(np.mean(j>=args.locus_power)))
            rows.append({'pve':pve,'retention':ret,'N80_80':find_min(ns,vals,args.coverage)})
    return pd.DataFrame(records),pd.DataFrame(rows)

def maf_bins_summary(args,mafs,groups,ns):
    gd=ou_grams(groups['PPMI'],args.rho,args.pattern,args.horizon)
    gr=ou_grams(groups['PDBP'],args.rho,args.pattern,args.horizon)
    bins=[0,.05,.1,.2,.5]
    rows=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        mk=(mafs>lo)&(mafs<=hi)
        if not mk.sum():continue
        vals=[]
        for N in ns:
            _,_,j=compute_power(N,mafs[mk],args.focus_pve,gd,gr,args.alpha_disc,args.alpha_repl,retain=args.retain,f2_reference=args.pve_ref_f2,min_minor_hom=args.min_minor_hom)
            vals.append(float(np.mean(j>=args.locus_power)))
        rows.append({'maf_bin':f'({lo:g}, {hi:g}]','n_loci':int(mk.sum()),'N80_80':find_min(ns,vals,args.coverage), 'maf_median':float(np.median(mafs[mk]))})
    return pd.DataFrame(rows)

def dynamic_secondary(args,mafs,groups,ns):
    gd=ou_grams(groups['PPMI'],args.rho,args.pattern,args.horizon)
    gr=ou_grams(groups['PDBP'],args.rho,args.pattern,args.horizon)
    vals=[]
    for N in ns:
        _,_,j=compute_power(N,mafs,args.focus_pve,gd,gr,args.alpha_disc,args.alpha_repl,alpha_inter=args.alpha_inter,retain=args.retain,test='dynamic',f2_reference=args.pve_ref_f2,min_minor_hom=args.min_minor_hom)
        vals.append({'n':N,'fraction_ge80':float(np.mean(j>=args.locus_power)),'mean_joint_power':float(np.mean(j))})
    return pd.DataFrame(vals)

def plot_all(args,primary,mins,visit_rho,retain,maf_bins,dynamic,empirical,out,cohort_n,is_demo):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.6,'axes.titlesize':11.6,
        'axes.labelsize':9.5,'axes.linewidth':.85,'savefig.facecolor':'white','figure.facecolor':'white'})
    fig,axs=plt.subplots(2,3,figsize=(16.5,9.5),constrained_layout=True)
    ax=axs[0,0]
    pp=sorted(primary.pve.unique())
    palette=[COLORS['gray'],COLORS['gold'],COLORS['red'],COLORS['teal'],COLORS['navy'],COLORS['purple']]
    for i,p in enumerate(pp):
        d=primary[primary.pve==p]
        ax.plot(d.n,d.fraction_ge80,lw=2.1,color=palette[i%len(palette)],label=f'{p*100:g}% PVE')
    ax.axhline(args.coverage,c=COLORS['gray'],ls='--',lw=1.1)
    if cohort_n:
        for k,color in [('PPMI',COLORS['blue']),('PDBP',COLORS['teal'])]:
            ax.axvline(cohort_n[k],lw=.9,ls=':',color=color,alpha=.85)
    ax.set(xlabel='Independent participants per cohort, N',ylabel='Fraction with joint power ≥80%',
           title='a  What fraction of loci can be discovered AND replicated?')
    ax.set(ylim=(0,1.02),xlim=(args.min_n,args.max_n));ax.legend(frameon=False,ncol=2,fontsize=8,loc='lower right')

    ax=axs[0,1]
    p=mins.pve.values*100
    ax.plot(p,mins.N80_80,marker='o',color=COLORS['navy'],lw=2.1,label='Temporal tQTL (8 df)')
    if dynamic is not None:
        dy=mins.loc[np.argmin(np.abs(mins.pve-args.focus_pve)),'N80_80'] if len(mins) else np.nan
        dn=find_min(dynamic.n.to_numpy(),dynamic.fraction_ge80.to_numpy(),args.coverage)
        if np.isfinite(dn):ax.scatter([args.focus_pve*100],[dn],marker='D',s=65,color=COLORS['red'],zorder=5,label='Dynamic tQTL (6 df), focus PVE')
    if cohort_n:
        ax.axhline(min(cohort_n.values()),ls=':',color=COLORS['teal'],lw=1,label='Smaller observed cohort')
    ax.set(xlabel='Assumed genetic PVE at reference MAF 0.20 (%)',ylabel='Minimum N per cohort (N80/80)',
        title='b  Minimum sample size depends on effect size')
    ax.set(ylim=(0,args.max_n*1.06));ax.legend(frameon=False,fontsize=8)
    for x,y in zip(p,mins.N80_80):
        if not np.isfinite(y):ax.annotate('>max',xy=(x,args.max_n),xytext=(0,-12),textcoords='offset points',ha='center',fontsize=7,color=COLORS['red'])

    ax=axs[0,2]
    table=visit_rho.pivot(index='visits',columns='rho',values='N80_80').sort_index()
    data=table.to_numpy(dtype=float);show=np.where(np.isfinite(data),data,args.max_n+args.step_n)
    finite=show[np.isfinite(data)]
    vlow=float(finite.min()) if len(finite) else args.min_n
    vhigh=float(finite.max()) if len(finite) else args.max_n
    im=ax.imshow(show,aspect='auto',cmap='YlGnBu',vmin=vlow,vmax=max(vlow+1.,vhigh))
    ax.set_xticks(np.arange(len(table.columns)),[f'{x:g}' for x in table.columns]);ax.set_yticks(np.arange(len(table.index)),table.index)
    ax.set(xlabel='Within-donor annual correlation (rho)',ylabel='Planned visits per participant',title='c  Visit frequency and residual correlation')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            t=f'{int(data[i,j]):,}' if np.isfinite(data[i,j]) else f'>{args.max_n:,}'
            ax.text(j,i,t,ha='center',va='center',fontsize=8,color='white' if show[i,j]>vlow+.6*(max(vlow+1.,vhigh)-vlow) else 'black')
    fig.colorbar(im,ax=ax,fraction=.046,pad=.03,label='Required N / cohort')

    ax=axs[1,0]
    bins=maf_bins.iloc[::-1]
    for k,(_,row) in enumerate(bins.iterrows()):
        val=row.N80_80
        ax.barh(k,args.max_n if not np.isfinite(val) else val,color=COLORS['blue'] if np.isfinite(val) else COLORS['red'],alpha=.84,height=.62)
        textval=f'{int(val):,}' if np.isfinite(val) else f'>{args.max_n:,}'
        ax.text(min(args.max_n,(val if np.isfinite(val) else args.max_n))+args.max_n*.012,k,textval,va='center',fontsize=8)
    ax.set_yticks(range(len(bins)),[f'{r.maf_bin}  (n={int(r.n_loci)})' for _,r in bins.iterrows()])
    ax.set(xlim=(0,args.max_n*1.16),xlabel='Required N per cohort',title='d  Rare alleles need more independent participants')

    ax=axs[1,1]
    tb=retain.pivot(index='pve',columns='retention',values='N80_80').sort_index()
    v=tb.to_numpy(dtype=float);disp=np.where(np.isfinite(v),v,args.max_n+args.step_n)
    ax.imshow(disp,aspect='auto',cmap='YlGnBu',vmin=args.min_n,vmax=args.max_n)
    ax.set_xticks(range(len(tb.columns)),[f'{x*100:.0f}%' for x in tb.columns]);ax.set_yticks(range(len(tb.index)),[f'{x*100:g}%' for x in tb.index]);
    ax.set(xlabel='Effect retained in replication (amplitude)',ylabel='Discovery effect PVE',title='e  Cross-cohort attenuation raises N80/80')
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j,i,f'{int(v[i,j]):,}' if np.isfinite(v[i,j]) else f'>{args.max_n:,}',ha='center',va='center',fontsize=8,
                    color='white' if disp[i,j]>.6*args.max_n else 'black')

    ax=axs[1,2]
    focus=primary[np.isclose(primary.pve,args.focus_pve)]
    if len(focus):
        ax.plot(focus.n,focus.fraction_ge80,color=COLORS['navy'],lw=2.1,label='Temporal joint power ≥80%')
    if dynamic is not None:
        ax.plot(dynamic.n,dynamic.fraction_ge80,color=COLORS['red'],lw=2.1,label='Dynamic joint power ≥80%')
    ax.axhline(args.coverage,color=COLORS['gray'],ls='--',lw=.9)
    if empirical.get('rate') is not None:
        rate=empirical['rate'];ci=empirical['wilson95']
        text=f"Observed PDBP nominal replication:\n{empirical['replicated']}/{empirical['matched']} = {rate*100:.1f}% (95% CI {ci[0]*100:.1f}–{ci[1]*100:.1f}%)"
        ax.text(.98,.03,text,transform=ax.transAxes,ha='right',va='bottom',fontsize=8,
                bbox={'facecolor':'white','edgecolor':'#D8DFE5','pad':4})
    ax.set(xlabel='Independent participants per cohort, N',ylabel='Fraction of loci with joint power ≥80%',
           title='f  Overall tQTL versus dynamic tQTL design',ylim=(0,1.02),xlim=(args.min_n,args.max_n))
    ax.legend(frameon=False,loc='upper left',fontsize=8)
    for ax in axs.ravel():
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('CTAR-MPTGA: prospective discovery + replication sample-size design',fontsize=15,fontweight='bold')
    extra='DEMONSTRATION ONLY — synthetic loci/visits; NOT PPMI/PDBP results' if is_demo else 'Real PPMI/PDBP locus MAF + visit templates; prospective assumed effects (NOT observed fitted effects)'
    fig.supxlabel(f'{extra}    |    N80/80: ≥{args.coverage:.0%} of loci each with ≥{args.locus_power:.0%} joint power',fontsize=8.6,color=COLORS['gray'])
    for suff in ['png','pdf','svg']:
        fig.savefig(out/f'Supplementary_Figure_6_CTAR_N80_80.{suff}',dpi=300,bbox_inches='tight', backend=(suff if suff in ("svg", "pdf") else None))
    plt.close(fig)

def run(args):
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    if args.demo:
        groups=ordered_templates(seed=args.seed)
        variants=np.arange(1800)
        mafs=np.minimum(rng.beta(1.8,3.8,size=len(variants)),.5)
        loci=pd.DataFrame({'variant_id':[f'DEMO_{i}' for i in variants],'gene_id':[f'G{i//3}' for i in variants],'maf':mafs})
        loci=loci.sample(args.n_pairs,random_state=args.seed).reset_index(drop=True)
        cohort_n={'PPMI':800,'PDBP':800}
        empirical={'note':'DEMO: no real replication results'}
        args.alpha_disc=args.alpha_disc if args.alpha_disc is not None else 1e-4
        args.alpha_inter=args.alpha_inter if args.alpha_inter is not None else 5e-4
        threshold_source='DEMO thresholds'
    else:
        root=Path(args.root)
        meta=Path(args.metadata) if args.metadata else root/'data/metadata/amp_pd_samples.parquet'
        qtl=Path(args.loci_file) if args.loci_file else root/'qtl/PP/submitted_followup_ctar_mptga/final/significant_tqtl.parquet'
        rep=Path(args.replication_file) if args.replication_file else root/'qtl/PD/submitted_followup_ctar_mptga/final/all_results.parquet'
        if not meta.exists():raise FileNotFoundError('Metadata missing: '+str(meta))
        if not qtl.exists():raise FileNotFoundError('PPMI QTL file missing: '+str(qtl))
        groups=meta_templates(meta,args.col_cohort,args.col_participant,args.col_time)
        cohort_n={k:len(v) for k,v in groups.items()}
        plink=args.plink_prefix or str(root/'data/genotype/PP/amp_pd_PP_qc')
        args.alpha_disc,args.alpha_inter,threshold_source=resolve_thresholds(
            qtl,args.alpha_disc,args.alpha_inter)
        loci=load_input_loci(qtl,args.seed,args.n_pairs,args.min_pairs,plink)
        empirical=empirical_replication(loci,rep)
    if len(loci)<args.min_pairs:raise ValueError(f'Only {len(loci)} loci: min={args.min_pairs}')
    mafs=loci.maf.to_numpy(dtype=float)
    # Fix effect magnitude across cohort/time-schedule scenarios: 5 planned annual visits
    # define the PVE standardization reference, never each scenario independently.
    reference_t=np.linspace(0,args.horizon,5)
    args.pve_ref_f2=float(np.mean(mode_values(reference_t,args.pattern,args.horizon)**2))
    if not (0<args.alpha_disc<1 and 0<args.alpha_inter<1):raise ValueError('Bad alpha thresholds')
    loci.to_csv(out/'loci_1500_real_or_demo.csv',index=False)
    primary,ns,matrices=primary_grid(args,mafs,groups)
    mins=[]
    for pve in args.pves:
        d=primary[np.isclose(primary.pve,pve)]
        mat=matrices[pve]
        ci=block_bootstrap_n80(mat,ns,loci.gene_id,args.bootstraps,args.seed,args.coverage)
        mins.append({'pve':pve,'N80_80':find_min(d.n,d.fraction_ge80,args.coverage),
                     'discovery_only_N80_80':find_min(d.n,d.fraction_dis_ge80,args.coverage),
                     'replication_only_N80_80':find_min(d.n,d.fraction_rep_ge80,args.coverage),
                     'bootstrap_ci_lower':ci[0],'bootstrap_ci_upper':ci[1],
                     'n_loci':len(mafs),'n_genes':loci.gene_id.nunique()})
    mins=pd.DataFrame(mins)
    visit_rho,retain=compute_secondary(args,mafs,groups,ns)
    observed=[]
    for pve in args.pves:
        pdsc,prep,j=compute_power((cohort_n['PPMI'],cohort_n['PDBP']),mafs,pve,
            ou_grams(groups['PPMI'],args.rho,args.pattern,args.horizon),
            ou_grams(groups['PDBP'],args.rho,args.pattern,args.horizon),
            args.alpha_disc,args.alpha_repl,retain=args.retain,f2_reference=args.pve_ref_f2,min_minor_hom=args.min_minor_hom)
        observed.append({'pve':pve,'PPMI_N':cohort_n['PPMI'],'PDBP_N':cohort_n['PDBP'],
            'fraction_loci_joint_power_ge80':float(np.mean(j>=args.locus_power)),
            'median_joint_power':float(np.median(j))})
    pd.DataFrame(observed).to_csv(out/'modeled_power_at_actual_cohort_sizes.csv',index=False)
    maf_bins=maf_bins_summary(args,mafs,groups,ns)
    pattern_rows=[]
    orig_pattern,orig_f2=args.pattern,args.pve_ref_f2
    for pat in ['linear','early','late']:
        args.pattern=pat
        args.pve_ref_f2=float(np.mean(mode_values(np.linspace(0,args.horizon,5),pat,args.horizon)**2))
        gd=ou_grams(groups['PPMI'],args.rho,pat,args.horizon)
        gr=ou_grams(groups['PDBP'],args.rho,pat,args.horizon)
        cover=[]
        for n in ns:
            _,_,j=compute_power(n,mafs,args.focus_pve,gd,gr,args.alpha_disc,args.alpha_repl,
                retain=args.retain,f2_reference=args.pve_ref_f2,min_minor_hom=args.min_minor_hom)
            cover.append(np.mean(j>=args.locus_power))
        pattern_rows.append({'pattern':pat,'pve':args.focus_pve,'N80_80':find_min(ns,cover,args.coverage)})
    args.pattern,args.pve_ref_f2=orig_pattern,orig_f2
    pd.DataFrame(pattern_rows).to_csv(out/'N_by_trajectory_shape.csv',index=False)
    dynamic=dynamic_secondary(args,mafs,groups,ns)
    for df,name in [(primary,'power_grid'),(mins,'min_N80_80_by_PVE'),(visit_rho,'N_by_visits_rho'),(retain,'N_by_crosscohort_retention'),(maf_bins,'N_by_MAF_bin'),(dynamic,'dynamic_power_focus_PVE')]:
        df.to_csv(out/(name+'.csv'),index=False)
    config={k:v for k,v in vars(args).items() if isinstance(v,(int,float,str,bool,type(None),list))}
    config.update({'loci_used':len(loci),'genes_used':int(loci.gene_id.nunique()),'cohort_n':cohort_n,
                   'alpha_disc_source':threshold_source,'empirical_replication':empirical,
                   'locus_sampling_audit':loci.attrs.get('loci_audit',{}),
                   'method':'OU innovation precision, fixed-rho nc-chi-square approximation to full 8df CTAR and nested 6df interaction',
                   'caveat':'Not production 100-point-rho GPU profile LRT; final numeric N conditional on all inputs; real-data validation needed',
                   'design_target':{'target_joint_power_per_locus':args.locus_power,'fraction_of_loci':args.coverage}})
    (out/'analysis_manifest.json').write_text(json.dumps(config,ensure_ascii=False,indent=2,default=lambda x:float(x)),encoding='utf-8')
    plot_all(args,primary,mins,visit_rho,retain,maf_bins,dynamic,empirical,out,cohort_n,args.demo)
    print('\nCTAR sample-size analysis complete. Files:',out)
    print('Observed cohort Ns:',cohort_n,'\nEmpirical replication:',empirical)
    print('Discovery raw P:',args.alpha_disc,'; stage-wise interaction alpha:',args.alpha_inter)
    print(mins.to_string(index=False))
    print('IMPORTANT: These are scenario-specific analytic estimates, not exact CTAR profile-LRT guarantees.')

def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description=__doc__)
    ap.add_argument('--root',default='/public2/data/luojunfeng/2sGenGPS_revised_chatgpt')
    ap.add_argument('--output',default=None)
    ap.add_argument('--loci-file',default=None,help='Original PPMI significant QTL parquet or optional TSV/CSV containing variant_id,gene_id,MAF')
    ap.add_argument('--metadata',default=None)
    ap.add_argument('--col-cohort',default=None,help='Override cohort field in sample metadata')
    ap.add_argument('--col-participant',default=None,help='Override donor/participant field in sample metadata')
    ap.add_argument('--col-time',default=None,help='Override follow-up time field (must be in years or named months/days)')
    ap.add_argument('--replication-file',default=None)
    ap.add_argument('--plink-prefix',default=None)
    ap.add_argument('--alpha-disc',type=float,default=None,help='Raw genome-wide PPMI BH threshold, NOT q-value 0.05')
    ap.add_argument('--alpha-inter',type=float,default=None,help='stagewise alpha_II = 0.05R/G, NOT 0.05')
    ap.add_argument('--alpha-repl',type=float,default=.05)
    ap.add_argument('--n-pairs',type=int,default=1500)
    ap.add_argument('--min-pairs',type=int,default=1000)
    ap.add_argument('--seed',type=int,default=20261030)
    ap.add_argument('--min-n',type=int,default=100)
    ap.add_argument('--max-n',type=int,default=8000)
    ap.add_argument('--step-n',type=int,default=100)
    ap.add_argument('--pves',nargs='+',type=float,default=[.0025,.005,.01,.02,.04])
    ap.add_argument('--focus-pve',type=float,default=.01)
    ap.add_argument('--sensitivity-pves',nargs='+',type=float,default=[.005,.01,.02])
    ap.add_argument('--rho',type=float,default=.5)
    ap.add_argument('--rhos',nargs='+',type=float,default=[0.,.3,.6,.9])
    ap.add_argument('--visits',nargs='+',type=int,default=[2,3,4,5])
    ap.add_argument('--horizon',type=float,default=4.)
    ap.add_argument('--pattern',choices=['linear','late','early','static'],default='linear')
    ap.add_argument('--missing',type=float,default=.15)
    ap.add_argument('--jitter',type=float,default=.08)
    ap.add_argument('--retain',type=float,default=1.)
    ap.add_argument('--retains',nargs='+',type=float,default=[.5,.75,1.])
    ap.add_argument('--min-minor-hom',type=float,default=5.,help='Expected minor homozygotes required per cohort for full-rank two-channel coding; HWE approximation')
    ap.add_argument('--locus-power',type=float,default=.8)
    ap.add_argument('--coverage',type=float,default=.8)
    ap.add_argument('--bootstraps',type=int,default=200)
    ap.add_argument('--demo',action='store_true',help='Pure synthetic smoke test; NEVER report demo numbers as real results')
    args=ap.parse_args()
    if args.output is None:args.output=str(Path(args.root)/'manuscript_figures/tqtl_benchmark_v4/sfigure6_ctar_N80_final')
    if not 0<=args.rho<1:ap.error('--rho must be in [0,1)')
    if args.focus_pve not in args.pves:ap.error('--focus-pve must be present in --pves')
    if args.min_n>args.max_n:ap.error('--max-n must >= --min-n')
    run(args)
if __name__=='__main__':main()
