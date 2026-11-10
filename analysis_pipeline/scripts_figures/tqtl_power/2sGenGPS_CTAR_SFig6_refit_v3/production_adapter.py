"""Exact Stage-I sample/covariate/expression interfaces for SFig6.

Single source of truth: gengps.amp_pd.real_qtl.  Never guess nuisance columns.
The expression Parquet has HGNC columns while QTL tables have Ensembl IDs.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path: sys.path.insert(0,str(REPO))
from gengps.amp_pd.config import AmpPdPaths
from gengps.amp_pd.real_qtl import (
    load_cohort_metadata, _augment_fixed_genotype_pcs,
    _prepare_irregular_followup_rows, _drop_missing_covariates,
    method_covariates, resolve_expression_gene_columns,
)
METHOD='submitted_followup_ctar_mptga'

class ProductionMetadata:
    def __init__(self,root,cohort):
        root=Path(root)
        paths=AmpPdPaths(root=str(root),plink19='/usr/bin/plink1.9')
        m=load_cohort_metadata(paths,cohort)
        m=_augment_fixed_genotype_pcs(paths,cohort,m,n_pcs=8)
        # Stage-I resolves duplicate participant/time BEFORE filtering covariates
        # (gene expression input index mirrors meta at this stage).
        dummy=pd.DataFrame(index=m.index)
        m,dummy,ndup=_prepare_irregular_followup_rows(m,dummy)
        cov=method_covariates(METHOD,cohort,m,cell_adjust=True)
        dummy=dummy.reindex(m.index)
        m,_=_drop_missing_covariates(m,dummy,cov)
        if not len(cov):raise ValueError('No Stage-I covariates')
        if any(x not in m for x in ('sample_id','participant_id','followup_time')):
            raise ValueError('Production metadata missing required alignment keys')
        if len(m)<100:raise ValueError(f'{cohort}: no usable Stage-I observations')
        self.data=m
        self.cov=cov
        # Cache invariant Stage-I observation/covariate arrays once per worker;
        # do not allocate a full metadata DataFrame for each of 5,000 loci.
        self._samples=m['sample_id'].astype(str).to_numpy()
        self._donors=m['participant_id'].astype(str).to_numpy()
        self._time=m['followup_time'].to_numpy(float)
        self._cov_numeric=m[cov].apply(pd.to_numeric,errors='raise').to_numpy(np.float64)
        self._first_donor_indices=np.flatnonzero(~pd.Index(self._donors).duplicated())
        if not np.isfinite(self._cov_numeric).all():
            raise ValueError('Nonfinite Stage-I nuisance columns')
        self.cohort=cohort
        self.paths=paths
        self.summary={'n_obs':len(m),'n_donors':m.participant_id.nunique(),
                      'n_covariates':len(cov),'covariates':cov,
                      'n_duplicate_visits_removed':ndup,
                      'expression_ids':'Ensembl resolved by real_qtl.resolve_expression_gene_columns'}

class ProductionExpression:
    def __init__(self,root,selected_genes,cache_file=None):
        self.paths=AmpPdPaths(root=str(root),plink19='/usr/bin/plink1.9')
        self.gene_map=resolve_expression_gene_columns(self.paths,list(map(str,selected_genes)),allow_unresolved=True)
        self.unknown=set(map(str,selected_genes))-set(self.gene_map)
        self.path=Path(cache_file) if cache_file else self.paths.data_dir/'expression'/'amp_pd_expression_z.parquet'
        self.columns=['sample_id']+list(dict.fromkeys(self.gene_map.values()))
        import pyarrow.parquet as pq
        available=set(pq.ParquetFile(self.path).schema.names)
        missing=[c for c in self.columns if c not in available]
        if missing:raise KeyError(f'Expression columns absent in {self.path}: {missing[:20]}')
        self.data=None
        self.memo={}
        self.mode='wide_symbol_mapped'  # Provenance in 02_run_power; Stage-I Ensembl -> HGNC resolver
        self.source_mode=self.mode
        # For a small preflight set, one multi-column read is much faster.
        # For full power analysis, only the selected gene columns are loaded once.
        x=pd.read_parquet(self.path,columns=self.columns)
        if x.sample_id.astype(str).duplicated().any():raise ValueError('Duplicate expression sample_id')
        self.data=x.set_index(x.sample_id.astype(str)).drop(columns='sample_id')
    def get_gene(self,g):
        gene=str(g)
        if gene not in self.gene_map:raise KeyError(f'{gene}: no unambiguous Ensembl-to-HGNC expression mapping')
        return pd.to_numeric(self.data[self.gene_map[gene]],errors='coerce')

class ProductionGenotypes:
    """PLINK1.9 recode A; match FID/IID against production Stage-I participant IDs.

    Note: alleles may be reversed relative to Stage-I BED order. Full 8df LRT is
    invariant to 0/2 label swap when all 3 classes present; beta coefficients aren't.
    """
    def __init__(self,path,participant_ids):
        d=pd.read_csv(path,sep=r'\s+',low_memory=False)
        if 'FID' not in d or 'IID' not in d:raise ValueError(f'{path}: missing FID or IID')
        wanted=set(map(str,participant_ids))
        counts={k:int(d[k].astype(str).isin(wanted).sum()) for k in ('FID','IID')}
        which=max(counts,key=counts.get)
        if counts[which]==0:raise ValueError(f'{path}: no participant IDs matched {counts}; Stage I uses participant_id')
        d.index=d[which].astype(str)
        if d.index.duplicated().any():raise ValueError(f'{path}: duplicate {which} IDs')
        self.data=d
        self.which=which
        self.coverage=counts
        self.columns=[str(c) for c in d.columns if c not in ('FID','IID','PAT','MAT','SEX','PHENOTYPE')]
    def vector(self,variant,participants):
        exact=[c for c in self.columns if c==str(variant) or c.startswith(str(variant)+'_')]
        if len(exact)!=1:raise KeyError(f'{variant}: expected exactly one PLINK .raw dosage column, found {len(exact)}')
        v=pd.to_numeric(self.data[exact[0]],errors='coerce')
        ids=pd.Index(pd.Series(participants).astype(str))
        x=v.reindex(ids).to_numpy(float)
        # Stage-I uses -1 for missing hardcalls, NOT dropped, NOT mean-imputed.
        x=np.where(np.isfinite(x),x,-1.)
        bad=~np.isin(x,[-1.,0.,1.,2.])
        if bad.any():raise ValueError(f'{variant}: non-hardcall genotypes present: {np.unique(x[bad])[:8]}')
        return x,int(ids.isin(self.data.index).sum())


def build_production_locus(meta,geno,expression,gene,variant,min_donors=80,min_minor_group=2):
    # All sample/covariate arrays were prebuilt by ProductionMetadata.
    # The gene-specific work is now only one indexed expression lookup and
    # one already-loaded BED genotype lookup, never a 51-column frame copy.
    donors=getattr(meta,'_donors',meta.data.participant_id.astype(str).to_numpy())
    samples=getattr(meta,'_samples',meta.data.sample_id.astype(str).to_numpy())
    times=getattr(meta,'_time',meta.data.followup_time.to_numpy(float))
    y=expression.get_gene(gene).reindex(samples).to_numpy(float)
    g,match=geno.vector(variant,donors)
    if match<min_donors:
        raise ValueError(f'Genotype participants matched only {match}/{len(donors)} observation rows')
    if not np.isfinite(y).all():
        raise ValueError(f'Expression {gene}: {int((~np.isfinite(y)).sum())} nonfinite among production Stage-I observations')
    first=getattr(meta,'_first_donor_indices',np.flatnonzero(~pd.Index(donors).duplicated()))
    n=len(first)
    if n<min_donors:raise ValueError(f'Only {n} donors')
    gb=g[first]
    cnt=[int(np.sum(gb==k)) for k in (0,1,2,-1)]
    # Require both 4-df genetic channels to be estimable under ORIGINAL binary
    # coding; three nonempty conventional 0/1/2 classes are not necessary if
    # a missing-call (-1 -> 11) category is explicitly present in Stage-I.
    channel=np.column_stack([np.ones(len(gb)),np.isin(gb,[2.,-1.]).astype(float),
                             np.isin(gb,[1.,-1.]).astype(float)])
    if np.linalg.matrix_rank(channel)<3:
        raise ValueError(f'Genotype-channel design rank<3: genotype counts 0/1/2/missing={cnt}')
    if sum(v>=min_minor_group for v in cnt)<3:
        raise ValueError(f'Fewer than 3 adequately represented genotype classes: counts={cnt}')
    cov=getattr(meta,'_cov_numeric',None)
    if cov is None:cov=meta.data[meta.cov].apply(pd.to_numeric,errors='raise').to_numpy(float)
    if not np.isfinite(cov).all():raise ValueError('Nonfinite production covariates')
    return dict(time=times, donor=donors, genotype=g, y=y, cov=cov,
                n_donors=n,n_obs=len(donors),g_counts=cnt,
                maf=min(float(np.mean(gb[gb>=0])/2),1-float(np.mean(gb[gb>=0])/2)),
                n_covariates=cov.shape[1], covariates=meta.cov)

class ProductionBedGenotypes:
    """Preferred genotype source: EXACT Stage-I BED reader and participant/SNP ordering.

    Reads all selected SNPs once per cohort. No PLINK export, allele flips, or
    alteration of missing hard-call encoding. Falls back to Stage-I's PLINK1.9
    export reader if bed_reader is unavailable.
    """
    def __init__(self,root,cohort,participant_ids,variants):
        from gengps.amp_pd.real_qtl import bfile_for_cohort
        from gengps.amp_pd.genotype import make_genotype_source
        from gengps.amp_pd.config import AmpPdPaths
        paths=AmpPdPaths(root=str(root),plink19='/usr/bin/plink1.9')
        bfile=bfile_for_cohort(paths,cohort)
        gsrc=make_genotype_source(str(bfile),'/usr/bin/plink1.9',prefer_bed_reader=True)
        pids=list(pd.unique(pd.Series(participant_ids).astype(str)))
        requested=list(dict.fromkeys(map(str,variants)))
        bim=Path(str(bfile)+'.bim')
        present=set()
        with bim.open() as f:
            for line in f:
                z=line.split()
                if len(z)>1 and z[1] in requested:present.add(z[1])
        available=[v for v in requested if v in present]
        if not available:raise ValueError(f'{cohort} BED: selected variants do not overlap BIM')
        table=gsrc.read(pids,available,mean_impute=False,dtype=np.float64)
        table.index=table.index.astype(str)
        self.data=table.fillna(-1.)
        self.missing_variants=set(requested)-set(available)
        self.coverage={'participants':len(pids),'variants_requested':len(requested),
                       'variants_available':len(available),'missing_variants':len(self.missing_variants),
                       'source':'Stage-I BED reader / PLINK 1.9 fallback','prefix':str(bfile)}
    def vector(self,variant,participants):
        if str(variant) not in self.data:raise KeyError(f'{variant}: not in Stage-I BED source')
        ids=pd.Index(pd.Series(participants).astype(str))
        x=self.data[str(variant)].reindex(ids).to_numpy(float)
        if not np.isfinite(x).all():raise ValueError('Stage-I missing participant genotype rows')
        return x,len(ids)
