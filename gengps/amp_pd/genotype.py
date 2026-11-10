from __future__ import annotations
from pathlib import Path
import tempfile, subprocess
import numpy as np
import pandas as pd

class BedGenotypeSource:
    """Lazy random-access PLINK BED reader using bed-reader."""
    def __init__(self,bfile_prefix: str):
        from bed_reader import open_bed
        self.prefix=str(bfile_prefix)
        self.bed=open_bed(self.prefix+".bed")
        self.fam=pd.read_csv(self.prefix+".fam",sep=r"\s+",header=None,
                             names=["FID","IID","PAT","MAT","SEX","PHENO"],dtype=str)
        self.bim=pd.read_csv(self.prefix+".bim",sep=r"\s+",header=None,
                             names=["chr","snpid","cm","pos","a1","a2"],dtype={"chr":str,"snpid":str})
        self.snp_to_index=pd.Series(np.arange(len(self.bim),dtype=np.int64),index=self.bim.snpid.astype(str)).to_dict()

    def resolve_sample_ids(self,participant_ids):
        ids=set(map(str,participant_ids))
        nf=self.fam.FID.astype(str).isin(ids).sum();ni=self.fam.IID.astype(str).isin(ids).sum()
        col="FID" if nf>=ni else "IID"
        vals=self.fam[col].astype(str);idx={v:i for i,v in enumerate(vals)}
        missing=[x for x in map(str,participant_ids) if x not in idx]
        if missing:raise KeyError(f"{len(missing)} requested participants absent from BED {col}; first={missing[:5]}")
        return np.array([idx[str(x)] for x in participant_ids],dtype=np.int64)

    def read(self,participant_ids,snpids,mean_impute=True,dtype=np.float64):
        snpids=[str(s) for s in snpids]
        missing=[s for s in snpids if s not in self.snp_to_index]
        if missing:raise KeyError(f"{len(missing)} SNPs absent from BIM; first={missing[:5]}")
        si=self.resolve_sample_ids(participant_ids);vi=np.array([self.snp_to_index[s] for s in snpids],dtype=np.int64)
        X=np.asarray(self.bed.read(index=np.s_[si,vi],dtype=dtype),dtype=dtype)
        if mean_impute and np.isnan(X).any():
            mu=np.nanmean(X,axis=0);mu[~np.isfinite(mu)]=0.0
            rr,cc=np.where(np.isnan(X));X[rr,cc]=mu[cc]
        return pd.DataFrame(X,index=pd.Index(list(map(str,participant_ids)),name="participant_id"),columns=snpids)

class PlinkExportGenotypeSource:
    """Portable fallback when bed-reader is unavailable.

    It invokes PLINK 1.9 for each gene chunk. This is slower than BED random
    access but keeps the pipeline functional without additional Python binary
    dependencies.
    """
    def __init__(self,bfile_prefix: str,plink19="plink1.9"):
        self.prefix=str(bfile_prefix);self.plink19=plink19
        self.fam=pd.read_csv(self.prefix+".fam",sep=r"\s+",header=None,
                             names=["FID","IID","PAT","MAT","SEX","PHENO"],dtype=str)
    def read(self,participant_ids,snpids,mean_impute=True,dtype=np.float64):
        ids=set(map(str,participant_ids))
        nf=self.fam.FID.astype(str).isin(ids).sum();ni=self.fam.IID.astype(str).isin(ids).sum();col="FID" if nf>=ni else "IID"
        keepfam=self.fam[self.fam[col].astype(str).isin(ids)]
        with tempfile.TemporaryDirectory(prefix="gengps_plink_") as td:
            td=Path(td);keep=td/"keep.txt";extract=td/"extract.txt";out=td/"geno"
            keepfam[["FID","IID"]].to_csv(keep,sep='\t',header=False,index=False)
            pd.Series(list(map(str,snpids))).to_csv(extract,index=False,header=False)
            subprocess.run([self.plink19,"--bfile",self.prefix,"--keep",str(keep),"--extract",str(extract),
                            "--recode","A","--out",str(out)],check=True,stdout=subprocess.DEVNULL)
            raw=pd.read_csv(str(out)+".raw",sep=r"\s+")
            # PLINK --recode A columns are usually SNP_ALLELE. Map back by prefix.
            idcol="FID" if col=="FID" else "IID"
            raw[idcol]=raw[idcol].astype(str);raw=raw.set_index(idcol)
            result={}
            for s in map(str,snpids):
                cand=[c for c in raw.columns if c==s or c.startswith(s+"_")]
                if not cand:continue
                result[s]=pd.to_numeric(raw[cand[0]],errors='coerce')
            X=pd.DataFrame(result).reindex(list(map(str,participant_ids)))
            if mean_impute:
                X=X.apply(lambda x:x.fillna(x.mean() if np.isfinite(x.mean()) else 0.0))
            return X.astype(dtype)

def make_genotype_source(bfile_prefix: str,plink19="plink1.9",prefer_bed_reader=True):
    if prefer_bed_reader:
        try:
            import bed_reader  # noqa
            return BedGenotypeSource(bfile_prefix)
        except ImportError:
            pass
    return PlinkExportGenotypeSource(bfile_prefix,plink19)
