#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,traceback,subprocess,tempfile
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gengps.amp_pd.genotype import make_genotype_source, PlinkExportGenotypeSource
from gengps.amp_pd.grex import legacy_single_iv_from_qtl_row
class _A1PreservingPlinkExportGenotypeSource(PlinkExportGenotypeSource):
    """AD-only fallback for absent bed-reader, with explicit frozen A1 verification.

    Generic genotype.py's PLINK-export fallback does not set keep-allele-order;
    never silently allow reorientation of a frozen Stage1 IV dosage.
    """
    def __init__(self,bfile_prefix,plink19):
        super().__init__(bfile_prefix,plink19)
        bim=pd.read_csv(str(bfile_prefix)+'.bim',sep=r'\s+',header=None,
                        names=['chr','snp','cm','bp','a1','a2'],dtype=str)
        if bim.snp.duplicated().any():
            raise RuntimeError('Duplicate frozen SNPs in AD Stage2 genotype BIM')
        self._a1=dict(zip(bim.snp,bim.a1))

    def read(self,participant_ids,snpids,mean_impute=True,dtype=np.float64):
        pids=list(map(str,participant_ids));snps=list(map(str,snpids));pidset=set(pids)
        nf=self.fam.FID.astype(str).isin(pidset).sum()
        ni=self.fam.IID.astype(str).isin(pidset).sum()
        idcol='FID' if nf>=ni else 'IID'
        selected=self.fam.loc[self.fam[idcol].astype(str).isin(pidset)]
        if len(selected)!=len(set(pids)):
            raise KeyError(f'Missing genotype participants in PLINK fallback: expected={len(set(pids))}, got={len(selected)}')
        with tempfile.TemporaryDirectory(prefix='ad_mmse_A1_plink_') as dirname:
            td=Path(dirname);keep=td/'keep.txt';extract=td/'extract.txt';out=td/'geno'
            selected[['FID','IID']].to_csv(keep,sep='\t',header=False,index=False)
            pd.Series(snps).to_csv(extract,header=False,index=False)
            cmd=[self.plink19,'--bfile',self.prefix,'--keep',str(keep),
                 '--extract',str(extract),'--keep-allele-order',
                 '--recode','A','--out',str(out)]
            proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            if proc.returncode:
                raise RuntimeError('Allele-safe PLINK export failed: '+str(proc.stdout)[-2000:])
            raw=pd.read_csv(str(out)+'.raw',sep=r'\s+',dtype={'FID':str,'IID':str})
            raw[idcol]=raw[idcol].astype(str)
            raw=raw.set_index(idcol)
            data={}
            for snp in snps:
                expected=self._a1.get(snp)
                if expected is None:
                    raise KeyError(f'{snp}: not found in frozen genotype BIM')
                wanted=f'{snp}_{expected}'
                if wanted not in raw.columns:
                    matching=[c for c in raw.columns if c==snp or c.startswith(snp+'_')]
                    raise RuntimeError(f'Frozen Stage1 allele direction uncertain for {snp}: '
                                       f'expected PLINK additive column {wanted}, got {matching}. '
                                       'Refuse to silently flip GReX dosage.')
                data[snp]=pd.to_numeric(raw[wanted],errors='coerce')
            ans=pd.DataFrame(data).reindex(pids)
            if mean_impute:
                for s in snps:
                    mu=ans[s].mean(skipna=True)
                    ans[s]=ans[s].fillna(mu if np.isfinite(mu) else 0.0)
            return ans.astype(dtype)

_WORKER={}

def clean_cov(path):
    x=pd.read_csv(path,low_memory=False);x=x.drop(columns=[c for c in x.columns if str(c).startswith("Unnamed:")],errors="ignore")
    for c in ["cohort","participant_id","time"]:
        if c not in x.columns:raise ValueError(f"Missing required phenotype column {c}")
    x["cohort"]=x.cohort.astype(str);x["participant_id"]=x.participant_id.astype(str);x["time"]=pd.to_numeric(x.time,errors="coerce");x=x.dropna(subset=["participant_id","time"]).sort_values(["cohort","participant_id","time"]).drop_duplicates(["cohort","participant_id","time"],keep="first").reset_index(drop=True);x["single_iv_row_id"]=np.arange(len(x));return x

def followup_support_mask(rows,tmin,tmax,tol=1e-10):
    t=pd.to_numeric(rows["time"],errors="coerce").to_numpy(float);return np.isfinite(t)&(t>=float(tmin)-float(tol))&(t<=float(tmax)+float(tol))

def read_genotype_frame(gsrc,rows,snp):
    pids=pd.Index(pd.unique(rows.participant_id.astype(str)),dtype=str);G=np.asarray(gsrc.read(pids.tolist(),[str(snp)],mean_impute=True,dtype=np.float64),float)
    if G.shape!=(len(pids),1):raise ValueError(f"genotype shape {G.shape}, expected {(len(pids),1)}")
    return pd.Series(G[:,0],index=pids,name="iv_dosage")

def meta_for_model(rows):
    m=rows.copy();m["participant_id"]=m.participant_id.astype(str);m["followup_time"]=pd.to_numeric(m.time,errors="coerce")
    if "age" in m.columns:m["current_age"]=pd.to_numeric(m.age,errors="coerce")+m.followup_time
    else:m["current_age"]=m.followup_time
    if "sample_id" not in m.columns:m["sample_id"]=m.participant_id+"__singleIV__"+m.followup_time.astype(str)
    return m

def direct_component(model,gsrc,rows,snp):
    m=meta_for_model(rows);dose=read_genotype_frame(gsrc,m,snp);donorG=pd.DataFrame({str(snp):dose.values},index=dose.index.astype(str));pred=np.asarray(model.predict_grex(donorG,m),float)
    if pred.shape!=(len(m),):
        G=pd.DataFrame({str(snp):m.participant_id.map(dose).to_numpy(float)},index=m.index);pred=np.asarray(model.predict_grex(G,m),float)
    if pred.shape!=(len(m),):raise ValueError(f"component shape {pred.shape}, expected {(len(m),)}")
    return pred,m.participant_id.map(dose).to_numpy(float)

def _support_columns(z,tmin,tmax):
    # AD MMSE protocol: evaluate every eligible observed follow-up time.
    # Stage1's verified training-time support is not available as a fixed bound;
    # DO NOT mark all MMSE rows as if they were inside Stage1 training support.
    return z, np.ones(len(z),dtype=bool)

def _summary(aid,gene,snp,z,support,status="ok"):
    t=pd.to_numeric(z.time,errors="coerce") if len(z) else pd.Series(dtype=float)
    return {"artifact_id":aid,"gene":gene,"lead_snp":snp,"status":status,
            "n_rows":int(len(z)),"n_participants":int(z.participant_id.nunique()) if "participant_id" in z.columns else 0,
            "phenotype_followup_min":float(t.min()) if len(t) else np.nan,
            "phenotype_followup_max":float(t.max()) if len(t) else np.nan,
            "stage1_followup_min":None,"stage1_followup_max":None,
            "n_rows_for_prediction":int(len(z)),
            "stage1_time_support_assessed":False}

def _init(cfg):
    global _WORKER
    _WORKER=dict(cfg);_WORKER["cov"]=pd.read_parquet(cfg["cov_cache"])
    if not cfg.get("rewrite_support_only",False):
        gsrc=make_genotype_source(cfg["bfile"],cfg["plink"],prefer_bed_reader=True)
        if isinstance(gsrc,PlinkExportGenotypeSource):
            # Guarantee A1 counting even if Python bed-reader is unavailable.
            gsrc=_A1PreservingPlinkExportGenotypeSource(cfg["bfile"],cfg["plink"])
        _WORKER["gsrc"]=gsrc

def _one(row):
    try:
        gene=str(row["gene"]);snp=str(row["lead_snp"]);aid=str(row["artifact_id"]);pf=Path(_WORKER["out"])/"single_iv_genetic_component"/f"{aid}.parquet"
        if _WORKER.get("rewrite_support_only",False):
            if not pf.exists():raise FileNotFoundError(f"Existing component required for --rewrite-support-only: {pf}")
            z=pd.read_parquet(pf);z,support=_support_columns(z,_WORKER["stage1_followup_min"],_WORKER["stage1_followup_max"]);z["support_definition"]="not_assessed_all_observed_MMSE_followups";z.to_parquet(pf,index=False,compression="zstd");return _summary(aid,gene,snp,z,support,"support_rewritten"),None
        payload=dict(row);payload["Variant_ID"]=snp;payload["Ensembl_ID"]=gene;payload["Pvalue"]=float(pd.to_numeric(pd.Series([row.get("ppmi_tqtl_p",np.nan)]),errors="coerce").iloc[0]);model=legacy_single_iv_from_qtl_row(payload,qtl_method=_WORKER["qtl_label"]);cov=_WORKER["cov"].copy();pred,dose=direct_component(model,_WORKER["gsrc"],cov,snp);z=cov.copy();z["gene"]=gene;z["lead_snp"]=snp;z["artifact_id"]=aid;z["grex_single_iv"]=pred;z["iv_dosage"]=dose;z["component_source"]="full_frozen_PPMI_StageI_temporal_single_IV_row";z,support=_support_columns(z,_WORKER["stage1_followup_min"],_WORKER["stage1_followup_max"]);z["support_definition"]="not_assessed_all_observed_MMSE_followups";keep=[c for c in ["single_iv_row_id","cohort","participant_id","time","gene","lead_snp","artifact_id","grex_single_iv","iv_dosage","support_definition","component_source"] if c in z.columns];z[keep].to_parquet(pf,index=False,compression="zstd");return _summary(aid,gene,snp,z,support),None
    except Exception as e:return {"artifact_id":str(row.get("artifact_id","")),"gene":str(row.get("gene","")),"lead_snp":str(row.get("lead_snp","")),"status":"error","n_rows":0},{"artifact_id":str(row.get("artifact_id","")),"gene":str(row.get("gene","")),"lead_snp":str(row.get("lead_snp","")),"error":repr(e),"traceback":traceback.format_exc()}

def main():
    ap=argparse.ArgumentParser(description="Materialize or update the frozen PPMI Stage-I temporal single-IV genetic component and explicitly encode its follow-up-time support domain.");ap.add_argument("--root",default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt");ap.add_argument("--analysis-name",default="updrs3");ap.add_argument("--phenotype-cov",required=True);ap.add_argument("--candidate-file",default="");ap.add_argument("--qtl-label",default="submitted_followup_ctar_mptga");ap.add_argument("--single-iv-bfile",default="");ap.add_argument("--plink",default="plink");ap.add_argument("--stage1-followup-min",type=float,default=0.0);ap.add_argument("--stage1-followup-max",type=float,default=None);ap.add_argument("--rewrite-support-only",action="store_true");ap.add_argument("--n-jobs",type=int,default=8);a=ap.parse_args()
    if a.stage1_followup_max is not None:raise ValueError("AD all-followup materializer does not accept a Stage1 followup cutoff")
    if a.rewrite_support_only:raise ValueError("AD full-followup protocol does not support --rewrite-support-only")
    root=Path(a.root);out=root/"single_iv_causal"/a.analysis_name;(out/"single_iv_genetic_component").mkdir(parents=True,exist_ok=True);cf=Path(a.candidate_file) if a.candidate_file else root/"single_iv_causal"/"iv_validation"/"single_iv_primary_eligible.csv";cand=pd.read_csv(cf,low_memory=False);bfile=a.single_iv_bfile or str(out/"genotype"/"single_iv_snps");need={"gene","lead_snp","artifact_id"};missing=need-set(cand.columns)
    if not a.rewrite_support_only:need|={"basis_json"};missing=need-set(cand.columns)
    if missing:raise ValueError(f"candidate file missing {sorted(missing)}")
    if not a.rewrite_support_only:
        for ext in [".bed",".bim",".fam"]:
            if not Path(bfile+ext).exists():raise FileNotFoundError(bfile+ext)
    cov=clean_cov(a.phenotype_cov);cache=out/"phenotype_clean.parquet";cov.to_parquet(cache,index=False,compression="zstd");rows=cand.to_dict("records");cfg={"cov_cache":str(cache),"bfile":bfile,"plink":a.plink,"out":str(out),"qtl_label":a.qtl_label,"stage1_followup_min":None,"stage1_followup_max":None,"rewrite_support_only":bool(a.rewrite_support_only)};summ=[];errs=[]
    if a.n_jobs<=1:
        _init(cfg)
        for r in rows:s,e=_one(r);summ.append(s);errs.extend([e] if e else [])
    else:
        with ProcessPoolExecutor(max_workers=a.n_jobs,initializer=_init,initargs=(cfg,)) as ex:
            futs=[ex.submit(_one,r) for r in rows]
            for i,f in enumerate(as_completed(futs),1):
                s,e=f.result();summ.append(s);errs.extend([e] if e else [])
                if i%50==0 or i==len(futs):print(f"components/support {i}/{len(futs)}",flush=True)
    sdf=pd.DataFrame(summ).sort_values(["gene","lead_snp"]);sdf.to_csv(out/"single_iv_genetic_component_summary.csv",index=False);pd.DataFrame(errs).to_csv(out/"single_iv_genetic_component_errors.csv",index=False)
    manifest={"analysis_name":a.analysis_name,"candidate_file":str(cf),
              "conceptual_role":"frozen PPMI Stage1 temporal single-IV genetic-expression component",
              "temporal_component":"complete frozen Stage1 genetic function evaluated at every observed MMSE follow-up time",
              "stage1_time_axis":"followup_time","stage1_model_cohort":"PP",
              "stage1_followup_min":None,"stage1_followup_max":None,
              "support_definition":"NOT ASSESSED: all valid MMSE follow-up times included; predictions beyond Stage1 domain may be extrapolations",
              "rewrite_support_only":False,"n_candidate_pairs":int(len(cand)),
              "n_candidate_genes":int(cand.gene.nunique()),"n_component_errors":int(len(errs)),
              "bfile":bfile}
    (out/"single_iv_genetic_component_manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)
    if errs:
        print(f"[ERROR] {len(errs)} GReX prediction jobs failed; inspect single_iv_genetic_component_errors.csv",flush=True)
        raise RuntimeError(f"Failed to predict frozen Stage1 GReX for {len(errs)} IV-gene pairs")
if __name__=="__main__":main()
