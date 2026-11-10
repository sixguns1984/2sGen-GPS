#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
_BACKEND_READY=False

def _init_process(train_cfg):
    global _BACKEND_READY
    from gengps.amp_pd.stage2_strength import init_worker
    init_worker(train_cfg);_BACKEND_READY=True

def _run_batch(tasks):
    try:
        from gengps.amp_pd.stage2_strength import process_pair_batch
        return process_pair_batch(tasks)
    except Exception as e:
        tb=traceback.format_exc();return [{"gene":t.get("gene",""),"error":repr(e),"traceback":tb,"metrics_pp":[],"metrics_ext":[],"legacy_train":[]} for t in tasks]

def _val(row,names):
    if row is None:return np.nan
    for c in names:
        if c in row.index:
            v=pd.to_numeric(pd.Series([row[c]]),errors="coerce").iloc[0]
            if np.isfinite(v):return float(v)
    return np.nan

def _select(rows,cohort=None,level=None):
    if not rows:return None
    d=pd.DataFrame(rows)
    if cohort is not None and "cohort" in d.columns:d=d[d.cohort.astype(str).str.upper().eq(str(cohort).upper())]
    if level is not None and "level" in d.columns:d=d[d.level.astype(str).eq(level)]
    return d.iloc[0] if len(d) else None

def _parse_result(res,lookup):
    aid=str(res.get("gene",""));meta=lookup.get(aid,{})
    pp=_select(res.get("legacy_train",[]),"PPMI","sample_training_apparent")
    if pp is None:
        x=res.get("legacy_train",[])
        if x:
            d=pd.DataFrame(x);d=d[d.level.astype(str).str.startswith("sample")] if "level" in d.columns else d;pp=d.iloc[0] if len(d) else None
    pds=_select(res.get("metrics_ext",[]),"PDBP","sample");pdw=_select(res.get("metrics_ext",[]),"PDBP","within_participant");out=dict(meta)
    ppF=_val(pp,["grex_score_F","first_stage_F"]);ppP=_val(pp,["grex_score_p","first_stage_p"]);ppB=_val(pp,["grex_score_beta","first_stage_beta"]);ppSE=abs(ppB)/np.sqrt(ppF) if np.isfinite(ppB) and np.isfinite(ppF) and ppF>0 else np.nan;ppPear=_val(pp,["pearson"]);ppFrozen=_val(pp,["R2_frozen"]);pdF=_val(pds,["grex_score_F","first_stage_F"]);pdP=_val(pds,["grex_score_p","first_stage_p"]);pdB=_val(pds,["grex_score_beta","first_stage_beta"]);pdSE=abs(pdB)/np.sqrt(pdF) if np.isfinite(pdB) and np.isfinite(pdF) and pdF>0 else np.nan;pdPear=_val(pds,["pearson"]);pdFrozen=_val(pds,["R2_frozen"])
    out.update({"ppmi_score_strength_beta":ppB,"ppmi_score_strength_se":ppSE,"ppmi_score_strength_F":ppF,"ppmi_score_strength_p":ppP,"ppmi_score_partial_R2":float(ppPear*ppPear) if np.isfinite(ppPear) else np.nan,"ppmi_frozen_R2":ppFrozen,"ppmi_first_stage_beta":ppB,"ppmi_first_stage_se":ppSE,"ppmi_first_stage_F":ppF,"ppmi_first_stage_p":ppP,"ppmi_first_stage_partial_R2":float(ppPear*ppPear) if np.isfinite(ppPear) else np.nan,"ppmi_full_genetic_block_partial_R2":_val(pp,["full_genetic_block_partial_R2_descriptive"]),"ppmi_full_genetic_block_rank":_val(pp,["full_genetic_block_rank"]),"ppmi_sample_rho":_val(pp,["spearman","pearson"]),"pdbp_score_strength_beta":pdB,"pdbp_score_strength_se":pdSE,"pdbp_score_strength_F":pdF,"pdbp_score_strength_p":pdP,"pdbp_score_partial_R2":float(pdPear*pdPear) if np.isfinite(pdPear) else np.nan,"pdbp_first_stage_beta":pdB,"pdbp_first_stage_se":pdSE,"pdbp_first_stage_F":pdF,"pdbp_first_stage_p":pdP,"pdbp_frozen_R2":pdFrozen,"pdbp_full_genetic_block_partial_R2":_val(pds,["full_genetic_block_partial_R2_descriptive"]),"pdbp_full_genetic_block_rank":_val(pds,["full_genetic_block_rank"]),"pdbp_sample_rho":_val(pds,["spearman","pearson"]),"pdbp_within_rho":_val(pdw,["spearman","pearson"]),"ppmi_strength_n":int(_val(pp,["n","n_samples"])) if np.isfinite(_val(pp,["n","n_samples"])) else np.nan,"pdbp_strength_n":int(_val(pds,["n","n_samples"])) if np.isfinite(_val(pds,["n","n_samples"])) else np.nan,"pdbp_strength_n_participants":int(_val(pds,["n_participants"])) if np.isfinite(_val(pds,["n_participants"])) else np.nan,"strength_status":"error" if res.get("error") else "ok","strength_error":str(res.get("error",""))})
    return out

def _build_tasks(cand,expr_map):
    tasks=[];lookup={}
    for r in cand.itertuples(index=False):
        d=r._asdict();gene=str(d["gene"]);snp=str(d["lead_snp"]);aid=str(d["artifact_id"]);pseudo=f"IVPAIR__{aid}";payload=dict(d);payload["Variant_ID"]=snp;payload["Ensembl_ID"]=gene;payload["Pvalue"]=float(pd.to_numeric(pd.Series([d.get("ppmi_tqtl_p",np.nan)]),errors="coerce").iloc[0]);beta=[c for c in payload if str(c).startswith("beta_g_")]
        if "basis_json" not in payload or not beta:raise ValueError(f"{aid} lacks basis_json/beta_g_* needed for the frozen single-IV temporal MPTGA component")
        tasks.append({"gene":pseudo,"expr_gene":expr_map[gene],"snps":[snp],"legacy_row":payload});lookup[pseudo]={"artifact_id":aid,"gene":gene,"lead_snp":snp}
    return tasks,lookup

def _group_batches(tasks,lookup,batch_genes=8):
    by_gene={}
    for t in tasks:
        g=lookup[t["gene"]]["gene"];by_gene.setdefault(g,[]).append(t)
    for g in by_gene:by_gene[g]=sorted(by_gene[g],key=lambda t:t["gene"])
    active=sorted(by_gene);batches=[];bs=max(1,int(batch_genes))
    while active:
        chosen=active[:bs];batch=[];next_active=[]
        for g in chosen:
            batch.append(by_gene[g].pop(0))
            if by_gene[g]:next_active.append(g)
        next_active.extend(active[bs:]);batches.append(batch);active=next_active
    return batches

def main():
    ap=argparse.ArgumentParser(description="Compute descriptive first-stage strength/expression-support metrics for every unified temporal-tQTL IV. These metrics NEVER redefine primary IV eligibility.")
    ap.add_argument("--root",default="/public2/data/luojunfeng/2sGenGPS_revised_chatgpt");ap.add_argument("--qtl-label",default="submitted_followup_ctar_mptga");ap.add_argument("--candidate-file",default="");ap.add_argument("--n-jobs",type=int,default=4);ap.add_argument("--blas-threads",type=int,default=1);ap.add_argument("--batch-genes",type=int,default=8);ap.add_argument("--out-dir",default="")
    a=ap.parse_args();root=Path(a.root).resolve();ivdir=root/"single_iv_causal"/"iv_validation";cf=Path(a.candidate_file) if a.candidate_file else ivdir/"single_iv_primary_eligible.csv"
    if not cf.exists():raise FileNotFoundError(cf)
    cand=pd.read_csv(cf,low_memory=False);req={"artifact_id","gene","lead_snp","ppmi_tqtl_p","basis_json"};miss=req-set(cand.columns)
    if miss:raise ValueError(f"candidate file missing {sorted(miss)}")
    from gengps.amp_pd.config import AmpPdPaths
    from gengps.amp_pd.real_qtl import resolve_expression_gene_columns
    from gengps.amp_pd.stage2_strength import init_worker,process_pair_batch
    paths=AmpPdPaths(root=str(root));genes=sorted(cand.gene.astype(str).unique());expr_map=resolve_expression_gene_columns(paths,genes,allow_unresolved=True);unresolved=[g for g in genes if g not in expr_map]
    if unresolved:raise RuntimeError(f"{len(unresolved)} final IV genes cannot be resolved in expression matrix; first={unresolved[:20]}")
    out=Path(a.out_dir) if a.out_dir else root/"single_iv_causal"/"iv_strength_all";out.mkdir(parents=True,exist_ok=True)
    for d in ["models/legacy_single_iv","legacy_training_predictions","pdbp_predictions","fold_metrics"]:(out/d).mkdir(parents=True,exist_ok=True)
    tasks,lookup=_build_tasks(cand,expr_map);batches=_group_batches(tasks,lookup,a.batch_genes);cfg={"root":str(root),"out_dir":str(out),"qtl_label":a.qtl_label,"blas_threads":max(1,int(a.blas_threads))};parsed=[];errors=[]
    if a.n_jobs<=1:
        init_worker(cfg)
        for i,b in enumerate(batches,1):
            for res in process_pair_batch(b):
                z=_parse_result(res,lookup);parsed.append(z)
                if res.get("error"):errors.append({"artifact_id":z.get("artifact_id",""),"gene":z.get("gene",""),"lead_snp":z.get("lead_snp",""),"error":res.get("error",""),"traceback":res.get("traceback","")})
            if i%10==0 or i==len(batches):print(f"strength batches {i}/{len(batches)}",flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.n_jobs,initializer=_init_process,initargs=(cfg,)) as ex:
            futs=[ex.submit(_run_batch,b) for b in batches]
            for i,f in enumerate(as_completed(futs),1):
                for res in f.result():
                    z=_parse_result(res,lookup);parsed.append(z)
                    if res.get("error"):errors.append({"artifact_id":z.get("artifact_id",""),"gene":z.get("gene",""),"lead_snp":z.get("lead_snp",""),"error":res.get("error",""),"traceback":res.get("traceback","")})
                if i%10==0 or i==len(futs):print(f"strength batches {i}/{len(futs)}",flush=True)
    strength=pd.DataFrame(parsed).sort_values(["gene","lead_snp"])
    # Deprecated aliases: old 'first_stage_F' was frozen GReX score t^2, not a four-term genetic IV strength test.
    # Preserve as explicit score metric only; never use a classical F>=10 classification.
    legacy_cols=[c for c in strength if "first_stage_" in c]
    strength=strength.drop(columns=legacy_cols)
    strength.to_csv(out/"all_IV_first_stage_strength.csv",index=False);pd.DataFrame(errors).to_csv(out/"errors.csv",index=False)
    drop=[c for c in ["ppmi_score_strength_beta","ppmi_score_strength_se","ppmi_score_strength_F","ppmi_score_strength_p","ppmi_score_partial_R2","ppmi_frozen_R2","ppmi_first_stage_beta","ppmi_first_stage_se","ppmi_first_stage_F","ppmi_first_stage_p","ppmi_first_stage_partial_R2","ppmi_sample_rho","pdbp_score_strength_beta","pdbp_score_strength_se","pdbp_score_strength_F","pdbp_score_strength_p","pdbp_score_partial_R2","pdbp_first_stage_beta","pdbp_first_stage_se","pdbp_first_stage_F","pdbp_first_stage_p","pdbp_frozen_R2","pdbp_sample_rho","pdbp_within_rho","strength_status","strength_error"] if c in cand.columns]
    merged=cand.drop(columns=drop,errors="ignore").merge(strength,on=["artifact_id","gene","lead_snp"],how="left",validate="one_to_one");merged["pdbp_sample_rho_positive_descriptive"]=pd.to_numeric(merged.pdbp_sample_rho,errors="coerce")>0;merged["pdbp_within_rho_positive_descriptive"]=pd.to_numeric(merged.pdbp_within_rho,errors="coerce")>0;merged["iv_strength_available"]=merged.strength_status.astype(str).eq("ok")&np.isfinite(pd.to_numeric(merged.ppmi_full_genetic_block_partial_R2,errors="coerce"));merged["iv_primary_eligible"]=True;merged=merged.sort_values(["gene","ppmi_tqtl_p","lead_snp"]);merged.to_csv(out/"single_iv_primary_eligible_with_updated_strength.csv",index=False)
    audit=pd.DataFrame([{"n_primary_IVs":int(len(merged)),"n_genes_primary":int(merged.gene.nunique()),"n_strength_ok":int(merged.iv_strength_available.sum()),"strength_coverage":float(merged.iv_strength_available.mean()) if len(merged) else np.nan,"n_full_genetic_R2_available":int(pd.to_numeric(merged.ppmi_full_genetic_block_partial_R2,errors="coerce").notna().sum()),"n_pdbp_sample_rho_positive":int(merged.pdbp_sample_rho_positive_descriptive.fillna(False).sum()),"n_pdbp_within_rho_positive":int(merged.pdbp_within_rho_positive_descriptive.fillna(False).sum())}]);audit.to_csv(out/"single_iv_allIV_strength_summary.csv",index=False)
    manifest={"qtl_label":a.qtl_label,"candidate_file":str(cf),"strength_output":str(out/"all_IV_first_stage_strength.csv"),"method":"each LD-independent replicated temporal tQTL is evaluated separately using its own frozen Stage-I MPTGA coefficients","backend":"gengps.amp_pd.stage2_strength; no retired multi-tQTL GReX script dependency","no_multi_snp_model":True,"eligibility":"Primary IV eligibility was fixed BEFORE this script by PPMI temporal FDR<0.05 + PDBP temporal P<0.05 + trajectory_r>0 + PPMI LD pruning. F/R2/rho are descriptive diagnostics only.","F_ge_10":"deprecated; NOT a valid rule for ctAR temporal genetic component","n_primary_IVs":int(len(merged)),"n_strength_ok":int(merged.iv_strength_available.sum()),"strength_coverage":float(merged.iv_strength_available.mean()) if len(merged) else np.nan};(out/"single_iv_allIV_strength_manifest.json").write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2));print("\n",audit.to_string(index=False))
if __name__=="__main__":main()
