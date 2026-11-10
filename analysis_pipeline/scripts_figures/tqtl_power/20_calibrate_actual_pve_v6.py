#!/usr/bin/env python3
"""Calibrate actual full CTAR eight-coefficient genetic-trajectory PVE from real RNA.

Read-only. No Stage-I/II refitting or IV selection.  The 'signal fraction' is
nuisance-residualized *frozen fitted genetic curve* energy divided by energy
plus residual (Y - frozen fitted curve) after nuisance adjustment. This is a
DESCRIPTIVE, selection-biased signal-to-noise fraction, not unbiased causal PVE.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

METHOD='submitted_followup_ctar_mptga'
BETAS=[f'beta_g_{j}' for j in range(8)]

def frozen_effect(B,g,beta):
    b=np.asarray(beta,float)
    if B.shape[1]!=4 or b.shape!=(8,):raise ValueError('Requires 4 time basis columns and 8 CTAR coefficients')
    # Stage-I genotype_channels(mode='binary'): g==2 -> channel0, g==1 -> channel1.
    return (g==2)*(B@b[:4])+(g==1)*(B@b[4:])

def adjusted_fraction(X0, y, f):
    if not np.isfinite(y).all() or not np.isfinite(f).all():return np.nan,np.nan
    # Same nuisance predictors as production CTAR (continuous-time covariance
    # ignored *only* for this descriptive effect-size calibration).
    d1=np.linalg.lstsq(X0,f,rcond=1e-10)[0]
    d2=np.linalg.lstsq(X0,y-f,rcond=1e-10)[0]
    sig=np.sum((f-X0@d1)**2)
    err=np.sum(((y-f)-X0@d2)**2)
    if not (sig>0 and err>0):return np.nan,np.nan
    return float(sig/(sig+err)),float(np.sqrt(err/len(y)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True);ap.add_argument('--code',required=True);ap.add_argument('--figroot',required=True)
    ap.add_argument('--n-calibration',type=int,default=500)
    ap.add_argument('--seed',type=int,default=20261029)
    ap.add_argument('--min-valid',type=int,default=35)
    ap.add_argument('--candidates',default='')
    ap.add_argument('--batch',type=int,default=30)
    a=ap.parse_args()
    sys.path.insert(0,a.code)
    from gengps.amp_pd.config import AmpPdPaths
    from gengps.amp_pd.real_qtl import (load_cohort_metadata,load_expression_columns,
        resolve_expression_gene_columns,bfile_for_cohort,_augment_fixed_genotype_pcs,
        _prepare_irregular_followup_rows,method_covariates,_drop_missing_covariates)
    from gengps.amp_pd.genotype import make_genotype_source
    from gengps.basis import TimeBasis
    root=Path(a.root);out=Path(a.figroot)/'power_analysis_CTAR_v6';out.mkdir(parents=True,exist_ok=True)
    infile=Path(a.candidates) if a.candidates else root/'single_iv_causal/iv_validation/single_iv_primary_eligible.csv'
    if not infile.is_file():raise FileNotFoundError(infile)
    z=pd.read_csv(infile,dtype={'gene':str,'lead_snp':str},low_memory=False)
    required=['gene','lead_snp','basis_json']+BETAS
    missing=set(required)-set(z.columns)
    if missing:raise RuntimeError(f'{infile}: missing {sorted(missing)}')
    z=z.dropna(subset=required).copy()
    if 'ppmi_tqtl_p' in z.columns:z=z.sort_values('ppmi_tqtl_p')
    z=z.drop_duplicates(['gene','lead_snp'])
    if len(z)<a.min_valid:raise RuntimeError('Too few 8-parameter real IV-gene curves')
    rng=np.random.default_rng(a.seed)
    n=min(a.n_calibration,len(z))
    selected=z.iloc[np.sort(rng.choice(len(z),size=n,replace=False))].reset_index(drop=True)
    paths=AmpPdPaths(root=str(root))
    meta=load_cohort_metadata(paths,'PP')
    expr_file=root/'data/expression/amp_pd_expression_z.parquet'
    ids=set(pd.read_parquet(expr_file,columns=['sample_id']).sample_id.astype(str))
    meta=meta[meta.sample_id.astype(str).isin(ids)].copy()
    meta=_augment_fixed_genotype_pcs(paths,'PP',meta,n_pcs=8)
    meta,_,duplicates=_prepare_irregular_followup_rows(meta,pd.DataFrame(index=meta.index))
    covars=method_covariates(METHOD,'PP',meta,cell_adjust=True)
    meta,_=_drop_missing_covariates(meta,pd.DataFrame(index=meta.index),covars)
    partids=pd.Index(meta.participant_id.astype(str).unique(),name='participant_id')
    validmap=resolve_expression_gene_columns(paths,selected.gene.astype(str).unique().tolist(),allow_unresolved=True)
    bfile=bfile_for_cohort(paths,'PP');gs=make_genotype_source(str(bfile),paths.plink19,prefer_bed_reader=True)
    # Real covariates, identical categories to the CTAR Stage1 nuisance design.
    t=meta.followup_time.to_numpy(float)
    Xt=np.column_stack([np.ones(len(meta)),t,t*t,t*t*t,meta[covars].to_numpy(float)])
    obs_pid=meta.participant_id.astype(str).to_numpy()
    results=[];errors=[]
    for start in range(0,len(selected),a.batch):
        block=selected.iloc[start:start+a.batch]
        genes=list(dict.fromkeys(validmap[g] for g in block.gene if g in validmap))
        if not genes:continue
        expr=load_expression_columns(paths,genes).reindex(meta.index)
        snps=block.lead_snp.drop_duplicates().astype(str).tolist()
        try:gt=gs.read(partids,snps,mean_impute=False,dtype=np.float64)
        except Exception as e:
            raise RuntimeError(f'Cannot read frozen Stage1 calibration SNPs {snps[:5]} from {bfile}') from e
        gobs=gt.reindex(obs_pid)
        for row in block.itertuples(index=False):
            d=row._asdict();gene=str(d['gene']);snp=str(d['lead_snp'])
            try:
                if gene not in validmap:raise KeyError('Ensembl gene not mapped to expression column')
                y=pd.to_numeric(expr[validmap[gene]],errors='coerce').to_numpy(float)
                g=pd.to_numeric(gobs[snp],errors='coerce').to_numpy(float)
                called=np.isfinite(g)&np.isin(g,[0,1,2])&np.isfinite(y)
                if called.sum()<100:raise ValueError('Insufficient paired observations with real hardcalls')
                b=TimeBasis.from_dict(json.loads(d['basis_json'])).transform(t[called])
                theta=np.array([d[x] for x in BETAS],float)
                f=frozen_effect(b,g[called],theta)
                frac,sigma=adjusted_fraction(Xt[called],y[called],f)
                # Original PVE is per observed RNA visit. Since CTAR accepts
                # irregular visits, also report donor-equal-weight PVE as sensitivity.
                ids_observed=obs_pid[called]
                visit_counts=pd.Series(ids_observed).value_counts().to_dict()
                w=np.asarray([1./visit_counts[x] for x in ids_observed],float)
                yf=y[called]
                mf=np.average(f,weights=w);my=np.average(yf,weights=w)
                vwf=np.average((f-mf)**2,weights=w)
                vwy=np.average((yf-my)**2,weights=w)
                pve_donor_balanced=float(vwf/vwy) if vwy>0 else np.nan
                if not (np.isfinite(frac) and 0<frac<1 and np.isfinite(sigma) and sigma>0):
                    raise ValueError('Non-finite real signal fraction or residual scale')
                gv=gt[snp].to_numpy(float)
                called_person=np.isfinite(gv)&np.isin(gv,[0,1,2])
                dose=np.rint(gv[called_person]).astype(int)
                n0,n1,n2=np.bincount(dose,minlength=3)
                maf=float(min(np.mean(dose)/2,1-np.mean(dose)/2))
                results.append({'gene':gene,'lead_snp':snp,'signal_fraction':frac,
                    'residual_sd':sigma,'n_participants_called':len(dose),
                    'n_hom_ref':int(n0),'n_het':int(n1),'n_hom_alt':int(n2),
                    'MAF_observed':maf,'n_expression_obs':int(called.sum()),
                    'observed_PVE_original_VarGenetic_over_VarRNA':float(np.var(f, ddof=1)/np.var(y[called],ddof=1)) if np.var(y[called],ddof=1)>0 else np.nan,
                    'dosage_allele_frequency':float(np.mean(dose)/2.0),
                    'observed_PVE_donor_balanced_sensitivity':pve_donor_balanced,
                    'observed_PVE_nuisance_residualized_OLS':frac,
                    'n_missing_obs_for_calibration':int((~called).sum()),
                    'rho_PP':d.get('rho',np.nan),
                    'basis_json':d['basis_json'],**{x:float(d[x]) for x in BETAS}})
            except Exception as e:errors.append({'gene':gene,'lead_snp':snp,'reason':str(e)})
        print(f'[CALIBRATION] {min(start+a.batch,len(selected))}/{len(selected)} processed, valid={len(results)}',flush=True)
    r=pd.DataFrame(results)
    pd.DataFrame(errors,columns=['gene','lead_snp','reason']).to_csv(out/'calibration_exclusions.csv',index=False)
    if len(r)<a.min_valid:raise RuntimeError(f'Only {len(r)} real calibrated pairs; see calibration_exclusions.csv')
    r.to_csv(out/'calibrated_real_full8_CTAR_PVE.csv',index=False)
    # Select one profile for each empirical quantile. The user can inspect
    # every selected gene, SNP, p, MAF, and genotype class count.
    picks=[];used=set()
    for label,percentile in [('Lower',.20),('Median',.50),('Higher',.80)]:
        target=float(r.signal_fraction.quantile(percentile))
        ordering=np.argsort(np.abs(r.signal_fraction.to_numpy(float)-target))
        i=next((int(i) for i in ordering if int(i) not in used),None)
        if i is None:raise RuntimeError('Cannot choose distinct real profiles')
        used.add(i);q=r.iloc[i].to_dict();q.update(template_label=label,quantile=percentile,quantile_target=target)
        picks.append(q)
    pd.DataFrame(picks).to_csv(out/'three_real_CTAR_curve_templates_audit.csv',index=False)
    print('[EMPIRICAL TEMPLATE SUMMARY]')
    print(pd.DataFrame(picks)[['template_label','gene','lead_snp','signal_fraction','MAF_observed','n_hom_alt']].to_string(index=False))
    (out/'calibration_manifest.json').write_text(json.dumps({
        'source':str(infile),'n_sampled':n,'n_calibrated':len(r),
        'n_excluded':len(errors),'n_PPMI_participants':int(len(partids)),
        'n_PP_RNA_visits':len(meta),'duplicate_observations_resolved':duplicates,
        'effect_definition':'8-dimensional frozen Stage1 CTAR genotype-specific trajectory, NOT additive dosage',
        'observed_original_PVE':'sample Var(frozen full 8-param genetic curve evaluated at observed genotype and visit time) / sample Var(observed RNA), as in original manuscript Eq.(9); descriptive, may exceed one due to estimation error',
        'observed_partial_PVE':'nuisance-OLS-residualized frozen genetic curve energy / (that energy + residualized observed RNA minus fitted genetic curve energy); in-sample proxy, not unbiased partial R2 nor causal heritability',
        'bias_warning':'Selected significant replicated tQTLs are subject to winner\'s curse; treat templates as illustrative rather than unbiased population-effect distribution',
        'no_new_IV_QC':True},indent=2))

if __name__=='__main__':main()
