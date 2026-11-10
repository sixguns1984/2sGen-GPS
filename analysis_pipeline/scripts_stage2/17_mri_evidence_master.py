#!/usr/bin/env python3
"""MRI-DBM post-selection evidence: AR-type, multi-IV, Q/I2, LOO, LD and UPDRS provenance.

Uses original Stage-II MRI visit alignment and cluster-robust OLS; all auxiliary IVs
come from outcome-independent Stage-I eligibility. This is an exploratory PPMI-only
imaging robustness profile, not a new independent causal-identification test.
"""
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2,norm
from statsmodels.stats.multitest import multipletests

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import importlib
mri=importlib.import_module('14_mri_dbm_lag1')

def read_csv(path,required=()):
    p=Path(path)
    if not p.is_file():raise FileNotFoundError(p)
    d=pd.read_csv(p,low_memory=False,dtype={'gene':'string','lead_snp':'string','artifact_id':'string'})
    missing=set(required)-set(d.columns)
    if missing:raise ValueError(f'Missing columns in {p}: {missing}')
    return d

def yes(s):return s.astype('string').str.lower().str.strip().isin(('true','yes','1')).fillna(False).astype(bool)

def qvals(p):
    x=pd.to_numeric(p,errors='coerce');out=pd.Series(np.nan,index=x.index,dtype=float)
    good=np.isfinite(x)&x.between(0,1)
    if good.any():out.loc[good]=multipletests(x.loc[good],method='fdr_bh')[1]
    return out

def fixed_pool(d):
    b=pd.to_numeric(d['beta_standardized_GReX'],errors='coerce').to_numpy(float)
    s=pd.to_numeric(d['se_standardized_GReX'],errors='coerce').to_numpy(float)
    good=np.isfinite(b)&np.isfinite(s)&(s>0)
    b=b[good];s=s[good]
    if len(b)<2:return None
    w=1/s**2; beta=float(np.sum(w*b)/np.sum(w)); se=float(np.sqrt(1/np.sum(w)))
    p=float(2*norm.sf(abs(beta/se)))
    Q=float(np.sum(w*(b-beta)**2));df=int(len(b)-1)
    return {'multiIV_n':int(len(b)),'multiIV_beta':beta,'multiIV_se':se,
            'multiIV_p':p,'multiIV_Q':Q if df>0 else np.nan,'multiIV_Q_df':df if df>0 else np.nan,
            'multiIV_Q_p':float(chi2.sf(Q,df)) if df>0 else np.nan,
            'multiIV_I2':float(max(0,(Q-df)/Q)) if df>0 and Q>0 else (0.0 if df>0 else np.nan),
            'multiIV_direction_concordance_fraction':float(max((b>0).mean(),(b<0).mean()))}

def multi_summary(details,expected):
    rows=[];loos=[]
    for (gene,roi),z in details.groupby(['gene','roi'],dropna=False,sort=True):
        z=z[z.ok].drop_duplicates('lead_snp').copy()
        pooled=fixed_pool(z)
        r={'gene':gene,'roi':roi,'n_eligible_stage1_IVs':int(expected.get(str(gene),0)),
           'n_estimable_independent_IVs':int(len(z))}
        if pooled:r.update(pooled)
        else:r['multiIV_n']=int(len(z))
        if pooled and len(z)>=3:
            loo=[]
            for j in range(len(z)):
                z2=z.drop(index=z.index[j]);p=fixed_pool(z2)
                if p is None:continue
                val={'gene':gene,'roi':roi,'left_out_snp':str(z.iloc[j].lead_snp),
                     'n_remaining':int(len(z2)),'beta':p['multiIV_beta'],'se':p['multiIV_se'],'p':p['multiIV_p']}
                loos.append(val);loo.append(val)
            bs=np.array([x['beta'] for x in loo]);ps=np.array([x['p'] for x in loo]);r['LOO_n']=len(loo)
            r['LOO_all_same_direction']=bool(np.all(np.sign(bs)==np.sign(pooled['multiIV_beta']))) if len(loo) else np.nan
            r['LOO_max_p']=float(np.nanmax(ps)) if len(loo) else np.nan
            r['LOO_all_nominal_p05']=bool(np.all(ps<.05)) if len(loo) else np.nan
        else:r.update(LOO_n=0,LOO_all_same_direction=np.nan,LOO_max_p=np.nan,LOO_all_nominal_p05=np.nan)
        rows.append(r)
    return pd.DataFrame(rows),pd.DataFrame(loos,columns=['gene','roi','left_out_snp','n_remaining','beta','se','p'])

def annotations(hits,ar,multi,clinical,ldflags):
    keys=['gene','lead_snp','artifact_id','roi']
    if hits.duplicated(keys).any():raise ValueError('Duplicate MRI FDR hit keys')
    # Exact AR from independent-existing MRI robust script (not necessarily classical AR).
    arcols=[c for c in ar.columns if c not in ['gene','lead_snp','artifact_id','roi']]
    t=hits.merge(ar[keys+arcols],on=keys,how='left',validate='one_to_one',suffixes=('','_ar'))
    t=t.merge(multi,on=['gene','roi'],how='left',validate='many_to_one')
    if clinical.duplicated(['gene','lead_snp']).any():raise ValueError('Duplicated clinical IV-gene pairs')
    want=['gene','lead_snp','phenotypes_significant','n_significant_phenotypes']+[f'significant_updrs{i}' for i in (1,2,3)]
    cc=clinical[[c for c in want if c in clinical]].copy()
    for ph in ['updrs1','updrs2','updrs3']:
        k='significant_'+ph
        if k not in cc:raise ValueError(f'Missing clinical phenotype provenance {k}')
        cc[k]=yes(cc[k])
    cc['phenotypes_significant']=cc.apply(lambda r:';'.join(ph for ph in ('updrs1','updrs2','updrs3') if r['significant_'+ph]),axis=1)
    cc['n_significant_phenotypes']=cc[['significant_updrs1','significant_updrs2','significant_updrs3']].sum(axis=1)
    t=t.merge(cc,on=['gene','lead_snp'],how='left',validate='many_to_one')
    if t['phenotypes_significant'].isna().any():raise ValueError('MRI hits not found in three-UPDRS clinical IV-source; examine candidate provenance')
    t=t.merge(ldflags,on=['gene','lead_snp'],how='left',validate='many_to_one')
    # Uncomputable LD remains NA; never infer 'LD clean' from missing data.
    if 'cross_gene_ld_flag' not in t:t['cross_gene_ld_flag']=pd.NA
    for col in ['iv_reduced_form_p','multiIV_p_q_BH','multiIV_Q_p_q_BH',
                'LOO_all_same_direction','LOO_all_nominal_p05',
                'multiIV_n','multiIV_direction_concordance_fraction']:
        if col not in t:t[col]=pd.NA
    t['iv_reduced_form_q_BH']=qvals(t['iv_reduced_form_p'])
    # Family is unique gene x ROI and is calculated BEFORE merging to lead SNP rows.
    t['multiIV_FDR_support']=t.multiIV_p_q_BH.lt(.05).fillna(False)
    t['multiIV_heterogeneity_FDR_flag']=t.multiIV_Q_p_q_BH.lt(.05).fillna(False)
    t['MRI_AR_type_FDR_support']=t.iv_reduced_form_q_BH.lt(.05).fillna(False)
    t['cross_gene_LD_clean']=t.cross_gene_ld_flag.map(
        lambda x: pd.NA if pd.isna(x) else not (str(x).strip().lower() in ('true','1','yes'))).astype('boolean')

    n=pd.to_numeric(t.get('multiIV_n'),errors='coerce')
    loo_stable=(n.eq(2)&pd.to_numeric(t.multiIV_direction_concordance_fraction,errors='coerce').ge(1.0))|((n>=3)&t.LOO_all_same_direction.eq(True)&t.LOO_all_nominal_p05.eq(True))
    no_heterogeneity=pd.to_numeric(t.multiIV_Q_p_q_BH,errors='coerce').ge(.05)
    multi_stable=t.multiIV_FDR_support & n.ge(2) & no_heterogeneity & loo_stable.fillna(False).astype(bool) & t.cross_gene_LD_clean.fillna(False).astype(bool)
    ar_stable=t.MRI_AR_type_FDR_support & no_heterogeneity & t.cross_gene_LD_clean.fillna(False).astype(bool)
    tier=np.full(len(t),'MRI_C_original_FDR_only',dtype=object)
    tier[t.MRI_AR_type_FDR_support.to_numpy(bool)]='MRI_B_AR_type_supported'
    tier[(t.multiIV_FDR_support & n.ge(2)).fillna(False).to_numpy(bool)]='MRI_B_multiIV_supported'
    # Unlike UPDRS, MRI lacks independent external imaging replication; AR alone
    # cannot justify an A-level cross-cohort descriptor.
    tier[multi_stable.to_numpy(bool)]='MRI_A_multiIV_consistent'
    tier[(ar_stable&multi_stable).to_numpy(bool)]='MRI_A_both_postselection_support'
    t['MRI_descriptive_evidence_tier']=tier
    t['evidence_tier_is_selection_gate']=False
    t['MRI_replication_available']=False
    t['AR_test_is_classical_Anderson_Rubin']=False
    return t

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--component',type=int,default=1)
    ap.add_argument('--mri-file',default='/public2/data/luojunfeng/2sTGCS_binary/outcome/DBM/confound_pass_snp/MRI_DBM.csv')
    ap.add_argument('--hits',default='');ap.add_argument('--all',default='');ap.add_argument('--clinical-wide',default='');ap.add_argument('--ar-detail',default='')
    ap.add_argument('--aux-analysis-name',default='mri_dbm_lag1_multiiv');ap.add_argument('--aux-candidates',default='');ap.add_argument('--ld-file',default='');ap.add_argument('--out-dir',default='')
    ap.add_argument('--min-participants',type=int,default=30);ap.add_argument('--min-obs',type=int,default=50)
    args=ap.parse_args();root=Path(args.root);base=root/f'single_iv_causal/mri_dbm_lag1/pc{args.component}'
    out=Path(args.out_dir) if args.out_dir else base/'evidence_v1';out.mkdir(parents=True,exist_ok=True)
    hits=read_csv(args.hits or base/'MRI_DBM_lag1_global_FDR05.csv',['gene','lead_snp','artifact_id','roi','p_cluster','q_global_BH'])
    if hits.empty:raise ValueError('No frozen MRI global BH FDR hits')
    qs=pd.to_numeric(hits.q_global_BH,errors='coerce')
    if qs.isna().any() or (qs>=.05).any() or (qs<0).any():raise ValueError('MRI FDR hit table contains missing/out-of-range q values; cannot freeze primary selection')
    alltab=read_csv(args.all or base/'MRI_DBM_lag1_ALL_associations.csv')
    annotation=read_csv(args.clinical_wide or root/'single_iv_causal/cross_phenotype_summary_lag1/PD_three_phenotypes_publication_wide.csv',['gene','lead_snp'])
    ar_path=Path(args.ar_detail or base/'publication_robustness/MRI_DBM_IV_reduced_form_AR_type_diagnostics.csv')
    if ar_path.exists():
        ar=read_csv(ar_path,['gene','lead_snp','artifact_id','roi'])
    else:
        print('[WARN] AR diagnostics unavailable; no alternative AR method attempted',flush=True)
        ar=hits[['gene','lead_snp','artifact_id','roi']].copy()
    if 'iv_reduced_form_p' not in ar.columns:ar['iv_reduced_form_p']=np.nan
    cand=read_csv(args.aux_candidates or out/'MRI_hit_genes_ALL_eligible_stage1_IVs.csv',['gene','lead_snp','artifact_id'])
    if cand.duplicated(['gene','lead_snp']).any():raise ValueError('Duplicate auxiliary Stage-I IVs')
    expected=cand.groupby('gene').lead_snp.nunique().astype(int).to_dict()
    ldfile=Path(args.ld_file or out/'ld_audit/lead_iv_ld_pairs.csv')
    if ldfile.exists():
        import importlib.util
        spec=importlib.util.spec_from_file_location('stage2_assemble_final',HERE/'10_assemble_final_master.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        ldflags=mod.ld_annotations(ldfile,hits[['gene','lead_snp']].drop_duplicates(),.10)
    else:
        print('[WARN] LD audit unavailable; LD annotations remain NA',flush=True)
        ldflags=hits[['gene','lead_snp']].drop_duplicates().copy()
        ldflags['cross_gene_max_r2']=np.nan
        ldflags['cross_gene_ld_flag']=pd.NA
    pc,pcs=mri.get_pcs(root,'','')
    m,rois=mri.prepare_mri(args.mri_file,pc,pcs,args.component,1e-6)
    rois_needed=sorted(set(hits.roi))
    if not set(rois_needed).issubset(rois):raise ValueError(f'ROI missing in MRI source {set(rois_needed)-set(rois)}')
    gene_rois=hits.groupby('gene').roi.apply(lambda x:sorted(set(x))).to_dict()
    covnames=['age_c','age_c2','age_c3','sex_num','IQR']+pcs
    records=[];errors=[];audit=[]
    # Each IV is tested against ALL hit ROIs of its target gene, independent of its own MRI significance.
    for idx,r in enumerate(cand.itertuples(index=False),1):
        gene=str(r.gene);snp=str(r.lead_snp);aid=str(r.artifact_id)
        try:
            pair=type('Pair',(),{'gene':gene,'lead_snp':snp,'artifact_id':aid})
            g=mri.get_grex(root,pair,args.aux_analysis_name)
            z=m.copy();z['GReX_lag1']=mri.grex_at_times(z,g)['GReX_lag1'].to_numpy()
            ids=z.pid.to_numpy(str);x=z.GReX_lag1.to_numpy(float)
            C=z[covnames].to_numpy(float)
            for roi in gene_rois.get(gene,[]):
                y=z[roi].to_numpy(float);yp=z['prev__'+roi].to_numpy(float)
                mask=np.isfinite(y)&np.isfinite(yp)&np.isfinite(x)&np.isfinite(C).all(axis=1)
                people=int(pd.Series(ids[mask]).nunique());n=int(mask.sum())
                row={'gene':gene,'lead_snp':snp,'artifact_id':aid,'roi':roi,'n_obs':n,'n_people':people,'ok':False,
                     'beta_standardized_GReX':np.nan,'se_standardized_GReX':np.nan,'p_standardized_GReX':np.nan}
                if n<args.min_obs or people<args.min_participants:
                    row['reason']='insufficient_data';records.append(row);continue
                # Rescale each IV-specific GReX to one SD to make multiple-IV coefficients comparable.
                sd=float(np.std(x[mask],ddof=1));
                if sd<1e-12 or not np.isfinite(sd):row['reason']='zero_GReX_variance';records.append(row);continue
                rfit=mri.fit_one(y,yp,x/sd,C,ids,args.min_participants,args.min_obs)
                if rfit is None:row['reason']='singular_fit';records.append(row);continue
                row.update(ok=True,reason='ok',beta_standardized_GReX=float(rfit['beta_grex_lag1']),
                           se_standardized_GReX=float(rfit['se_cluster']),p_standardized_GReX=float(rfit['p_cluster']),
                           GReX_sd=sd)
                records.append(row)
        except Exception as e:errors.append({'gene':gene,'lead_snp':snp,'artifact_id':aid,'error':repr(e)})
        if idx%25==0 or idx==len(cand):print(f'[MRI MULTI-IV] {idx}/{len(cand)} eligible IVs, {len(records)} ROI fits',flush=True)
    details=pd.DataFrame(records)
    if details.empty:
        print('[WARN] No auxiliary IV could be fitted; multi-IV, Q/I2 and LOO remain NA',flush=True)
        details=pd.DataFrame(columns=['gene','lead_snp','artifact_id','roi','ok','beta_standardized_GReX','se_standardized_GReX'])
    details.to_csv(out/'MRI_all_eligible_IVs_ROI_commonlag_details.csv',index=False)
    pd.DataFrame(errors,columns=['gene','lead_snp','artifact_id','error']).to_csv(out/'multi_iv_errors.csv',index=False)
    multi,loo=multi_summary(details,expected)
    # The MRI hypothesis family is unique gene x ROI; correct once per gene x ROI.
    if not multi.empty:
        multi['multiIV_p_q_BH']=qvals(multi['multiIV_p']) if 'multiIV_p' in multi else np.nan
        multi['multiIV_Q_p_q_BH']=qvals(multi['multiIV_Q_p']) if 'multiIV_Q_p' in multi else np.nan
    if multi.empty:
        multi=pd.DataFrame(columns=['gene','roi','multiIV_n','multiIV_p','multiIV_p_q_BH','multiIV_Q_p','multiIV_Q_p_q_BH','multiIV_I2','LOO_n','LOO_all_same_direction','LOO_all_nominal_p05'])
    multi.to_csv(out/'MRI_multiIV_gene_ROI_consistency_summary.csv',index=False)
    loo.to_csv(out/'MRI_multiIV_leave_one_out.csv',index=False)
    master=annotations(hits,ar,multi,annotation,ldflags)
    master.to_csv(out/'Supplement_MRI_DBM_significant_IV_gene_ROI_evidence_FULL.csv',index=False)
    # Annotate all tested MRI results without modifying the original global BH family.
    small=master[['gene','lead_snp','artifact_id','roi','MRI_descriptive_evidence_tier','phenotypes_significant']].copy()
    alltab=alltab.merge(small,on=['gene','lead_snp','artifact_id','roi'],how='left',validate='one_to_one')
    alltab.to_csv(out/'MRI_DBM_ALL_associations_threeUPDRS_evidence_annotated.csv',index=False)
    tiers=master.MRI_descriptive_evidence_tier.value_counts(dropna=False).rename_axis('tier').reset_index(name='n_IV_gene_ROI_hits')
    tiers.to_csv(out/'MRI_descriptive_evidence_tier_counts.csv',index=False)
    manifest={'MRI_only_PP':True,'independent_MRI_replication':False,'clinical_phenotypes':['updrs1','updrs2','updrs3'],
              'MRI_primary_selection':'Original global BH FDR<0.05 frozen before robustness',
              'AR':'SNP dosage x polynomial time participant-cluster reduced-form Wald (AR-type; not classical Anderson-Rubin)',
              'multiIV':'All Stage-I eligible LD-pruned independent IVs for each MRI hit gene, tested at same MRI lag1 per hit ROI, IV-specific GReX standardized to 1 SD. Fixed-effect inverse-variance pooling is descriptive and may neglect correlated estimates within MRI donors.',
              'LOO':'available for >=3 valid independent IVs','LD':'reused original Stage-II cross-gene LD annotation function, r2 >=0.10 flag',
              'tier':'MRI-specific descriptive score; cannot reuse cross-cohort UPDRS A tier; post-selection robustness is not causal proof or an additional candidate selection gate',
              'n_MRI_FDR_hits':len(master),'n_candidate_genes':int(hits.gene.nunique()),'n_aux_IVs':len(cand),'n_aux_errors':len(errors),
              'n_multiiv_gene_ROI':len(multi),'MRI_ROIs':len(rois_needed)}
    (out/'MRI_evidence_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('[MRI EVIDENCE]',out, 'MRI FDR hits',len(master),'multiIV gene/ROI',len(multi),'errors',len(errors))
if __name__=='__main__':main()
