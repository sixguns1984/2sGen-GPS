#!/usr/bin/env python3
"""Normalize MAP/ADNI MMSE phenotype, harmonize IV IDs and genotypes without touching original bfiles."""
import argparse,json,re,subprocess
from pathlib import Path
import numpy as np
import pandas as pd

def read_bim(prefix, *, snp_filter=None, key_filter=None, chunksize=250_000):
    """Read just candidate variants from a potentially multi-million-SNP source BIM.

    Filtering during streaming is essential for original MAP TOPMed and ADNI WGS:
    avoid materializing their full BIM files in RAM.
    """
    p=Path(str(prefix)+'.bim')
    if not p.exists(): raise FileNotFoundError(p)
    snps=set(map(str,snp_filter)) if snp_filter is not None else None
    keys=set(map(str,key_filter)) if key_filter is not None else None
    frames=[]
    for chunk in pd.read_csv(p,sep=r'\s+',header=None,
                             names=['chr','snp','cm','bp','a1','a2'],
                             dtype=str,keep_default_na=False,chunksize=chunksize):
        if snps is not None: chunk=chunk[chunk.snp.isin(snps)].copy()
        if chunk.empty: continue
        chunk['key']=chunk['chr'].str.replace('chr','',case=False,regex=False)+':'+chunk['bp']
        if keys is not None: chunk=chunk[chunk.key.isin(keys)].copy()
        if not chunk.empty: frames.append(chunk)
    if not frames:
        return pd.DataFrame(columns=['chr','snp','cm','bp','a1','a2','key'],dtype=str)
    return pd.concat(frames,ignore_index=True)

def normalize_id(value,cohort):
    s=str(value).strip()
    if cohort=='MAP':
        match=re.search(r'MAP\s*(\d+)',s,flags=re.I)
        return match.group(1) if match else (s if re.fullmatch(r'\d+',s) else '')
    # ADNI WGS FAM IDs may contain dataset prefixes/suffixes around PTID.
    # Extract a canonical PTID only when its format is explicitly present.
    match=re.search(r'(?<!\d)(\d{3}_S_\d{4})(?!\d)',s,flags=re.I)
    return match.group(1).upper() if match else s

def prepare_cov(src,cohort):
    x=pd.read_csv(src,low_memory=False).loc[:,lambda d:~d.columns.str.startswith('Unnamed:')].copy()
    if cohort=='MAP':
        rename={'projid':'participant_id','fu_year':'time','cts_mmse30':'mmse','age':'age','sex':'sex','eduyrs':'eduyrs'}
    else:
        rename={'PTID':'participant_id','time':'time','MMSE':'mmse','AGE':'age','PTGENDER':'sex','PTEDUCAT':'eduyrs'}
    for c in rename:
        if c not in x.columns:raise ValueError(f'{cohort}: missing {c}')
    x=x.rename(columns=rename)
    x['participant_id']=x.participant_id.map(lambda z:normalize_id(z,cohort))
    for c in ['time','mmse','age','eduyrs']:x[c]=pd.to_numeric(x[c],errors='coerce')
    x['sex']=x.sex.astype(str).str.strip().str.lower().map({'female':0.,'f':0.,'0':0.,'male':1.,'m':1.,'1':1.,'2':0.})
    x['cohort']=cohort
    # AGE/age is baseline age in the supplied tables. Current age is calculated by Stage2 as age + time.
    x=x.dropna(subset=['participant_id','time','mmse','age','sex','eduyrs'])
    x=x[(x.participant_id!='')&(x.time>=0)&x.mmse.between(0,30)].copy()
    cols=['cohort','participant_id','time','mmse','age','sex','eduyrs']
    x=x[cols].sort_values(['participant_id','time'])
    duplicates=x.duplicated(['cohort','participant_id','time'],keep=False)
    if duplicates.any():
        conflict=x.loc[duplicates].groupby(['cohort','participant_id','time']).mmse.nunique()
        if (conflict>1).any():raise ValueError(f'{cohort}: conflicting MMSE for duplicate person-time rows')
        x=x.drop_duplicates(['cohort','participant_id','time'])
    return x.reset_index(drop=True)

def read_fam(prefix,cohort,expected_ids=None):
    """Match a phenotype ID against both FID and IID (original MAP TOPMed may encode the MAP ID in FID)."""
    x=pd.read_csv(str(prefix)+'.fam',sep=r'\s+',header=None,dtype=str,
                  names=['FID','IID','father','mother','sex','phenotype'])
    iid_id=x.IID.map(lambda y:normalize_id(y,cohort))
    fid_id=x.FID.map(lambda y:normalize_id(y,cohort))
    if expected_ids is not None:
        valid=set(map(str,expected_ids))
        # Prefer a matching IID; otherwise use an FID match. Nonmatching rows
        # remain unselected (they are excluded by subsequent --keep).
        x['participant_id']=iid_id.where(iid_id.isin(valid),fid_id.where(fid_id.isin(valid),''))
    else:
        x['participant_id']=iid_id.where(iid_id.ne(''),fid_id)
    x=x[x.participant_id!=''].copy()
    if x.participant_id.duplicated().any():
        raise ValueError(f'{cohort} FAM contains repeated normalized participant IDs')
    return x

DNA=frozenset('ACGT')
COMPLEMENT={'A':'T','T':'A','C':'G','G':'C'}
PALINDROMES={frozenset(('A','T')),frozenset(('C','G'))}
VALID_STATUSES={'allele_aligned','strand_flipped_aligned'}

def normalize_chr(x):
    return str(x).strip().upper().removeprefix('CHR').lstrip('0') or '0'

def harmonize(candidates,source,target,*,mode='coordinate'):
    """MAP: G38 chromosome+BP; ADNI: stable rsID across differing genome builds.

    All SNPs are compared against frozen Stage1 alleles; exact rsID does NOT
    override a chromosome mismatch, ambiguous alleles, or duplicate IDs.
    Complement strands are explicitly flipped with plink1.9 --flip later.
    """
    source=source[source.snp.isin(set(candidates.lead_snp.astype(str)))].copy()
    if source.snp.duplicated().any():
        raise ValueError('Source tQTL reference BIM contains duplicate rsIDs')
    idx={}
    for t in target.itertuples(index=False):
        key=t.key if mode=='coordinate' else t.snp
        idx.setdefault(key,[]).append(t)
    rec=[]
    for v in source.itertuples(index=False):
        matches=idx.get(v.key if mode=='coordinate' else v.snp,[])
        status='unmapped';old='';target_chr='';target_bp=''
        if len(matches)>1:
            status='multiple_variants_same_position' if mode=='coordinate' else 'duplicate_target_rsid'
        elif len(matches)==1:
            t=matches[0];old=str(t.snp);target_chr=str(t.chr);target_bp=str(t.bp)
            sa={str(v.a1).upper(),str(v.a2).upper()}
            ta={str(t.a1).upper(),str(t.a2).upper()}
            if normalize_chr(v.chr)!=normalize_chr(t.chr):
                status='chromosome_mismatch'
            elif len(sa)!=2 or len(ta)!=2 or not sa.issubset(DNA) or not ta.issubset(DNA):
                status='non_biallelic_or_non_ACGT'
            elif frozenset(sa) in PALINDROMES or frozenset(ta) in PALINDROMES:
                status='palindromic_excluded'
            elif ta==sa:
                status='allele_aligned'
            elif {COMPLEMENT[a] for a in ta}==sa:
                status='strand_flipped_aligned'
            else:
                status='allele_mismatch'
        rec.append(dict(lead_snp=str(v.snp),source_key=str(v.key),source_chr=str(v.chr),
                        source_bp=str(v.bp),source_a1=str(v.a1),source_a2=str(v.a2),
                        target_id=old,target_chr=target_chr,target_bp=target_bp,
                        target_key=(normalize_chr(target_chr)+':'+target_bp) if target_chr else '',
                        mapping_mode=mode,status=status))
    audit=pd.DataFrame(rec)
    good=audit[audit.status.isin(VALID_STATUSES)]
    repeated=set(good.loc[good.target_id.duplicated(keep=False),'target_id'])
    if repeated:
        audit.loc[audit.status.isin(VALID_STATUSES)&audit.target_id.isin(repeated),'status']='duplicate_target_variant_id'
    return audit[audit.status.isin(VALID_STATUSES)].copy(),audit

def call(cmd,log):
    with open(log,'w') as f:r=subprocess.run([str(c) for c in cmd],stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:
        lines=Path(log).read_text(errors='replace').splitlines()
        tail='\n'.join(lines[-35:])
        raise RuntimeError(f'PLINK failed ({r.returncode}) command: {cmd}\n--- PLINK LOG TAIL ---\n{tail}\n--- END LOG ---\nFull log: {log}')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True);ap.add_argument('--map-cov',required=True);ap.add_argument('--adni-cov',required=True)
    ap.add_argument('--map-bfile',required=True);ap.add_argument('--adni-bfile',required=True)
    ap.add_argument('--source-bim',required=True,help='AMP-PD/Stage1 genotype BIM on the SAME genome build')
    ap.add_argument('--candidate-file',required=True);ap.add_argument('--plink',default='plink1.9');ap.add_argument('--out-dir',default='')
    a=ap.parse_args();root=Path(a.root);out=Path(a.out_dir) if a.out_dir else root/'single_iv_causal'/'ad_mmse'/'prep';out.mkdir(parents=True,exist_ok=True)
    candidates=pd.read_csv(a.candidate_file,low_memory=False)
    for c in ['gene','lead_snp','artifact_id','basis_json']:
        if c not in candidates:raise ValueError(f'candidate file lacks {c}; use frozen full Stage1 eligible candidates')
    if not any(c.startswith('beta_g_') for c in candidates):raise ValueError('Missing frozen Stage1 beta_g_* coefficients')
    source_prefix=a.source_bim[:-4] if a.source_bim.endswith('.bim') else a.source_bim
    source=read_bim(source_prefix, snp_filter=set(candidates.lead_snp.dropna().astype(str)))
    if source.empty:raise RuntimeError('No frozen Stage1 candidate IV SNPs found in AMP-PD reference BIM')
    if source.snp.duplicated().any():raise RuntimeError('Duplicate reference SNP IDs; cannot safely map genotype orientation')
    print(f'[REFERENCE] candidate_pairs={len(candidates)} candidate_IVs={candidates.lead_snp.nunique()} '
          f'AMP-PD_reference_IVs={len(source)}',flush=True)
    all_audit=[];cohorts=[]
    for cohort,cp,bf in [('MAP',a.map_cov,a.map_bfile),('ADNI',a.adni_cov,a.adni_bfile)]:
        c=prepare_cov(cp,cohort);fam=read_fam(bf,cohort,expected_ids=set(c.participant_id))
        overlap=set(c.participant_id)&set(fam.participant_id)
        c=c[c.participant_id.isin(overlap)].copy()
        if c.empty:raise RuntimeError(f'{cohort}: no MMSE participants match genotype FAM')
        c.to_csv(out/f'{cohort}_mmse_cov.csv',index=False)
        matched=fam[fam.participant_id.isin(overlap)].copy()
        matched[['FID','IID','participant_id']].to_csv(out/f'{cohort}_sample_id_audit.csv',index=False)
        # Keep the ORIGINAL FAM FID/IID: numeric MAP IDs are not in the original .fam.
        # Without --keep, --update-ids changes only matched subjects and leaves all
        # other genotype samples in the output, causing a false FAM/cov mismatch.
        keep_file=out/f'{cohort}_keep_samples.txt'
        matched[['FID','IID']].to_csv(keep_file,sep=' ',index=False,header=False)
        if len(matched)!=matched.participant_id.nunique():
            raise RuntimeError(f'{cohort}: normalized genotype participant IDs are not unique')
        print(f'[SAMPLE ID] {cohort}: original FAM={len(fam)}, MMSE cov people={c.participant_id.nunique()}, matched keep={len(matched)}',flush=True)
        # PLINK update-ids: old_FID old_IID new_FID new_IID
        ids=matched[['FID','IID']].copy();ids['new_FID']=matched.participant_id;ids['new_IID']=matched.participant_id
        ids.to_csv(out/f'{cohort}_update_ids.txt',sep=' ',index=False,header=False)
        # MAP TOPMed is on the AMP-PD Stage1 coordinate system; ADNI 808 WGS
        # commonly uses GRCh37. Never match ADNI by GRCh38 BP alone.
        mode='coordinate' if cohort=='MAP' else 'rsid'
        target=(read_bim(bf,key_filter=set(source.key)) if mode=='coordinate'
                else read_bim(bf,snp_filter=set(source.snp)))
        print(f'[BIM MATCH] {cohort}: original={bf}.bim, mode={mode}, candidate_target_rows={len(target)}',flush=True)
        valid,audit=harmonize(candidates,source,target,mode=mode)
        audit['cohort']=cohort;all_audit.append(audit)
        print(f'[ALLELE MATCH] {cohort}: {audit.status.value_counts().to_dict()}; kept={len(valid)}',flush=True)
        if cohort=='ADNI':
            shifted=int((valid.source_bp!=valid.target_bp).sum())
            print(f'[BUILD CHECK] ADNI matched rsIDs with different BP vs Stage1 GRCh38: '
                  f'{shifted}/{len(valid)}; original WGS positions will be retained in audit',flush=True)
        if valid.empty:raise RuntimeError(f'{cohort}: no variants pass coordinate+allele matching; check genome build and BIM')
        valid[['target_id','lead_snp']].to_csv(out/f'{cohort}_update_names.txt',sep=' ',index=False,header=False)
        valid[['target_id']].to_csv(out/f'{cohort}_extract.txt',sep=' ',index=False,header=False)
        # PLINK --a1-allele: rsID desired A1 (the source/Stage1 allele), after update-name.
        valid[['lead_snp','source_a1']].to_csv(out/f'{cohort}_source_a1.txt',sep=' ',index=False,header=False)
        # PLINK 1.9 requires --update-name in its own invocation; do NOT combine
        # it with --update-ids or another update command. Avoid PLINK implicit
        # allele reordering at every step.
        # IMPORTANT: PLINK updates variant names before applying --extract.
        # Split into separate invocations so --extract uses original chr:pos IDs.
        extracted=out/f'{cohort}_extracted_original_ids'
        call([a.plink,'--bfile',bf,'--extract',out/f'{cohort}_extract.txt',
              '--keep',keep_file,
              '--keep-allele-order','--make-bed','--out',extracted],
             out/f'{cohort}_extract_plink.log')
        # Check participant selection BEFORE any ID rewrite.  This also covers ADNI.
        extracted_fam=read_fam(extracted,cohort,expected_ids=set(matched.participant_id))
        expected_ids=set(matched.participant_id)
        if len(extracted_fam)!=len(expected_ids) or set(extracted_fam.participant_id)!=expected_ids:
            missing=sorted(expected_ids-set(extracted_fam.participant_id))[:10]
            extra=sorted(set(extracted_fam.participant_id)-expected_ids)[:10]
            raise RuntimeError(f'{cohort}: PLINK --keep failed to retain the intended MMSE subjects: '
                               f'expected={len(expected_ids)} actual={len(extracted_fam)} '
                               f'missing(example)={missing} extra(example)={extra}')
        extracted_bim=read_bim(extracted)
        if len(extracted_bim)!=len(valid) or set(extracted_bim.snp)!=set(valid.target_id):
            raise RuntimeError(f'{cohort}: extracted variant count/IDs do not match mapping: {len(extracted_bim)} vs {len(valid)}')
        # For ADNI original BIM already has rsIDs; skip a no-op --update-name.
        if (valid.target_id==valid.lead_snp).all():
            renamed=extracted
            print(f'[RSID] {cohort}: target IDs already match frozen Stage1 rsIDs; rename skipped',flush=True)
        else:
            renamed=out/f'{cohort}_renamed_variants'
            call([a.plink,'--bfile',extracted,'--update-name',out/f'{cohort}_update_names.txt',
                  '--keep-allele-order','--make-bed','--out',renamed],
                 out/f'{cohort}_rename_plink.log')
        renamed_bim=read_bim(renamed)
        if len(renamed_bim)!=len(valid) or set(renamed_bim.snp)!=set(valid.lead_snp):
            raise RuntimeError(f'{cohort}: renamed variant count/IDs do not match Stage1 rsID mapping')
        # Nonpalindromic strand-complement variants require an actual PLINK flip;
        # A1 selection alone cannot convert nucleotides across strands.
        n_flip=int(valid.status.eq('strand_flipped_aligned').sum())
        prepared=renamed
        if n_flip:
            fliplist=out/f'{cohort}_strand_flip_rsIDs.txt'
            valid.loc[valid.status.eq('strand_flipped_aligned'),'lead_snp'].to_csv(
                fliplist,sep=' ',index=False,header=False)
            prepared=out/f'{cohort}_strand_aligned'
            call([a.plink,'--bfile',renamed,'--flip',fliplist,
                  '--keep-allele-order','--make-bed','--out',prepared],
                 out/f'{cohort}_flip_plink.log')
        print(f'[STRAND] {cohort}: nonpalindromic variants complemented={n_flip}',flush=True)
        # Stage1's actual genotype dosage allele must be confirmed independently;
        # here we align to AMP-PD Stage1 reference BIM A1 as in prior versions.
        adjusted=out/f'{cohort}_harmonized_A1'
        call([a.plink,'--bfile',prepared,
              '--update-ids',out/f'{cohort}_update_ids.txt',
              '--a1-allele',out/f'{cohort}_source_a1.txt','2','1',
              '--keep-allele-order','--make-bed','--out',adjusted],
             out/f'{cohort}_allele_plink.log')
        output_bim=read_bim(adjusted)
        bad=valid.merge(output_bim[['snp','chr','a1','a2']],left_on='lead_snp',right_on='snp',how='left',validate='one_to_one')
        if not ((bad.source_a1.str.upper()==bad.a1.str.upper()) &
                (bad.source_a2.str.upper()==bad.a2.str.upper()) &
                (bad.source_chr.map(normalize_chr)==bad.chr.map(normalize_chr))).all():
            raise RuntimeError(f'{cohort}: Stage1 A1/A2 or chromosome mismatch after PLINK; do not use predictions')
        # ADNI variant coordinates may be GRCh37. AFTER matching unique rsID,
        # chromosome and both alleles, annotate a DERIVED BIM with the Stage1
        # GRCh38 coordinates so the two study bfiles merge consistently.
        # This does not alter original WGS data or genotype dosages.
        if cohort=='ADNI':
            src_pos=dict(zip(valid.lead_snp,valid.source_bp))
            before=output_bim[['snp','bp']].copy()
            output_bim['bp']=output_bim.snp.map(src_pos).astype(str)
            if output_bim.bp.isna().any():
                raise RuntimeError('ADNI reference coordinate mapping incomplete')
            output_bim[['chr','snp','cm','bp','a1','a2']].to_csv(
                str(adjusted)+'.bim',sep='\t',index=False,header=False)
            print(f'[ANNOTATE] ADNI derived BIM: {sum(before.bp!=output_bim.bp)} matched rsID BP coordinates '
                  'reannotated to Stage1 reference; original WGS BIM untouched',flush=True)
            output_bim=read_bim(adjusted)
            if not (output_bim.snp.map(src_pos).astype(str)==output_bim.bp.astype(str)).all():
                raise RuntimeError('ADNI derived BIM annotation verification failed')
        actual_fam=read_fam(adjusted,cohort,expected_ids=set(matched.participant_id))
        expected_ids=set(matched.participant_id)
        if (len(actual_fam)!=len(expected_ids) or
            set(actual_fam.IID)!=expected_ids or set(actual_fam.FID)!=expected_ids):
            unexpected=sorted(set(actual_fam.IID)-expected_ids)[:10]
            missing=sorted(expected_ids-set(actual_fam.IID))[:10]
            raise RuntimeError(f'{cohort}: harmonized FAM ID mismatch: '
                               f'expected={len(expected_ids)} actual={len(actual_fam)} '
                               f'unexpected IID (example)={unexpected} missing IID (example)={missing}; '
                               f'inspect {cohort}_sample_id_audit.csv and {cohort}_update_ids.txt')
        if output_bim.snp.duplicated().any() or set(output_bim.snp)!=set(valid.lead_snp):
            raise RuntimeError(f'{cohort}: harmonized BIM variant IDs do not match expected Stage1 rsIDs')
        cohorts.append({'cohort':cohort,'observations':len(c),'participants':c.participant_id.nunique(),'variants_harmonized':len(valid),'bfile':str(adjusted)})
        print('[PREP]',cohort,cohorts[-1],flush=True)
    audit=pd.concat(all_audit,ignore_index=True);audit.to_csv(out/'SNP_harmonization_audit.csv',index=False)
    keep_by_cohort={c:set(audit[(audit.cohort==c)&audit.status.isin(VALID_STATUSES)].lead_snp) for c in ['MAP','ADNI']}
    shared=keep_by_cohort['MAP']&keep_by_cohort['ADNI'];candidate=candidates[candidates.lead_snp.isin(shared)].copy()
    if candidate.empty:raise RuntimeError('No shared variants after allele matching')
    candidate.to_csv(out/'AD_MMSE_shared_frozen_candidates.csv',index=False)
    cov=pd.concat([pd.read_csv(out/f'{cohort}_mmse_cov.csv') for cohort in ['MAP','ADNI']],ignore_index=True)
    cov.to_csv(out/'AD_MMSE_phenotype_cov.csv',index=False)
    (out/'preparation_manifest.json').write_text(json.dumps({'cohorts':cohorts,'n_candidate_pairs':len(candidate),'n_candidate_variants':candidate.lead_snp.nunique(),'n_unmapped_or_excluded':int((audit.status!='allele_aligned').sum()),'warning':'ADNI matched by rsID+chr+alleles across potentially different builds, with derived BIM reannotated to frozen Stage1 positions; verify Stage1 dosage allele is reference A1'},indent=2))
    print('[READY]',len(candidate),'IV-gene pairs shared across MAP and ADNI',flush=True)

if __name__=='__main__':main()
