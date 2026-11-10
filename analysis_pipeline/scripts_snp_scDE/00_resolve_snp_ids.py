#!/usr/bin/env python3
"""Resolve target rsIDs to exact PLINK BIM IDs by assembly-aware chromosome,
position and (when available) allele comparison. Never changes the original BIM.
"""
import argparse
import csv
import json
import re
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

SNPS = ('rs2253588', 'rs35162690')
# Published GRCh37 coordinate; this is ONLY a fallback and needs allele audit.
FALLBACK = {'rs2253588': [{'rsid':'rs2253588', 'chrom':'6', 'pos':30899377,
                            'build':'GRCh37','alleles':'','source':'TreeWAS_GRCh37_position_only'}]}

def norm_chr(x):
    return re.sub(r'^chr', '', str(x).strip(), flags=re.I).upper()

def norm_build(x):
    v = str(x).strip().lower()
    return {'37':'GRCh37','grch37':'GRCh37','hg19':'GRCh37',
            '38':'GRCh38','grch38':'GRCh38','hg38':'GRCh38'}.get(v, str(x))

def is_snp_alleles(s):
    return bool(s) and all(re.fullmatch('[ACGT]+', a or '') for a in s.split('/'))

def local_records(path):
    if not path: return []
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(p)
    with p.open() as f:
        rd=csv.DictReader(f, delimiter='\t' if p.suffix.lower() in ('.tsv','.txt') else ',')
        required={'rsid','chrom','pos','build'}
        if not required.issubset(rd.fieldnames or []):
            raise ValueError(f'{p} requires columns {sorted(required)}, optionally ref,alt or alleles')
        records=[]
        for row in rd:
            if row['rsid'] not in SNPS: continue
            r={'rsid':row['rsid'],'chrom':norm_chr(row['chrom']),
               'pos':int(row['pos']), 'build':norm_build(row['build']),
               'alleles':row.get('alleles','') or (('/'.join([row['ref'],row['alt']])) if row.get('ref') and row.get('alt') else ''),
               'source':'user_variant_map'}
            records.append(r)
        return records

def ensembl_records(rsid, server, build):
    url=f'https://{server}/variation/human/{rsid}?content-type=application%2Fjson'
    req=urllib.request.Request(url, headers={'Accept':'application/json','User-Agent':'2sGenGPS-variant-mapper/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=7) as r:
            data=json.loads(r.read().decode('utf-8'))
    except (urllib.error.URLError,TimeoutError, ValueError,OSError) as e:
        print(f'[MAP][WARN] API {build}/{rsid}: {e}',flush=True)
        return []
    out=[]
    for m in data.get('mappings',[]):
        c=norm_chr(m.get('seq_region_name',''))
        pos=m.get('start')
        if c not in set(map(str,range(1,23))) | {'X','Y','MT'} or not pos:
            continue
        if m.get('end')!=pos: # indel or interval, not a simple SNP position
            continue
        assembly=norm_build(m.get('assembly_name', build))
        if assembly not in ('GRCh37','GRCh38'): assembly=build
        out.append({'rsid':rsid,'chrom':c,'pos':int(pos),'build':assembly,
                    'alleles':m.get('allele_string','') or '',
                    'source':f'Ensembl_{build}'})
    return out

def tqtl_reference_queries(ppmi_bim='', pdbp_bim=''):
    """Read exact rsID sites from the project's Stage-I PLINK BIMs.

    These BIMs provide coordinates and allele pairs, NOT explicit genome build.
    Matching target PLINK by position + exact unordered allele pair is required.
    Conflicting PPMI/PDBP positions/alleles are a hard error.
    """
    refs = [('PPMI_tQTL', ppmi_bim), ('PDBP_tQTL', pdbp_bim)]
    candidates = defaultdict(list)
    audit = []
    errors = []
    existing = 0
    for source, filename in refs:
        if not filename:
            continue
        path = Path(filename)
        if not path.is_file():
            audit.append({'rsid': '', 'source': source, 'bim_file': str(path),
                          'found': False, 'chrom': '', 'pos': '', 'a1': '', 'a2': '',
                          'note': 'BIM not available'})
            print(f'[TQTL][WARN] {source} reference BIM not found: {path}', flush=True)
            continue
        existing += 1
        found = defaultdict(list)
        with path.open() as fh:
            for line in fh:
                cols = line.split()
                if len(cols) < 6 or cols[1] not in SNPS:
                    continue
                chrom, rsid, _cm, pos, a1, a2 = cols[:6]
                try:
                    pos = int(pos)
                except ValueError:
                    continue
                item = {'rsid': rsid, 'chrom': norm_chr(chrom), 'pos': pos,
                        'build': 'not_verified', 'alleles': f'{a1.upper()}/{a2.upper()}',
                        'source': source}
                found[rsid].append(item)
        for rsid in SNPS:
            hits = found[rsid]
            if len(hits) > 1:
                errors.append(f'{source} BIM contains {len(hits)} records for {rsid}; exact site is ambiguous')
            if hits:
                for item in hits:
                    candidates[rsid].append(item)
                    audit.append({'rsid': rsid, 'source': source, 'bim_file': str(path),
                                  'found': True, 'chrom': item['chrom'],
                                  'pos': item['pos'], 'a1': item['alleles'].split('/')[0],
                                  'a2': item['alleles'].split('/')[1], 'note': ''})
            else:
                audit.append({'rsid': rsid, 'source': source, 'bim_file': str(path),
                              'found': False, 'chrom': '', 'pos': '', 'a1': '', 'a2': '',
                              'note': 'rsID absent'})
        print(f'[TQTL] {source}: ' + ', '.join(f'{rsid}={len(found[rsid])}' for rsid in SNPS), flush=True)
    if not existing:
        errors.append('Neither Stage-I tQTL BIM was found; verify ROOT/data/genotype/PP and REP')
    queries = {}
    for rsid in SNPS:
        found = candidates[rsid]
        sites = {(x['chrom'], x['pos'], frozenset(x['alleles'].split('/'))) for x in found}
        if len(sites) > 1:
            errors.append(f'{rsid}: conflicting tQTL BIM coordinates/alleles: {found}')
        if not found:
            errors.append(f'{rsid}: rsID absent from available Stage-I tQTL BIMs')
        # A single authoritative tQTL coordinate+allele pair, or nothing on conflict.
        queries[rsid] = [found[0]] if len(sites) == 1 else []
    return queries, audit, errors

def make_queries(snp_map='', offline=False, build='auto'):
    all_local=local_records(snp_map)
    builds=('GRCh37','GRCh38') if build=='auto' else (build,)
    data={}
    for snp in SNPS:
        rs=[r for r in all_local if r['rsid']==snp and r['build'] in builds]
        if not rs and not offline:
            for b in builds:
                host='grch37.rest.ensembl.org' if b=='GRCh37' else 'rest.ensembl.org'
                rs+=ensembl_records(snp,host,b)
        if not rs and snp in FALLBACK:
            rs=[r for r in FALLBACK[snp] if r['build'] in builds]
        # never override explicit local mappings with unknown API data
        unique={}
        for r in rs:
            key=(r['chrom'],r['pos'],r['build'],r['alleles'])
            unique[key]=r
        data[snp]=list(unique.values())
    return data

def matches_alleles(got1,got2,expected):
    """Return True for exact A1/A2 matches; None if no authoritative alleles."""
    if not is_snp_alleles(expected):return None
    expect=set(a.upper() for a in expected.split('/'))
    have={got1.upper(), got2.upper()}
    if len(expect)!=2: return None
    return expect==have

def resolve(bim_path, snp_map='', offline=False, build='auto', allow_position_only=False,
            tqtl_ppmi_bim='', tqtl_pdbp_bim=''):
    if tqtl_ppmi_bim or tqtl_pdbp_bim:
        q, source_audit, source_errors = tqtl_reference_queries(tqtl_ppmi_bim, tqtl_pdbp_bim)
        if snp_map:
            print('[TQTL][INFO] --variant-map ignored: Stage-I BIM is the selected source of coordinates', flush=True)
    else:
        q=make_queries(snp_map,offline,build)
        source_audit, source_errors = [], []
    for snp,values in q.items():
        if not values:
            print(f'[MAP][WARN] No usable reference coordinates for {snp}; check the Stage-I BIM source audit and any PPMI/PDBP conflicts.',flush=True)
    coord_to_candidates=defaultdict(list)
    for snp,rows in q.items():
        for r in rows:
            coord_to_candidates[(r['chrom'],r['pos'])].append(r)
    hits=defaultdict(list)
    scanned=0
    with Path(bim_path).open() as f:
        for line in f:
            fields=line.split()
            if len(fields)<6: continue
            scanned+=1
            chrom,vid,cm,pos,a1,a2=fields[:6]
            cn=norm_chr(chrom)
            try: position=int(pos)
            except ValueError: continue
            if vid in SNPS:
                hits[vid].append({'rsid':vid,'bim_id':vid,'chrom':cn,'pos':position,
                                  'a1':a1,'a2':a2,'build':'BIM_rsID',
                                  'source':'exact_BIM_ID','allele_match':True})
            for r in coord_to_candidates.get((cn,position),()):
                am=matches_alleles(a1,a2,r['alleles'])
                if am is False: continue
                hits[r['rsid']].append({'rsid':r['rsid'],'bim_id':vid,'chrom':cn,'pos':position,
                                         'a1':a1,'a2':a2,'build':r['build'],
                                         'source':r['source'],
                                         'allele_match':am})
    print('[MAP] scanned BIM records:',scanned,flush=True)
    resolved=[]; errors=list(source_errors)
    for snp in SNPS:
        rows=hits[snp]
        # Dedup identical rsid:ID:coordinate assignments; an exact BIM rsID takes precedence
        unique={}
        for row in rows:
            key=(row['bim_id'],row['chrom'],row['pos'])
            if key not in unique or row['source']=='exact_BIM_ID': unique[key]=row
        rows=list(unique.values())
        if len(rows)==1:
            row=rows[0]
            if row['allele_match'] is None:
                print(f'[MAP][WARN] {snp}: physical position matches but no verified allele information; do not treat as allele-confirmed mapping',flush=True)
                if not allow_position_only:
                    errors.append(f'{snp}: missing allele verification. Supply verified ref/alt in --variant-map or explicitly --allow-position-only (not recommended).')
            print(f"[MAP][OK] {snp}: BIM_ID={row['bim_id']} chr={row['chrom']} pos={row['pos']} alleles={row['a1']}/{row['a2']} build={row['build']} match={row['allele_match']}",flush=True)
            resolved.append(row)
        else:
            detail=rows[:8]
            msg=f'{snp}: expected exactly one BIM site, got {len(rows)}. Matches: {detail}'
            print('[MAP][ERROR]',msg,flush=True)
            errors.append(msg)
    if len({r['bim_id'] for r in resolved})!=len(resolved):errors.append('Different rsIDs map to same BIM ID; check mapping')
    return resolved,errors,source_audit

def run_audit(prefix, outdir, variant_map='', offline=False, build='auto', fail=True, allow_position_only=False,
              tqtl_ppmi_bim='', tqtl_pdbp_bim=''):
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    prefix=str(prefix)
    bim=prefix+'.bim'
    if not Path(bim).is_file():raise FileNotFoundError(bim)
    rows,errors,source_audit=resolve(bim,variant_map,offline,build,allow_position_only,
                                     tqtl_ppmi_bim,tqtl_pdbp_bim)
    if source_audit:
        with (out/'tqtl_source_BIM_audit.csv').open('w',newline='') as f:
            cols=['rsid','source','bim_file','found','chrom','pos','a1','a2','note']
            w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(source_audit)
    pd_fn=out/'snp_BIM_resolution_audit.csv'
    cols=['rsid','bim_id','chrom','pos','a1','a2','build','source','allele_match']
    with pd_fn.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
    (out/'snp_BIM_resolution_errors.txt').write_text('\n'.join(errors)+'\n' if errors else '')
    print(f'[MAP] audit: {pd_fn}',flush=True)
    if errors and fail:
        raise ValueError('SNP ID resolution incomplete. Fix --variant-map or --reference-build; see snp_BIM_resolution_errors.txt')
    return rows,errors

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plink-prefix',default='/public/labdata/public_data/AMP_PD_Postmortem_Brain/vcf/merged.100s.new.fixed')
    p.add_argument('--tqtl-ppmi-bim',default='',help='Stage-I PPMI BIM: authoritative rsID coordinate/allele source')
    p.add_argument('--tqtl-pdbp-bim',default='',help='Stage-I PDBP BIM: cross-check / fallback source')
    p.add_argument('--outdir',required=True)
    p.add_argument('--variant-map',default='',help='Optional TSV with rsid chrom pos build ref alt')
    p.add_argument('--reference-build',choices=('auto','GRCh37','GRCh38'),default='auto')
    p.add_argument('--offline',action='store_true')
    p.add_argument('--allow-position-only',action='store_true',help='Explicitly permit unverified coordinate-only hits (not recommended)')
    a=p.parse_args()
    run_audit(a.plink_prefix,a.outdir,a.variant_map,a.offline,a.reference_build,
              allow_position_only=a.allow_position_only,
              tqtl_ppmi_bim=a.tqtl_ppmi_bim,tqtl_pdbp_bim=a.tqtl_pdbp_bim)
if __name__=='__main__':main()
