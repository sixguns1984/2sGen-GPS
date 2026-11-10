#!/usr/bin/env python3
"""Harmonized carrier indicator for postmortem VCF ALT alleles; PLINK A1 may be REF or ALT."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

TARGETS = ('rs2253588','rs35162690')
EXPECTED = {'rs2253588': ('6', '30931600', 'C', 'G'),
            'rs35162690': ('7', '43178314', 'T', 'TC')}

def allele_definition(json_path, rsid):
    doc=json.loads(Path(json_path).read_text())
    if rsid not in doc: raise ValueError(f'{rsid} missing in allele audit {json_path}')
    record=doc[rsid]
    bid=str(record.get('bim_id',''))
    counted=str(record.get('counted_allele','')).upper()
    parts=bid.split(':')
    if len(parts)!=4:raise ValueError(f'{rsid}: cannot infer VCF REF/ALT from BIM ID {bid!r}; refusing to guess carrier allele')
    chrom,pos,ref,alt=parts
    chrom=chrom.removeprefix('chr'); ref=ref.upper();alt=alt.upper()
    exp=EXPECTED.get(rsid)
    if exp and (chrom,pos,ref,alt) != exp:
        raise ValueError(f'{rsid} expected {exp} in verified BIM, found {(chrom,pos,ref,alt)}; review assembly and alleles')
    if not ref or not alt or ref==alt or counted not in (ref,alt):
        raise ValueError(f'{rsid}: counted={counted}, REF={ref}, ALT={alt}; cannot orient genotype')
    return {'bim_id': bid,'chrom':chrom,'pos':pos,'ref':ref,'alt':alt,
            'plink_counted_allele':counted,'flip_to_alt': counted==ref}

def carrier_from_dosage(dosage, cfg):
    raw=pd.to_numeric(pd.Series(dosage),errors='coerce').to_numpy(dtype=float)
    is_valid=np.isfinite(raw)&(raw>=-1e-6)&(raw<=2+1e-6)&np.isclose(raw,np.rint(raw),atol=1e-6)
    alt=np.full(len(raw),np.nan)
    alt[is_valid]=2-raw[is_valid] if cfg['flip_to_alt'] else raw[is_valid]
    carrier=np.full(len(raw),np.nan)
    carrier[is_valid]=(alt[is_valid]>=1).astype(float)
    return alt,carrier
