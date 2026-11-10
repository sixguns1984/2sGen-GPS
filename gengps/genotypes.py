"""Genotype input adapters for Stage-I CTAR: donor-wide CSV or PLINK 1 BED.

PLINK 1 BED uses SNP-major layout, with two bits per sample. We count the
allele in BIM column 5 (A1): 00 -> 2, 10 -> 1, 11 -> 0 and 01 -> -1.
Missing calls use the original manuscript's explicit -1 hard-call category.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def _plink_prefix(path):
    p = Path(path)
    return p.with_suffix('') if p.suffix.lower() in {'.bed', '.bim', '.fam'} else p


def read_plink_bed(path, snp_ids, participant_ids, sample_id_mode='iid'):
    """Read only selected SNPs from SNP-major PLINK 1 BED/BIM/FAM files.

    Returns a participant-indexed DataFrame with hard-call A1 dosages in
    {0,1,2,-1}. Requested samples and SNPs must be uniquely resolvable.
    ``sample_id_mode='iid'`` uses FAM IID; ``'fid_iid'`` uses FID:IID.
    """
    prefix = _plink_prefix(path)
    bed = prefix.with_suffix('.bed')
    bim = prefix.with_suffix('.bim')
    fam = prefix.with_suffix('.fam')
    for p in (bed, bim, fam):
        if not p.is_file():
            raise FileNotFoundError(f'PLINK trio incomplete: missing {p}')
    fam_df = pd.read_csv(fam, sep=r'\s+', header=None, dtype=str)
    bim_df = pd.read_csv(bim, sep=r'\s+', header=None, dtype=str)
    if fam_df.shape[1] < 6 or bim_df.shape[1] < 6:
        raise ValueError('PLINK FAM and BIM must contain at least six columns')
    if sample_id_mode == 'iid':
        sid = fam_df.iloc[:, 1].astype(str)
    elif sample_id_mode == 'fid_iid':
        sid = fam_df.iloc[:, 0].astype(str) + ':' + fam_df.iloc[:, 1].astype(str)
    else:
        raise ValueError('sample_id_mode must be iid or fid_iid')
    if sid.duplicated().any():
        examples = sid[sid.duplicated(keep=False)].unique()[:5].tolist()
        raise ValueError(f'Non-unique PLINK participant IDs ({sample_id_mode}): {examples}. Use --plink-sample-id fid_iid when appropriate.')
    snps = bim_df.iloc[:, 1].astype(str)
    needed_snps = list(dict.fromkeys(map(str, snp_ids)))
    if not needed_snps:
        raise ValueError('Empty cis SNP selection')
    selected_snps = snps.isin(needed_snps)
    duplicated_selected = snps[selected_snps].duplicated(keep=False)
    if duplicated_selected.any():
        raise ValueError('Duplicate selected BIM SNP IDs: ' + ', '.join(snps[selected_snps][duplicated_selected].unique()[:5]))
    snp_to_row = {s: i for i, s in enumerate(snps) if selected_snps.iat[i]}
    missing_snps = [s for s in needed_snps if s not in snp_to_row]
    if missing_snps:
        raise ValueError(f'{len(missing_snps)} cis-map SNPs are missing from BIM: {missing_snps[:8]}')
    needed_ids = list(dict.fromkeys(map(str, participant_ids)))
    participant_index = pd.Index(sid.tolist())
    sample_indices = participant_index.get_indexer(needed_ids)
    if (sample_indices < 0).any():
        missing_ids = [needed_ids[i] for i in np.flatnonzero(sample_indices < 0)[:8]]
        raise ValueError(f'{int((sample_indices < 0).sum())} expression donors missing from FAM: {missing_ids}')
    n_samples = len(fam_df)
    bytes_per_variant = (n_samples + 3) // 4
    expected_size = 3 + bytes_per_variant * len(bim_df)
    actual_size = bed.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f'BED size mismatch: expected {expected_size} bytes for {n_samples} donors and {len(bim_df)} variants, found {actual_size}')
    out = np.empty((len(needed_ids), len(needed_snps)), dtype=np.int8)
    decode = np.array([2, -1, 1, 0], dtype=np.int8)
    with bed.open('rb') as fh:
        if fh.read(3) != b'\x6c\x1b\x01':
            raise ValueError('Unsupported BED header: expected PLINK 1 SNP-major .bed (6c 1b 01)')
        for j, variant in enumerate(needed_snps):
            fh.seek(3 + snp_to_row[variant] * bytes_per_variant)
            raw = fh.read(bytes_per_variant)
            if len(raw) != bytes_per_variant:
                raise ValueError(f'Truncated BED genotype data for {variant}')
            packed = np.frombuffer(raw, dtype=np.uint8)
            code = (packed[sample_indices // 4] >> (2 * (sample_indices % 4))) & 3
            out[:, j] = decode[code]
    return pd.DataFrame(out, index=pd.Index(needed_ids, name='participant_id'), columns=needed_snps)


def load_genotypes(path, cis_map, participant_ids, id_col='participant_id', plink_sample_id='iid'):
    """Load hard-call genotypes from CSV/Parquet or PLINK .bed/.bim/.fam.

    For CSV, retain the existing participant_id and SNP-column convention.
    For BED, retrieve only SNPs enumerated in cis_map; counts refer to BIM A1.
    """
    p = Path(path)
    if p.suffix.lower() in {'.bed', '.bim', '.fam'} or (not p.suffix and p.with_suffix('.bed').exists()):
        return read_plink_bed(p, cis_map['snpid'].astype(str), participant_ids, plink_sample_id)
    if p.suffix.lower() == '.csv':
        frame = pd.read_csv(p)
    elif p.suffix.lower() == '.parquet':
        frame = pd.read_parquet(p)
    else:
        raise ValueError('Genotype input must be a .csv/.parquet table, a .bed path or a PLINK prefix')
    if id_col not in frame.columns:
        raise ValueError(f'Genotype table must contain {id_col}')
    frame[id_col] = frame[id_col].astype(str)
    if frame[id_col].duplicated().any():
        raise ValueError(f'Duplicate participant IDs in genotype table: {id_col}')
    cols = list(dict.fromkeys(cis_map['snpid'].astype(str)))
    missing = [s for s in cols if s not in frame.columns]
    if missing:
        raise ValueError(f'{len(missing)} cis-map SNPs missing from genotype table: {missing[:8]}')
    index = frame.set_index(id_col)
    ids = list(dict.fromkeys(map(str, participant_ids)))
    absent = [s for s in ids if s not in index.index]
    if absent:
        raise ValueError(f'{len(absent)} expression donors missing from genotype table: {absent[:8]}')
    g = index.reindex(ids)[cols].apply(pd.to_numeric, errors='raise')
    if not np.isin(g.to_numpy(float)[np.isfinite(g.to_numpy(float))], [0, 1, 2, -1]).all():
        raise ValueError('CTAR genotype table must contain hard calls 0,1,2 or -1 (missing)')
    return g.fillna(-1).astype(np.int8)
