"""PLINK reader regression tests generate temporary binaries; no BED demo shipped."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from gengps.genotypes import read_plink_bed, load_genotypes
from gengps.cli import main


def write_temp_plink(prefix: Path, variants, samples, calls):
    # Raw SNP-major PLINK 1 BED encoding: 00=A1/A1, 01=missing, 10=heterozygote, 11=A2/A2.
    with prefix.with_suffix('.fam').open('w') as f:
        for fid, iid in samples:
            f.write(f'{fid} {iid} 0 0 0 -9\n')
    with prefix.with_suffix('.bim').open('w') as f:
        for i, v in enumerate(variants):
            f.write(f'1 {v} 0 {i + 1} A G\n')
    enc={2:0, -1:1, 1:2, 0:3}
    with prefix.with_suffix('.bed').open('wb') as f:
        f.write(b'\x6c\x1b\x01')
        for row in calls:
            for start in range(0,len(samples),4):
                byte=0
                for k,call in enumerate(row[start:start+4]):
                    byte |= enc[int(call)]<<(2*k)
                f.write(bytes([byte]))


def test_bed_decoding_selective_and_missing(tmp_path):
    prefix=tmp_path/'geno'
    samples=[('F1','A'),('F2','B'),('F3','C'),('F4','D'),('F5','E')]
    write_temp_plink(prefix,['rs0','rs1','rs2'],samples,[[0,1,2,-1,1],[2,2,1,0,-1],[1,0,1,2,2]])
    z=read_plink_bed(prefix.with_suffix('.bed'), ['rs2','rs0'], ['E','A','D'])
    assert z.columns.tolist()==['rs2','rs0']
    assert z.index.tolist()==['E','A','D']
    assert z.to_numpy().tolist()==[[2,1],[1,0],[2,-1]]
    with pytest.raises(ValueError,match='missing from BIM'):
        read_plink_bed(prefix,['not_present'],['A'])
    with pytest.raises(ValueError,match='missing from FAM'):
        read_plink_bed(prefix,['rs0'],['not_present'])
    z2=read_plink_bed(prefix,['rs0'],['F2:B'],sample_id_mode='fid_iid')
    assert z2.iloc[0,0]==1


def test_duplicate_iid_requires_composite(tmp_path):
    prefix=tmp_path/'g'
    write_temp_plink(prefix,['rs0'],[('F1','X'),('F2','X')],[[0,2]])
    with pytest.raises(ValueError,match='Non-unique'):
        read_plink_bed(prefix,['rs0'],['X'])
    assert read_plink_bed(prefix,['rs0'],['F2:X'],sample_id_mode='fid_iid').iat[0,0]==2


def test_ctar_bed_csv_equivalence(tmp_path):
    p=Path(__file__).resolve().parents[1]/'examples'
    cov=pd.read_csv(p/'stage1_covariates.csv')
    geno=pd.read_csv(p/'stage1_genotypes.csv')
    cis=pd.read_csv(p/'stage1_cis_map.csv')
    variants=cis.snpid.tolist()
    samples=[('F',s) for s in geno.participant_id.astype(str)]
    calls=[geno[v].fillna(-1).astype(int).tolist() for v in variants]
    prefix=tmp_path/'genotypes'
    write_temp_plink(prefix,variants,samples,calls)
    out_csv=tmp_path/'csv.csv';out_bed=tmp_path/'bed.csv'
    common=['ctar','--covariates',str(p/'stage1_covariates.csv'),
            '--expression',str(p/'stage1_expression.csv'),
            '--cis-map',str(p/'stage1_cis_map.csv'),
            '--covariates-cols','sex','--rho-points','5']
    main(common+['--genotypes',str(p/'stage1_genotypes.csv'),'--output',str(out_csv)])
    main(common+['--genotypes',str(prefix.with_suffix('.bed')),'--output',str(out_bed)])
    x=pd.read_csv(out_csv).sort_values(['Ensembl_ID','Variant_ID']).reset_index(drop=True)
    y=pd.read_csv(out_bed).sort_values(['Ensembl_ID','Variant_ID']).reset_index(drop=True)
    assert x[['Ensembl_ID','Variant_ID']].equals(y[['Ensembl_ID','Variant_ID']])
    np.testing.assert_allclose(x['Pvalue'],y['Pvalue'],rtol=1e-7,atol=1e-9)
    np.testing.assert_allclose(x['q_bh_global'],y['q_bh_global'],rtol=1e-7,atol=1e-9)
    assert not any((p/'stage1_genotypes.bed').exists() for _ in [0])
