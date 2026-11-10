from pathlib import Path
import pandas as pd
from gengps.cli import main

def test_small_ctar(tmp_path):
    p=Path(__file__).resolve().parents[1]/'examples'
    main(['ctar','--covariates',str(p/'stage1_covariates.csv'),
          '--expression',str(p/'stage1_expression.csv'),
          '--genotypes',str(p/'stage1_genotypes.csv'),
          '--cis-map',str(p/'stage1_cis_map.csv'),
          '--covariates-cols','sex','--rho-points','5',
          '--output',str(tmp_path/'ctar.csv')])
    z=pd.read_csv(tmp_path/'ctar.csv')
    assert len(z)==2
    assert z.q_bh_global.notna().all()
    assert z.Pvalue.between(0,1).all()

def test_small_arx(tmp_path):
    p=Path(__file__).resolve().parents[1]/'examples'
    main(['stage2','--input',str(p/'stage2_visits.csv'),
          '--outcome','UPDRS_III','--max-lag','3',
          '--age-cubic','--evidence',
          '--output',str(tmp_path/'stage2.csv')])
    z=pd.read_csv(tmp_path/'stage2.csv')
    assert len(z)==3
    assert 'q_bh_global' in z
    assert (tmp_path/'stage2_reduced_form_evidence.csv').exists()
