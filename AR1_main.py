"""
Example for teQTL analysis using first order Auto Regression
"""
import sys
sys.path.append('./StQTL_2sGenGPS')
import AR1 as ar
import time
import pandas as pd
from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

if __name__ == '__main__':
    path1 = './covariates.csv'
    path2 = './longitudinal.gene.expression.csv'
    path3 = './genotype.matrix.csv'
    path4 = './StQTL/geneloc.csv'
    path5 = './snpsloc.csv'

    cov = pd.read_csv(path1, header=0, index_col=0)
    qt = pd.read_csv(path2, header=0, index_col=0)
    GT = pd.read_csv(path3, header=0, index_col=0)
    genepos = pd.read_csv(path4, header=0, index_col=0)
    snpspos = pd.read_csv(path5, header=0, index_col=0)

    t1 = time.time()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1)))

    n_process = 3
    cisDist = 1e6

    out = ar.ar1_main(cov,qt,GT,n_process,cisDist,snpspos,genepos)

    print(out)
    t3 = time.time()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t3)))





