"""
Example for teQTL analysis using linear regression or cubic polynomial regression
"""
import sys
sys.path.append('./StQTL_2sGenGPS')
import polynomial_regression as poly
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
    path3 = './StQTL/genotype.matrix.csv'
    path4 = './geneloc.csv'
    path5 = './snpsloc.csv'


    cov = pd.read_csv(path1, header=0, index_col=0)
    qt = pd.read_csv(path2, header=0, index_col=0)
    GT = pd.read_csv(path3, header=0, index_col=0)
    genepos = pd.read_csv(path4, header=0, index_col=0)
    snpspos = pd.read_csv(path5, header=0, index_col=0)

    t1 = time.time()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1)))
    n_process = 3
    n_terms1 = 2 # linear regression
    n_terms2 = 4 # Cubic polynomial regression
    cisDist = 1e6

    #teQTL analysis using Linear regression
    out1 = poly.poly_teQTL_main(cov,qt,GT,n_process,n_terms1,cisDist,snpspos,genepos)

    #Dynamic eQTL analysis using Linear regression
    out2 = poly.poly_dynamic_main(cov,qt,GT,n_process,n_terms1,cisDist,snpspos,genepos)

    # teQTL analysis using cubic polynomial regression
    out3 = poly.poly_teQTL_main(cov,qt,GT,n_process,n_terms2,cisDist,snpspos,genepos)

    #Dynamic eQTL analysis using cubic polynomial regression
    out4 = poly.poly_dynamic_main(cov,qt,GT,n_process,n_terms2,cisDist,snpspos,genepos)

    t3 = time.time()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t3)))

