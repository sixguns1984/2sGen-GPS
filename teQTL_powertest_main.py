"""
Example for power test of significant teQTLs
"""
import sys
sys.path.append('./StQTL_2sGenGPS')
import teQTL_powertest as pwr
import time
import pandas as pd

if __name__ == '__main__':
    path1 = './covariates.csv'
    path2 = './longitudinal.gene.expression.csv'
    path3 = './genotype.matrix.csv'
    path4 = './teQTLs_pwr_test.csv'

    cov = pd.read_csv(path1, header=0, index_col=0)
    qt = pd.read_csv(path2, header=0, index_col=0)
    GT = pd.read_csv(path3, header=0, index_col=0)
    teQTLs = pd.read_csv(path4, header=0, index_col=0)#teQTL IDs for power test

    t1 = time.time()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1)))

    out = pwr.pwr_test_main(cov,qt,GT,teQTLs,5,3,4,50,5)

    t3 = time.time()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t3)))

