"""
Example for StQTL analysis
"""
import sys
sys.path.append('./StQTL_2sGenGPS')
import StQTL as st
import time
import numpy as np
import pandas as pd




if __name__ == '__main__':
    path1 = './covariates.csv'
    path2 = './longitudinal.gene.expression.csv'
    path3 = './genotype.matrix.csv'
    path4 = './geneloc.csv'
    path5 = './snpsloc.csv'
    cov = pd.read_csv(path1, header=0, index_col=0) #Covariates
    qt = pd.read_csv(path2, header=0, index_col=0) #Longitudinal gene expressions
    GT = pd.read_csv(path3, header=0, index_col=0) #Genotype
    genepos = pd.read_csv(path4, header=0, index_col=0) #Genes location
    snpspos = pd.read_csv(path5, header=0, index_col=0) #Variants location
    t1 = time.time()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t1)))
    """
    'cov': covariates dataframe;
    'y': longitudinal trait level dataframe;
    'genotype': genotype dataframe;
    'time_points': time points of follow-up time;
    'n_process': number of multiprocess of your task;
    'n_terms': number of time term of polynomial regression model;
    'cisDist':distance of cis-eQTL analysis;
    'snpspos': variant location;
    'genepos': gene location;
    """
    time_points = 5
    n_process = 3
    n_terms = 4 # Cubic polynomial regression
    cisDist = 1e6
    out1 = st.teQTL_main(cov,qt,GT,time_points,n_process,n_terms,cisDist,snpspos,genepos) #teQTL analysis
    print(out1)
    out2 = st.dynamic_eQTL_main(cov,qt,GT,time_points,n_process,n_terms,cisDist,snpspos,genepos) #Dynamic eQTL analysis
    print(out2)
    t3 = time.time()
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t3)))

    
    
