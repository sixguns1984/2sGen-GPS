"""
Example for two stage Genetic Granger Temporal Causality Study
"""
import prepare as pp
import pandas as pd
import sys
sys.path.append('./StQTL_2sGenGPS')
import twosGen_GPS as gps


if __name__ == '__main__':
    path1 = './genotype.matrix.csv'
    path2 = './y_cov.csv'
    path3 = './teQTL.coefs.csv'
    genotype = pd.read_csv(path1, header=0, index_col=0) #Genotype
    y_cov = pd.read_csv(path2, header=0, index_col=0) #Dataframe including dependent variable and covariates
    teQTL_coefs = pd.read_csv(path3, header=0, index_col=0) #Dataframe of coefficients of teQTL model for longitudinal gene expression prediction
    n_terms = 4
    trait_pred_teQTLs = pp.trait_pred_teQTL(genotype,y_cov,teQTL_coefs,n_terms)#Genetic prediction of longitudinal gene expression
    laglen = 5 # lag order

    #Temporal causality analysis between genetic prediting gene expression and complex phenotype
    #cov_fixed:fixed covariates ID;cov_long:dynamic covariates
    cov_fixed1 = ['participant_id', 'X1', 'X2', 'X4']
    cov_long = ['updrs_score', 'X3']
    res = gps.vector_AR_main(laglen, trait_pred_teQTLs, y_cov,cov_fixed1,cov_long)

    #Co-integration test
    cov_fixed2 =['time', 'participant_id', 'X1', 'X2', 'X4']
    cointest = gps.residual_cointest(laglen, res['id'].unique(), trait_pred_teQTLs, y_cov,cov_fixed2,cov_long)

    #Power test for each signal
    p_cutoff = 5e-5 #The cutoff of Pvalue for power test
    qtlid = res['id'].unique() #eQTLs for power test
    power_test = gps.power(laglen, qtlid, trait_pred_teQTLs, y_cov, p_cutoff,cov_fixed2,cov_long)

    #Importance test for each signal
    class_map = {'AR1': 1, 'AR2': 2, 'AR3': 3, 'AR4': 4, 'AR5': 5}
    res.AR = res.AR.map(class_map)
    importance = gps.importance(res, trait_pred_teQTLs, y_cov)






