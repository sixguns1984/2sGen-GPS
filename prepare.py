"""
Componets of data preprocess for StQTL and 2sGen-GPS analysis

"""
import numpy as np
import pandas as pd
from functools import reduce
import StQTL as st
import category_encoders as ce
import torch
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp

def cis(snpsloc,geneloc,dist):
    out = []
    for i in range(geneloc.shape[0]):
        gene_temp = geneloc.iloc[i, :]
        chr = gene_temp['chr']
        cis_snp = snpsloc[snpsloc['chr'] == chr]
        l = cis_snp['pos'] - gene_temp['start']
        r = cis_snp['pos'] - gene_temp['end']
        inter = (l+0.1)*(r+0.1)
        cis_snp1 = cis_snp[abs(l)< dist]
        cis_snp2 = cis_snp[abs(r)< dist]
        cis_snp3 = cis_snp[inter<0]
        cis_snp = pd.concat([cis_snp1, cis_snp2,cis_snp3],axis=0)
        cis_snp = cis_snp.drop_duplicates('snpid',keep='first')
        snpid = cis_snp['snpid']
        geneid = np.repeat(gene_temp['geneid'], cis_snp.shape[0])
        data = {'snpid': snpid.values, 'geneid': geneid}
        out.append(pd.DataFrame(data))
    out = reduce(lambda left, right: pd.concat([left, right], axis=0, sort=False), out)
    return out



def part_func(cut, id_data, gt_data): #Dataset partitioning for multiprocess analysis
    for n in range(0, id_data.shape[0], cut):
        start = n
        stop = n + cut if n + cut <= id_data.shape[0] else id_data.shape[0]
        id_temp = id_data.iloc[start:stop, :]
        temp = id_temp.drop_duplicates('snpid')
        gt_temp = gt_data.loc[:, temp.loc[:, 'snpid']]
        yield id_temp, gt_temp


def trait_pred_teQTL(genotype,cov,teQTL_coefs,n_terms): #Componet for genetic predicting longitudinal gene pression
    a = [1,2,-1]
    a = pd.DataFrame(a,columns=['a'])
    encoder = ce.BinaryEncoder(cols=['a']).fit(a)
    snp = torch.ones(teQTL_coefs.shape[0], cov.shape[0], 2)
    genotype2 = genotype.loc[cov.loc[:,'participant_id'].values,:]
    for m in range(teQTL_coefs.shape[0]):
        snpid = teQTL_coefs.loc[:, 'Variant_ID'].values[m]
        temp = genotype2[[snpid]]
        temp.columns = ['a']
        vr = encoder.transform(temp).values
        vr = torch.as_tensor(vr)
        snp[m, :, :] = vr
    beta = torch.as_tensor(teQTL_coefs.iloc[:,19:].values)

    cov_new = []
    column = []
    for i in range(n_terms):
        if i < 1:
            t0 = pd.DataFrame(np.ones((cov.shape[0], 1)), index=cov.index.values)
            cov_new.append(t0)
        else:
            t_temp = cov[['time']] ** i
            cov_new.append(t_temp)
        column.append('t' + str(i))
    cov_new = reduce(lambda left, right: pd.concat([left, right], axis=1, sort=False), cov_new)
    cov_new.index = cov.index
    cov_new.columns = column
    cov_new = cov_new.values
    cov_new = torch.as_tensor(cov_new)
    fitted = st.Predict(snp,cov_new).predict_2d(beta)
    fitted = fitted.cpu()
    fitted = fitted.transpose(0,1)
    fitted = fitted.numpy()
    id = teQTL_coefs['Variant_ID']+'_'+teQTL_coefs['Ensembl_ID']
    out = pd.DataFrame(fitted,columns=id.values)
    out.index = cov['participant_id'] + '_' + cov['time'].map(str)
    return out


def adfuller_batch(x, regression="ct"): #ADF test for temporal trait
    columns = x.columns.values
    participant_id = x[columns[0]]
    participant_id = participant_id.drop_duplicates()
    xdiff = []
    xt1 = []
    for i in range(participant_id.shape[0]):  
        temp = x[x[columns[0]] == participant_id.iloc[i]]
        temp.sort_values(columns[1], inplace=True)
        xdiff_temp = temp.iloc[:, 2:].diff()
        xdiff_temp = xdiff_temp.iloc[1:, :]
        xdiff.append(xdiff_temp)
        temp.drop(labels=temp.index.values[-1], inplace=True)
        xt1.append(temp)
    xt1 = reduce(lambda left, right: pd.concat([left, right], axis=0, sort=False), xt1)
    xdiff = reduce(lambda left, right: pd.concat([left, right], axis=0, sort=False), xdiff)
    xt1 = xt1.iloc[:, 1:]
    xt1 = xt1.values
    nobs = xt1.shape[0]
    x0 = np.ones(nobs)
    X = np.column_stack((x0, xt1))
    xdiff = xdiff.values
    # ols lm regression
    pvalue_batch = []
    for i in range(xdiff.shape[1]):
        if regression == 'ct':
            resols = OLS(xdiff[:, i], X[:, [0, 1, i + 2]]).fit()
            adfstat = resols.tvalues[-1]
            pvalue = mackinnonp(adfstat, regression=regression, N=1)
            pvalue_batch.append(pvalue)
        elif regression == 'nc':
            resols = OLS(xdiff[:, i], X[:, i + 2]).fit()
            adfstat = resols.tvalues[-1]
            pvalue = mackinnonp(adfstat, regression=regression, N=1)
            pvalue_batch.append(pvalue)
    out = {'id': columns[2:], 'adf.test': pvalue_batch}
    out = pd.DataFrame(out)
    return out


