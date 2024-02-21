"""
Componets of power test for significant teQTLs
"""
import sys
sys.path.append('./StQTL_2sGenGPS')
import StQTL as st
import numpy as np
import pandas as pd
import torch
from multiprocessing import Pool
from functools import reduce
from multiprocessing import set_start_method
import category_encoders as ce

try:
    set_start_method('spawn')
except RuntimeError:
    pass

def select(x, y):
    cov_sim = []
    for i in x:
        cov_sim.append(y[y['participant_id'].isin([i])])
    out = reduce(lambda left, right: pd.concat([left, right], axis=0, sort=False), cov_sim)
    return out

def print_error(value):
    print("error:", value)

def pwr_test(x, y, GT, columns,time_points,n_terms, I, N, snpid, pheid):
    y.index = x.index
    a = [1, 2, -1]
    a = pd.DataFrame(a, columns=['a'])
    encoder = ce.BinaryEncoder(cols=['a']).fit(a)

    t_term = []
    column_temp = []
    for i in range(n_terms):
        if i<1:
            t0 = pd.DataFrame(np.ones((x.shape[0], 1)), index=x.index.values)
            t_term.append(t0)
        else:
            t_temp = x[['time']]**i
            t_term.append(t_temp)
        column_temp.append('t'+str(i))
    t_term = reduce(lambda left, right: pd.concat([left, right], axis=1, sort=False), t_term)
    t_term.index = x.index
    t_term.columns = column_temp

    x = pd.concat([t_term, x], axis=1)
    index = list(GT.index)
    snpid = list(snpid)
    rd_llr = []
    m = 1
    while m <= I:
        ID = np.random.choice(index, N)
        gt = GT.loc[ID, :]
        x_temp = select(ID, x)
        y_temp = y.loc[x_temp.index.values, :]
        x_temp = x_temp.loc[:, columns]
        snp_sim = torch.ones(len(snpid), N, 2)

        for g in range(len(snpid)):
            temp = gt[[snpid[g]]]
            temp.columns = ['a']
            vr = encoder.transform(temp).values
            vr = torch.as_tensor(vr)
            snp_sim[g, :, :] = vr

        x_temp = x_temp.values
        x_temp = torch.as_tensor(x_temp)
        x_temp2 = torch.ones((x_temp.shape[1], N, time_points,))  # K*N*T
        for k in range(x_temp.shape[1]):
            vr = x_temp[:, k]
            vr = torch.reshape(vr, (N, time_points,))
            x_temp2[k, :, :] = vr

        y_temp = y_temp.values
        y_temp = torch.log2(torch.from_numpy(y_temp) + 0.01)
        y_temp2 = torch.ones(y_temp.shape[1], N, time_points,)
        for n in range(y_temp.shape[1]):
            vr = y_temp[:, n]
            vr = torch.reshape(vr, (N, time_points,))
            y_temp2[n, :, :] = vr

        res = st.teQTL(x_temp2, y_temp2, snp_sim, time_points, snpid, pheid,n_terms)
        lrt = res[['Log Likelihood Ratio']].values.flatten()
        rd_llr.append(lrt)
        m = m + 1

    rd_llr = np.array(rd_llr)
    out = pd.DataFrame(rd_llr)
    return out

def print_error(value):
    print("error:", value)

def pwr_test_main(cov,y,genotype,teQTLs,time_points,n_process,n_terms,n_part,n_permut):
    columns = []
    for i in range(n_terms):
        columns.append('t'+str(i))
    columns.extend(cov.columns[2:])
    llr_test = [] #Log likelihood ratio
    cut = 1000
    for n in range(0, teQTLs.shape[0], cut):
        start = n
        stop = n + cut if n + cut <= teQTLs.shape[0] else teQTLs.shape[0]
        cis_temp = teQTLs.iloc[start:stop, :]
        #Genes
        y2 = y.loc[:, cis_temp.loc[:, 'geneid']]
        pheid = y2.columns.values
        snpid = cis_temp.loc[:, 'snpid'].values

        N = n_part

        p = Pool(n_process)
        p_out = []
        data = []
        I = n_permut/5
        for n in range(0, 5):
            p_out.append(p.apply_async(pwr_test, (cov, y2, genotype, columns,time_points,n_terms, I, N, snpid, pheid),
                                       error_callback=print_error))
        p.close()
        p.join()

        for proce in p_out:
            data.append(proce.get())

        out = reduce(lambda left, right: pd.concat([left, right], axis=0, sort=False), data)
        out.columns = (cis_temp['snpid'] + '_' + cis_temp['geneid']).tolist()
        out.index = list(range(out.shape[0]))
        llr_test.append(out)
    llr_test = reduce(lambda left, right: pd.concat([left, right], axis=1, sort=False), llr_test)
    return llr_test

