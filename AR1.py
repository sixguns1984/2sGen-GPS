"""
Componets of teQTL analysis using first order Auto Regression
"""
import torch
import numpy as np
from scipy.stats import f
import pandas as pd
import prepare as pp
from multiprocessing import Pool
from functools import reduce
import category_encoders as ce

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.DoubleTensor)

def inv(A, eps = 1e-10):
    assert len(A.shape) == 3 and \
           A.shape[1] == A.shape[2]
    n = A.shape[1]
    U = A.clone().data
    L = A.new_zeros(A.shape).data
    L[:, range(n), range(n)] = 1
    I = L.clone()
    L_inv = I
    for i in range(n-1):
        L[:, i+1:, i:i+1] = U[:, i+1:, i:i+1] / (U[:, i:i+1, i:i+1] + eps)
        L_inv[:, i+1:, :] = L_inv[:, i+1:, :] - L[:, i+1:, i:i+1].matmul(L_inv[:, i:i+1, :])
        U[:, i+1:, :] = U[:, i+1:, :] - L[:, i+1:, i:i+1].matmul(U[:, i:i+1, :])
    # [U L^{-1}] -> [I U^{-1}L^{-1}] = [I (LU)^{-1}]
    A_inv = L_inv
    for i in range(n-1, -1, -1):
        A_inv[:, i:i+1, :] = A_inv[:, i:i+1, :] / (U[:, i:i+1, i:i+1] + eps)
        U[:, i:i+1, :] = U[:, i:i+1, :] / (U[:, i:i+1, i:i+1] + eps)
        if i > 0:
            A_inv[:, :i, :] = A_inv[:, :i, :] - U[:, :i, i:i+1].matmul(A_inv[:, i:i+1, :])
            U[:, :i, :] = U[:, :i, :] - U[:, :i, i:i+1].matmul(U[:, i:i+1, :])
    A_inv_grad = - A_inv.matmul(A).matmul(A_inv)
    return A_inv + A_inv_grad - A_inv_grad.data
    
class LinearRegression_batch():
    def __init__(self):
        self.x = None
        self.y = None
    def linear(self,x,y):
        k = x.shape[1]
        n = x.shape[0]
        alpha = torch.ones((n, k, k)).to(device)
        for i in range(k):
            temp = torch.einsum('ijk,ik->ijk', x.double(), x[:, i, :].double())
            temp = torch.einsum('ijk->ij', temp)
            alpha[:, i, :] = temp
        ymat = torch.ones((n, k)).to(device)
        for j in range(k):
            x_temp = x[:, j, :].double()
            x_temp = torch.reshape(x_temp,(n,1,-1))
            temp = torch.einsum('ij,ikj->ij', y.double(), x_temp)
            temp = torch.einsum('ij->i',temp)
            ymat[:, j] = temp
        beta = torch.einsum('ijk,ik->ijk',inv(alpha),ymat)
        beta = torch.einsum('ijk->ij', beta)
        return beta

    def teQTL(self,x,x_cov,y,snp,snpid, pheid):
        y = y.to(device)
        x = x.to(device)
        x_cov = x_cov.to(device)
        snp = snp.to(device)
        const = torch.ones((1,x.shape[1])).to(device)
        x0 = torch.einsum('ij,ij->ij', snp[:, :, 0], x)
        x1 = torch.einsum('ij,ij->ij', snp[:, :, 1], x)
        x = torch.reshape(x, (y.shape[0], -1))
        
        snp0 = torch.einsum('ij,kj->ikj', snp[:, :, 0], const)
        snp1 = torch.einsum('ij,kj->ikj', snp[:, :, 1], const)

        x_new = torch.ones((y.shape[0], x_cov.shape[0]+6, y.shape[1]))
        for i in range(y.shape[0]):
            temp = torch.cat((snp0[i,:,:],x0[i:i+1, :],snp1[i,:,:],x1[i:i+1, :],x[i:i+1, :],const, x_cov), dim=0)
            x_new[i, :, :] = temp
        x_new = x_new.to(device)
        x_null = x_new[:, 4:, :]
        beta0 = self.linear(x_null,y)
        beta = self.linear(x_new,y)
        r0 = torch.einsum('ij,ijk->ik', beta0.double(), x_null.double())
        r1 = torch.einsum('ij,ijk->ik', beta.double(), x_new.double())
        sser = torch.sum((y - r0) ** 2, dim=1)
        ssef = torch.sum((y - r1) ** 2, dim=1)
        Fvalue = ((sser - ssef) / (x_new.shape[1]-x_null.shape[1])) / (ssef / (y.shape[1] - x_new.shape[1]))
        p = f.sf(Fvalue.cpu(), x_new.shape[1]-x_null.shape[1], y.shape[1]-x_new.shape[1])
        out = {'Variant_ID': snpid, 'Ensembl_ID': pheid, 'Pvalue': p}
        out = pd.DataFrame(out)
        beta = pd.DataFrame(beta.cpu().numpy())
        out = pd.concat([out,beta],axis=1)
        return out

    def predict(self,x,snp,beta):
        beta = beta.to(device)
        x = x.to(device)
        snp = snp.to(device)
        const = torch.ones((1,x.shape[1],x.shape[2])).to(device)
        x0 = torch.einsum('ij,ijl->ijl', snp[:, :, 0], x)
        x0 = torch.reshape(x0, (x.shape[0],-1))        
        x1 = torch.einsum('ij,ijl->ijl', snp[:, :, 1], x)
        x1 = torch.reshape(x1, (x.shape[0], -1))        
        x = torch.reshape(x, (x.shape[0], -1))
        
        snp0 = torch.einsum('ij,kjl->ikjl', snp[:, :, 0], const)
        snp0 = torch.reshape(snp0, (x.shape[0],1,-1))        
        snp1 = torch.einsum('ij,kjl->ikjl', snp[:, :, 1], const)
        snp1 = torch.reshape(snp1, (x.shape[0],1, -1))        
        const = torch.reshape(const, (1, -1))
        
        x_new = torch.ones((x.shape[0], 6, x.shape[1]))
        for i in range(x.shape[0]):
            temp = torch.cat((snp0[i,:,:],x0[i:i+1, :],snp1[i,:,:],x1[i:i+1, :],x[i:i+1, :],const), dim=0)
            x_new[i, :, :] = temp
        x_new = x_new.to(device)
        y = torch.einsum('ij,ijk->ik', beta.double(), x_new.double())
        return y.cpu()


def print_error(value):
    print("error:", value)


def ar_data_prepare(x):
    xlag = []
    x_now = []
    participant_id = x['participant_id'].unique()
    for i in range(participant_id.shape[0]):
        temp = x[x['participant_id'] == participant_id[i]].copy()
        temp.sort_values('time', inplace=True)
        temp.drop(labels=temp.index.values[-1:], inplace=True)
        xlag.append(temp)
        temp = x[x['participant_id'] == participant_id[i]].copy()
        temp.sort_values('time', inplace=True)
        temp.drop(labels=temp.index.values[0:1], inplace=True)
        x_now.append(temp)
    x_now = reduce(lambda left, right: pd.concat([left, right], axis=0, sort=False), x_now)
    xlag = reduce(lambda left, right: pd.concat([left, right], axis=0, sort=False), xlag)
    return xlag,x_now

def ar1_main(cov,y,genotype,n_process,cisDist=None,snpspos=None,genepos=None):
    cov_lag,cov_now = ar_data_prepare(cov)
    cov_lag = torch.as_tensor(cov_lag.iloc[:,2:].values)
    x_cov = cov_lag.transpose(0, 1)
    y.index = cov.index
    y = pd.concat([cov.loc[:,['participant_id','time']], y], axis=1)
    ylag,y_now = ar_data_prepare(y)

    if cisDist is None:
        cis_snpgeneid = []
        for i in range(y.shape[1]):
            geneid_temp = np.repeat(y.columns.values[i], genotype.shape[1])
            data = {'snpid': genotype.columns.values, 'geneid': geneid_temp}
            cis_snpgeneid.append(pd.DataFrame(data))
        cis_snpgeneid = reduce(lambda left, right: pd.concat([left, right], axis=0, sort=False), cis_snpgeneid)
    else:
        cis_snpgeneid = pp.cis(snpspos, genepos, cisDist)

    p = Pool(n_process)
    p_out = []
    data = []
    cut = 2000
    a = [1, 2, -1]
    a = pd.DataFrame(a, columns=['a'])
    encoder = ce.BinaryEncoder(cols=['a']).fit(a)
    print(encoder.transform(a))
    genotype = genotype.loc[ylag.loc[:, 'participant_id'].values, :]
    for dt in pp.part_func(cut, cis_snpgeneid, genotype):
        cis_temp = dt[0]
        pheid = cis_temp.loc[:, 'geneid'].values
        genotype_temp = dt[1]

        y_now2 = y_now.loc[:, cis_temp.loc[:, 'geneid']]
        y_now2 = torch.log2(torch.from_numpy(y_now2.values) + 0.01)
        y_now2 = y_now2.transpose(0, 1)

        ylag2 = ylag.loc[:, cis_temp.loc[:, 'geneid']]
        ylag2 = torch.log2(torch.from_numpy(ylag2.values) + 0.01)
        ylag2 = ylag2.transpose(0, 1)
        # genotype
        snp = torch.ones(cis_temp.shape[0], int(ylag.shape[0]), 2)

        for g in range(cis_temp.shape[0]):
            snpid = cis_temp.loc[:, 'snpid'].values[g]
            temp = genotype_temp[[snpid]]
            temp.columns = ['a']
            vr = encoder.transform(temp).values
            vr = torch.as_tensor(vr)
            snp[g, :, :] = vr

        snpid = cis_temp.loc[:, 'snpid'].values
        last = p.apply_async(LinearRegression_batch().teQTL,(ylag2,x_cov,y_now2, snp,snpid, pheid), error_callback=print_error)
        p_out.append(last)
        if len(p._cache) > 1e2:
            last.wait()
    p.close()
    p.join()

    for proce in p_out:
        data.append(proce.get())

    out = reduce(lambda left,right:pd.concat([left,right],axis=0,sort=False),data)
    return out