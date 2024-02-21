"""
Componets of the teQTL based on polynomial regression model in tensor
"""
import torch
import numpy as np
from scipy.stats import f
import pandas as pd
from multiprocessing import Pool
from functools import reduce
import category_encoders as ce
import prepare as pp

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

    def teQTL(self,x,y,snp, snpid, pheid,n):
        y = y.to(device)
        x = x.to(device)
        snp = snp.to(device)

        x0 = torch.einsum('ij,kj->ikj', snp[:, :, 0], x[0:n, :])
        x1 = torch.einsum('ij,kj->ikj', snp[:, :, 1], x[0:n, :])
        x_temp = torch.cat([x0, x1],dim=1)

        x_new = torch.ones((y.shape[0], x.shape[0]+n*2, y.shape[1]))
        for i in range(y.shape[0]):
            temp = torch.cat((x_temp[i, :, :], x.reshape((x.shape[0],-1))), dim=0)
            x_new[i, :, :] = temp
        x_new = x_new.to(device)
        x_null = x_new[:, n*2:, :]
        beta0 = self.linear(x_null,y)
        beta = self.linear(x_new,y)
        r0 = torch.einsum('ij,ijk->ik', beta0.double(), x_null.double())
        r1 = torch.einsum('ij,ijk->ik', beta.double(), x_new.double())
        sser = torch.sum((y - r0) ** 2, dim=1)
        ssef = torch.sum((y - r1) ** 2, dim=1)
        Fvalue = ((sser - ssef) / (x_new.shape[1]-x_null.shape[1])) / (ssef / (y.shape[1] - x_new.shape[1]))
        p = f.sf(Fvalue.cpu(), x_new.shape[1]-x_null.shape[1], y.shape[1]-x_new.shape[1])
        #R square
        y2 = y.transpose(0, 1)
        r1_t = r1.transpose(0, 1)
        rss = torch.sum((y2 - r1_t) ** 2, dim=0) / y2.shape[0]
        tss = torch.sum((y2 - torch.mean(y2, axis=0)) ** 2, dim=0) / y2.shape[0]
        r_square = 1 - (rss / tss)

        out = {'Variant_ID': snpid, 'Ensembl_ID': pheid, 'Pvalue': p,'R2':r_square}
        out = pd.DataFrame(out)
        beta = pd.DataFrame(beta.cpu().numpy())
        out = pd.concat([out,beta],axis=1)
        return out

    def interac(self,x,y,snp,snpid, pheid,n):
        y = y.to(device)
        x = x.to(device)
        snp = snp.to(device)
        x0 = torch.einsum('ij,kj->ikj', snp[:, :, 0], x[0:n, :])
        x1 = torch.einsum('ij,kj->ikj', snp[:, :, 1], x[0:n, :])
        x_temp = torch.cat([x0, x1],dim=1)  

        x_new = torch.ones((y.shape[0], x.shape[0]+n*2, y.shape[1]))
        for i in range(y.shape[0]):
            temp = torch.cat((x_temp[i, :, :], x.reshape((x.shape[0],-1))), dim=0)
            x_new[i, :, :] = temp
        x_new = x_new.to(device)
        x_null = torch.cat([x_new[:, 0:1, :, ],x_new[:, n:n+1, :], x_new[:, 2*n:, :]],dim=1)
        beta0 = self.linear(x_null,y)
        beta = self.linear(x_new,y)
        r0 = torch.einsum('ij,ijk->ik', beta0.double(), x_null.double())
        r1 = torch.einsum('ij,ijk->ik', beta.double(), x_new.double())
        sser = torch.sum((y - r0) ** 2, dim=1)
        ssef = torch.sum((y - r1) ** 2, dim=1)
        Fvalue = ((sser - ssef) / (x_new.shape[1]-x_null.shape[1])) / (ssef / (y.shape[1] - x_new.shape[1])) 
        p = f.sf(Fvalue.cpu(), x_new.shape[1]-x_null.shape[1], y.shape[1]-x_new.shape[1])
        #R square
        y2 = y.transpose(0, 1)
        r1_t = r1.transpose(0, 1)
        rss = torch.sum((y2 - r1_t) ** 2, dim=0)/y2.shape[0]
        tss = torch.sum((y2 - torch.mean(y2,axis=0))**2,dim=0)/y2.shape[0]
        r_square = 1-(rss/tss)

        out = {'Variant_ID': snpid, 'Ensembl_ID': pheid, 'Pvalue': p,'R2':r_square}
        out = pd.DataFrame(out)
        beta = pd.DataFrame(beta.cpu().numpy())
        out = pd.concat([out,beta],axis=1)
        return out

    def r2(self,x,y,snp,n):
        y = y.to(device)
        x = x.to(device)
        snp = snp.to(device)
        if n>0:
            x0 = torch.einsum('ij,kjl->ikjl', snp[:, :, 0], x[0:n, :, :])
            x0 = torch.reshape(x0, (y.shape[0], n, -1))
            x1 = torch.einsum('ij,kjl->ikjl', snp[:, :, 1], x[0:n, :, :])
            x1 = torch.reshape(x1, (y.shape[0], n, -1))
            x_temp = torch.cat([x0, x1], dim=1)

            x_new = torch.ones((y.shape[0], x.shape[0] + n * 2, y.shape[1]))
            for i in range(y.shape[0]):
                temp = torch.cat((x_temp[i, :, :], x.reshape((x.shape[0], -1))), dim=0)
                x_new[i, :, :] = temp
            x_new = x_new.to(device)
            beta = self.linear(x_new, y)
        else:
            x_new = torch.ones((y.shape[0], x.shape[0] + n * 2, y.shape[1]))
            for i in range(y.shape[0]):
                x_new[i, :, :] = x.reshape((x.shape[0], -1))
            x_new = x_new.to(device)
            beta = self.linear(x_new, y)
        r1 = torch.einsum('ij,ijk->ik', beta.double(), x_new.double())
        r1 = r1.transpose(0,1)
        y = y.transpose(0,1)
        rss = torch.sum((y - r1) ** 2, dim=0)/y.shape[0]
        tss = torch.sum((y - torch.mean(y,axis=0))**2,axis=0)/y.shape[0]
        r_square = 1-(rss/tss)
        return r_square.cpu()      

class Predict():
    def __init__(self,gt,covars):
        self.gt = gt.to(device)
        self.covars = covars.to(device)

    def predict(self,beta,n):
        beta = beta.to(device)
        covars = self.covars
        gt = self.gt    
        delta0 = gt[:,:, 0]
        delta1 = gt[:,:, 1]

        g0 = torch.einsum('ik,mk->im', beta[:,0:n], covars)
        z0 = torch.einsum('ij,ij->ij', delta0, g0)

        g1 = torch.einsum('ik,mk->im', beta[:,n:2*n], covars)
        z1 = torch.einsum('ij,ij->ij', delta1, g1)

        z = torch.einsum('ik,mk->im', beta[:,2*n:3*n], covars)
        out = z0 + z1 + z
        return out

def print_error(value):
    print("error:", value)

def poly_teQTL_main(cov,y,genotype,n_process,n_terms,cisDist=None,snpspos=None,genepos=None):
    cov_new = []
    column = []
    for i in range(n_terms):
        if i<1:
            t0 = pd.DataFrame(np.ones((cov.shape[0], 1)), index=cov.index.values)
            cov_new.append(t0)
        else:
            t_temp = cov[['time']]**i
            cov_new.append(t_temp)
        column.append('t'+str(i))
    cov_new = reduce(lambda left, right: pd.concat([left, right], axis=1, sort=False), cov_new)
    cov_new.index = cov.index
    cov_new.columns = column

    cov_new = pd.concat([cov_new, cov.iloc[:, 2:]], axis=1)  
    cov_new = torch.as_tensor(cov_new.values)
    cov_new = cov_new.transpose(0, 1)

    if cisDist is None:
        cis_snpgeneis = []
        for i in range(y.shape[1]):
            geneid_temp = np.repeat(y.columns.values[i], genotype.shape[1])
            data = {'snpid': genotype.columns.values, 'geneid': geneid_temp}
            cis_snpgeneis.append(pd.DataFrame(data))
        cis_snpgeneid = reduce(lambda left, right: pd.concat([left, right], axis=0, sort=False), cis_snpgeneis)
    else:
        cis_snpgeneid = pp.cis(snpspos, genepos, cisDist)

    print('Number of cis-SNP-Gene pair:',cis_snpgeneid.shape[0])
    p = Pool(n_process)
    p_out = []
    data = []
    cut = 2000
    a = [1, 2, -1]
    a = pd.DataFrame(a, columns=['a'])
    encoder = ce.BinaryEncoder(cols=['a']).fit(a)
    print(encoder.transform(a))
    genotype = genotype.loc[cov.loc[:, 'participant_id'].values, :]
    for dt in pp.part_func(cut, cis_snpgeneid, genotype):
        cis_temp = dt[0]
        genotype_temp = dt[1]
        y2 = y.loc[:, cis_temp.loc[:, 'geneid']]
        pheid = y2.columns.values
        y2 = y2.values
        y2 = torch.log2(torch.from_numpy(y2) + 0.01)
        y2 = y2.transpose(0, 1)

        # genotype
        snp = torch.ones(cis_temp.shape[0], int(cov.shape[0]), 2)

        for g in range(cis_temp.shape[0]):
            snpid = cis_temp.loc[:, 'snpid'].values[g]
            temp = genotype_temp[[snpid]]
            temp.columns = ['a']
            vr = encoder.transform(temp).values
            vr = torch.as_tensor(vr)
            snp[g, :, :] = vr

        snpid = cis_temp.loc[:, 'snpid'].values
        last = p.apply_async(LinearRegression_batch().teQTL, (cov_new, y2, snp, snpid, pheid, n_terms), error_callback=print_error)
        p_out.append(last)
        if len(p._cache) > 1e2:
            last.wait()
    p.close()
    p.join()

    for proce in p_out:
        data.append(proce.get())

    out = reduce(lambda left, right: pd.concat([left, right], axis=0, sort=False), data)
    return out



def poly_dynamic_main(cov,y,genotype,n_process,n_terms,cisDist=None,snpspos=None,genepos=None):
    cov_new = []
    column = []
    for i in range(n_terms):
        if i<1:
            t0 = pd.DataFrame(np.ones((cov.shape[0], 1)), index=cov.index.values)
            cov_new.append(t0)
        else:
            t_temp = cov[['time']]**i
            cov_new.append(t_temp)
        column.append('t'+str(i))
    cov_new = reduce(lambda left, right: pd.concat([left, right], axis=1, sort=False), cov_new)
    cov_new.index = cov.index
    cov_new.columns = column

    cov_new = pd.concat([cov_new, cov.iloc[:, 2:]], axis=1)  #
    cov_new = torch.as_tensor(cov_new.values)
    cov_new = cov_new.transpose(0, 1)

    if cisDist is None:
        cis_snpgeneid = []
        for i in range(y.shape[1]):
            geneid_temp = np.repeat(y.columns.values[i], genotype.shape[1])
            data = {'snpid': genotype.columns.values, 'geneid': geneid_temp}
            cis_snpgeneid.append(pd.DataFrame(data))
        cis_snpgeneid = reduce(lambda left, right: pd.concat([left, right], axis=0, sort=False), cis_snpgeneid)
    else:
        cis_snpgeneid = pp.cis(snpspos, genepos, cisDist)

    print('Number of cis-SNP-Gene pair:',cis_snpgeneid.shape[0])
    p = Pool(n_process)
    p_out = []
    data = []
    cut = 2000
    a = [1, 2, -1]
    a = pd.DataFrame(a, columns=['a'])
    encoder = ce.BinaryEncoder(cols=['a']).fit(a)
    print(encoder.transform(a))
    genotype = genotype.loc[cov.loc[:, 'participant_id'].values, :]

    for dt in pp.part_func(cut, cis_snpgeneid, genotype):
        cis_temp = dt[0]
        genotype_temp = dt[1]
        y2 = y.loc[:, cis_temp.loc[:, 'geneid']]
        pheid = y2.columns.values
        y2 = y2.values
        y2 = torch.log2(torch.from_numpy(y2) + 0.01)
        y2 = y2.transpose(0, 1)
        # genotype
        snp = torch.ones(cis_temp.shape[0], int(cov.shape[0]), 2)

        for g in range(cis_temp.shape[0]):
            snpid = cis_temp.loc[:, 'snpid'].values[g]
            temp = genotype_temp[[snpid]]
            temp.columns = ['a']
            vr = encoder.transform(temp).values
            vr = torch.as_tensor(vr)
            snp[g, :, :] = vr

        snpid = cis_temp.loc[:, 'snpid'].values
        last = p.apply_async(LinearRegression_batch().interac, (cov_new, y2, snp, snpid, pheid, n_terms), error_callback=print_error)
        p_out.append(last)
        if len(p._cache) > 1e2:
            last.wait()
    p.close()
    p.join()

    for proce in p_out:
        data.append(proce.get())

    out = reduce(lambda left, right: pd.concat([left, right], axis=0, sort=False), data)
    return out


        