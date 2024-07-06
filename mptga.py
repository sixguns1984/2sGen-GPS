"""
Componets of the MPTGA
"""
import numpy as np
import prepare as pp
import pandas as pd
import torch
from scipy.stats import chi2
from multiprocessing import Pool
from functools import reduce
import category_encoders as ce

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.DoubleTensor)  


def inv(A, eps=1e-10): #Inverse matrix calculation
    assert len(A.shape) == 3 and \
           A.shape[1] == A.shape[2]
    n = A.shape[1]
    U = A.clone().data
    L = A.new_zeros(A.shape).data
    L[:, range(n), range(n)] = 1
    I = L.clone()

    L_inv = I
    for i in range(n - 1):
        L[:, i + 1:, i:i + 1] = U[:, i + 1:, i:i + 1] / (U[:, i:i + 1, i:i + 1] + eps)
        L_inv[:, i + 1:, :] = L_inv[:, i + 1:, :] - L[:, i + 1:, i:i + 1].matmul(L_inv[:, i:i + 1, :])
        U[:, i + 1:, :] = U[:, i + 1:, :] - L[:, i + 1:, i:i + 1].matmul(U[:, i:i + 1, :])

    A_inv = L_inv
    for i in range(n - 1, -1, -1):
        A_inv[:, i:i + 1, :] = A_inv[:, i:i + 1, :] / (U[:, i:i + 1, i:i + 1] + eps)
        U[:, i:i + 1, :] = U[:, i:i + 1, :] / (U[:, i:i + 1, i:i + 1] + eps)

        if i > 0:
            A_inv[:, :i, :] = A_inv[:, :i, :] - U[:, :i, i:i + 1].matmul(A_inv[:, i:i + 1, :])
            U[:, :i, :] = U[:, :i, :] - U[:, :i, i:i + 1].matmul(U[:, i:i + 1, :])

    A_inv_grad = - A_inv.matmul(A).matmul(A_inv)
    return A_inv + A_inv_grad - A_inv_grad.data


class Beta():  # Componet of multidimensional tensor computation for coefficients of StQTL model
    def __init__(self, time_points):
        self.time_points = time_points

    def ymat(self, y, x):
        M = self.time_points
        G = y.shape[0]
        K = x.shape[1]
        B1 = torch.ones((G, K))
        B2 = torch.ones((G, K))
        B3 = torch.ones((G, K))
        for i in range(K):
            V1 = x[:, i, :, :]
            Q1 = torch.einsum('ijk,ijk->ijk', y[:, :, (0, M - 1)], V1[:, :, (0, M - 1)])
            B1[:, i] = torch.einsum('ijk->i', Q1)
            Q2 = torch.einsum('ijk,ijk->ijk', y[:, :, 0:M - 1], V1[:, :, 1:]) + torch.einsum('ijk,ijk->ijk',
                                                                                             y[:, :, 1:],
                                                                                             V1[:, :, 0:M - 1])
            B2[:, i] = torch.einsum('ijk->i', Q2)
            Q3 = torch.einsum('ijk,ijk->ijk', y[:, :, 1:M - 1], V1[:, :, 1:M - 1])
            B3[:, i] = torch.einsum('ijk->i', Q3)
        return torch.stack([B1, B2, B3]).to(device)

    def alpha(self, x):
        M = self.time_points
        K = x.shape[1]
        R = x.shape[0]
        Alpha1 = torch.ones((R, K * K))
        Alpha2 = torch.ones((R, K * K))
        Alpha3 = torch.ones((R, K * K))
        iter = 0
        for i in range(K):
            for j in range(K):
                U1 = x[:, i, :, :]
                U2 = x[:, j, :, :]
                Q1 = torch.einsum('ijk,ijk->ijk', U1[:, :, (0, M - 1)], U2[:, :, (0, M - 1)])
                Alpha1[:, iter] = torch.einsum('ijk->i', Q1)
                Q2 = torch.einsum('ijk,ijk->ijk', U1[:, :, 0:M - 1], U2[:, :, 1:]) + torch.einsum('ijk,ijk->ijk',
                                                                                                  U1[:, :, 1:],
                                                                                                  U2[:, :, 0:M - 1])
                Alpha2[:, iter] = torch.einsum('ijk->i', Q2)
                Q3 = torch.einsum('ijk,ijk->ijk', U1[:, :, 1:M - 1], U2[:, :, 1:M - 1])
                Alpha3[:, iter] = torch.einsum('ijk->i', Q3)
                iter = iter + 1
        return torch.stack([Alpha1, Alpha2, Alpha3]).to(device)

    def effect(self, alpha, b):  # alpha0.shape=(R, Vn, Vn),b0.shape=(R,G,Vn)
        beta = torch.einsum('ijk,ik->ijk', inv(alpha), b)
        beta = torch.einsum('ijk->ij', beta)
        return beta


class T_Genetic(Beta): # Statitical componet of StQTL model
    def __init__(self, y, x, time_points):
        super(T_Genetic, self).__init__(time_points)
        self.x = x
        self.y = y
        self.Vn = x.shape[1]  # Number of covariate
        self.N = x.shape[2]  # Number of participant
        self.G = y.shape[0]  # Number of gene
        self.I = self.alpha(x)
        self.TI = self.ymat(y, x)

    def sumqfun(self, U, V):
        M = U.shape[2]
        Q1 = torch.einsum('ijk,ijk->i', U[:, :, (0, M - 1)], V[:, :, (0, M - 1)])
        Q2 = torch.einsum('ijk,ijk->i', U[:, :, 0:M - 1], V[:, :, 1:]) + torch.einsum('ijk,ijk->i', U[:, :, 1:],
                                                                                      V[:, :, 0:M - 1])
        Q3 = torch.einsum('ijk,ijk->i', U[:, :, 1:M - 1], V[:, :, 1:M - 1])
        return torch.stack([Q1, Q2, Q3])

    def Var(self, r, beta):  #Variance compution
        M = self.time_points
        y = self.y
        x = self.x
        N = self.N
        Rmat = torch.tensor([1 / (1 - r ** 2), -r / (1 - r ** 2), (1 + r ** 2) / (1 - r ** 2)])
        Rmat = Rmat.to(device)
        g = torch.einsum('ij,ijmn->imn', beta, x)
        z = y - g
        Z = self.sumqfun(z, z)
        var = torch.einsum('i,ij->j', Rmat, Z) / (M * N)
        return var

    def loglikhood_r(self, r, var):  # Loglikelihood
        M = self.time_points
        N = self.N
        pi = torch.as_tensor(np.pi)
        log_r = - M * N / 2 * (torch.log(2 * pi)) - (N * (M - 1) / 2) * torch.log(1 - torch.as_tensor(r ** 2)) - (
                    N * M / 2) * torch.log(var) - M * N / 2
        return log_r

    def max_r(self):  # Auto-correlation proportion at maximum Loglikelihood
        G = self.G  # Number of gene
        Vn = self.Vn  #Number of covariate
        R = G  # Number of variant
        I = self.I
        TI = self.TI
        llr = torch.ones((R, 100))  # Loglikelihood matrix
        beta_list = torch.ones((100, R, Vn)).to(device)
        for k in range(100):
            r = k / 100
            Rmat = torch.as_tensor([1 / (1 - r ** 2), -r / (1 - r ** 2), (1 + r ** 2) / (1 - r ** 2)])
            Rmat = Rmat.to(device)
            alpha = torch.reshape(torch.einsum('i,ijk->jk', Rmat, I), (R, Vn, Vn))
            b = torch.einsum('i,ijk->jk', Rmat, TI)
            beta = self.effect(alpha, b)
            var = self.Var(r, beta)
            log_r = self.loglikhood_r(r, var)
            llr[:, k] = log_r
            beta_list[k, :, :] = beta

        llr = llr.numpy()
        llr_max = np.nanmax(llr, 1)
        max_r = torch.as_tensor(
            list(map(lambda x: np.unique(np.where(llr == x)[1]), llr_max))).squeeze()  # 返回R*G的AR系数矩阵
        max_r = max_r.to(device)
        beta_out = torch.ones((R, Vn)).to(device)

        for i in range(max_r.shape[0]):
            beta_out[i, :] = beta_list[max_r[i], i, :]

        return max_r.true_divide(100), beta_out

    def lrt_var(self, r, beta):  # Variance compution at maximum Loglikelihood
        M = self.time_points
        y = self.y
        x = self.x
        N = self.N
        Rmat = torch.stack([1 / (1 - r ** 2), -r / (1 - r ** 2), (1 + r ** 2) / (1 - r ** 2)])  # 3*R*G
        Rmat = Rmat.to(device)
        g = torch.einsum('ij,ijmn->imn', beta, x)
        z = y - g
        Z = self.sumqfun(z, z)
        var = torch.einsum('ij,ij->j', Rmat, Z) / (M * N)
        return var

    def stat(self):
        r, beta = self.max_r()
        var = self.lrt_var(r, beta)
        L = self.loglikhood_r(r, var)
        return r, var, L, beta


class Predict():
    def __init__(self, gt, covars):
        self.gt = gt.to(device)
        self.covars = covars.to(device)

    def predict_3d(self, beta, n):  #3D input data for prediction by teQTL
        gt = self.gt
        x = self.covars
        x0 = torch.einsum('ij,kjl->ikjl', gt[:, :, 0], x[0:n, :, :])
        x1 = torch.einsum('ij,kjl->ikjl', gt[:, :, 1], x[0:n, :, :])
        x_temp = torch.cat([x0, x1], dim=1)
        x_new = torch.ones((x_temp.shape[0], x_temp.shape[1] + x.shape[0], x.shape[1], x.shape[2]))
        x_new = x_new.to(device)
        for i in range(x_new.shape[0]):
            temp = torch.cat((x_temp[i, :, :, :], x), dim=0)
            x_new[i, :, :, :] = temp

        beta = beta.to(device)
        y = torch.einsum('ij,ijkl->ikl', beta.float(), x_new.float())
        return y.reshape(y.shape[0],-1).cpu()

    def predict_time(self, beta):  #Full model prediction
        gt = self.gt
        x = self.covars
        x0 = torch.einsum('ij,kjl->ikjl', gt[:, :, 0], x[0:4, :, :])
        x1 = torch.einsum('ij,kjl->ikjl', gt[:, :, 1], x[0:4, :, :])
        x_new = torch.cat([x0, x1], dim=1)
        beta = beta.to(device)
        y = torch.einsum('ij,ijkl->ikl', beta.float(), x_new.float())
        return y

    def predict_2d(self, beta):  #2D input data for prediction by teQTL
        beta = beta.to(device)
        covars = self.covars
        gt = self.gt
        delta0 = gt[:, :, 0]
        delta1 = gt[:, :, 1]

        g0 = torch.einsum('ik,mk->im', beta[:, 0:4], covars)
        z0 = torch.einsum('ij,ij->ij', delta0, g0)

        g1 = torch.einsum('ik,mk->im', beta[:, 4:8], covars)
        z1 = torch.einsum('ij,ij->ij', delta1, g1)

        z = torch.einsum('ik,mk->im', beta[:, 8:12], covars)
        out = z0 + z1 + z
        return out


def teQTL(x, y, snp, time_points, snpid, pheid,n):
    x = x.to(device)
    y = y.to(device)
    snp = snp.to(device)
    x0 = torch.einsum('ij,kjl->ikjl', snp[:, :, 0], x[0:n, :, :])
    x1 = torch.einsum('ij,kjl->ikjl', snp[:, :, 1], x[0:n, :, :])
    x_temp = torch.cat([x0, x1],
                       dim=1)
    x_new = torch.ones((x_temp.shape[0], x_temp.shape[1] + x.shape[0], x.shape[1], x.shape[2]))
    x_new = x_new.to(device)
    for i in range(x_new.shape[0]):
        temp = torch.cat((x_temp[i, :, :, :], x), dim=0)
        x_new[i, :, :, :] = temp

    if snp.size()[0] > 0:
        r1, var1, L1, beta1 = T_Genetic(y, x_new, time_points).stat()

        r00, var00, L00, beta00 = T_Genetic(y, x_new[:, 2*n:, :, :],
                                            time_points).stat()
        lrt0 = 2 * (L1 - L00).cpu().numpy()
        pvalue = torch.tensor(list(map(lambda x: chi2.sf(x, 2*n), lrt0)))
        pvalue = pvalue.numpy()
        r = r1.cpu()
        r = r.numpy()
        var = var1.cpu()
        var = var.numpy()
        L1 = L1.cpu()
        L1 = L1.numpy()
        aic = -2 * L1 + 2 * (x_new.shape[1] - 1)
        bic = -2 * L1 + np.log(time_points * y.shape[1]) * (x_new.shape[1] - 1)
        #PVE
        fitted = torch.einsum('ij,ijkl->ikl', beta1[:,0:3*n].float(), x_new[:,0:3*n,:,:].float())
        fitted = fitted.reshape(fitted.shape[0], -1)
        fitted = fitted.transpose(0, 1)
        y0 = y.reshape(y.shape[0], -1)
        y0 = y0.transpose(0, 1)
        tss = torch.sum((y0 - torch.mean(y0, axis=0)) ** 2, axis=0)
        rss = torch.sum((fitted - torch.mean(fitted, axis=0)) ** 2, axis=0)
        # PVE:variance in phenotype explained
        pve = rss / tss
        pve = pve.numpy()
        out = {'Variant_ID': snpid, 'Ensembl_ID': pheid, 'Auto-correlation': r, 'Pvalue': pvalue, 'Loglikelihood': L1,'Log Likelihood Ratio':lrt0, 'AIC': aic, 'BIC': bic,
               'Variance': var,'PVE':pve}
        out = pd.DataFrame(out)
        beta1 = pd.DataFrame(beta1.cpu().numpy())
        out = pd.concat([out,beta1],axis=1)
    return out

def interac(x, y, snp, time_points, snpid, pheid, n):
    x = x.to(device)
    y = y.to(device)
    snp = snp.to(device)
    x0 = torch.einsum('ij,kjl->ikjl', snp[:, :, 0], x[0:n, :, :])
    x1 = torch.einsum('ij,kjl->ikjl', snp[:, :, 1], x[0:n, :, :])
    x_temp = torch.cat([x0, x1],
                       dim=1)
    x_new = torch.ones((x_temp.shape[0], x_temp.shape[1] + x.shape[0], x.shape[1], x.shape[2]))
    x_new = x_new.to(device)
    for i in range(x_new.shape[0]):
        temp = torch.cat((x_temp[i, :, :, :], x), dim=0)
        x_new[i, :, :, :] = temp

    if snp.size()[0] > 0:
        r1, var1, L1, beta1 = T_Genetic(y, x_new, time_points).stat()
        x_null = torch.cat([x_new[:, 0:1, :, :], x_new[:, n:n + 1, :, :], x_new[:, 2 * n:, :, :]], dim=1)
        r00, var00, L00, beta00 = T_Genetic(y, x_null,
                                            time_points).stat()
        lrt0 = 2 * (L1 - L00).cpu().numpy()
        pvalue = torch.tensor(list(map(lambda x: chi2.sf(x, 2 * n - 2), lrt0)))
        pvalue = pvalue.numpy()
        r = r1.cpu()
        r = r.numpy()
        var = var1.cpu()
        var = var.numpy()
        L1 = L1.cpu()
        L1 = L1.numpy()
        aic = -2 * L1 + 2 * (x_new.shape[1] - 1)
        bic = -2 * L1 + np.log(time_points * y.shape[1]) * (x_new.shape[1] - 1)
        out = {'Variant_ID': snpid, 'Ensembl_ID': pheid, 'Auto-correlation': r, 'Interaction Pvalue': pvalue, 'Loglikelihood': L1, 'Log Likelihood Ratio':lrt0, 'AIC': aic, 'BIC': bic,
               'Variance': var}
        out = pd.DataFrame(out)
        beta1 = pd.DataFrame(beta1.cpu().numpy())
        out = pd.concat([out,beta1],axis=1)
    return out


def r2(y, x, snp, time_points, n):
    y = y.to(device)
    x = x.to(device)
    snp = snp.to(device)
    if n > 0:
        x0 = torch.einsum('ij,kjl->ikjl', snp[:, :, 0], x[0:n, :, :])
        x1 = torch.einsum('ij,kjl->ikjl', snp[:, :, 1], x[0:n, :, :])
        x_temp = torch.cat([x0, x1],
                           dim=1)  
        x_new = torch.ones((x_temp.shape[0], x_temp.shape[1] + x.shape[0], x.shape[1], x.shape[2]))
        x_new = x_new.to(device)
        for i in range(x_new.shape[0]):
            temp = torch.cat((x_temp[i, :, :, :], x), dim=0)
            x_new[i, :, :, :] = temp
        r, var, L1, beta = T_Genetic(y, x_new, time_points).stat()
    else:  # null model
        x_new = torch.ones((snp.shape[0], x.shape[0], x.shape[1], x.shape[2]))
        x_new = x_new.to(device)
        for i in range(x_new.shape[0]):
            x_new[i, :, :, :] = x
        r, var, L1, beta = T_Genetic(y, x_new, time_points).stat()
    y_ = torch.einsum('ij,ijkl->ikl', beta.float(), x_new.float())
    y_ = y_.reshape(y_.shape[0], -1)
    y_ = y_.transpose(0, 1)
    y = y.reshape(y.shape[0], -1)
    y = y.transpose(0, 1)
    tss = torch.sum((y - torch.mean(y, axis=0)) ** 2, axis=0) / y.shape[0]
    rss = torch.sum((y - y_) ** 2, axis=0) / y.shape[0]
    r_square = 1 - (rss / tss)
    r_square = r_square.cpu()
    r_square = r_square.numpy()
    return r_square

def print_error(value):
    print("error:", value)


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
def teQTL_main(cov,y,genotype,time_points,n_process,n_terms,cisDist=None,snpspos=None,genepos=None):
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
    cov_new2 = torch.ones((cov_new.shape[1], int(cov_new.shape[0] / time_points), time_points))  
    for k in range(cov_new.shape[1]):
        vr = cov_new[:, k]
        vr = torch.reshape(vr, (int(cov_new.shape[0] / time_points), time_points))
        cov_new2[k, :, :] = vr

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
    for dt in pp.part_func(cut, cis_snpgeneid, genotype):
        cis_temp = dt[0]
        genotype_temp = dt[1]
        y2 = y.loc[:, cis_temp.loc[:, 'geneid']]
        pheid = y2.columns.values
        y2 = y2.values
        y2 = torch.log2(torch.from_numpy(y2) + 0.01)
        y3 = torch.ones(y2.shape[1], int(y2.shape[0] / time_points), time_points)
        for n in range(y2.shape[1]):
            vr = y2[:, n]
            vr = torch.reshape(vr, (int(y2.shape[0] / time_points), time_points))
            y3[n, :, :] = vr
        # genotype
        snp = torch.ones(cis_temp.shape[0], int(y2.shape[0] / time_points), 2)

        for g in range(cis_temp.shape[0]):
            snpid = cis_temp.loc[:, 'snpid'].values[g]
            temp = genotype_temp[[snpid]]
            temp.columns = ['a']
            vr = encoder.transform(temp).values
            vr = torch.as_tensor(vr)
            snp[g, :, :] = vr

        snpid = cis_temp.loc[:, 'snpid'].values
        last = p.apply_async(teQTL, (cov_new2,y3, snp, time_points, snpid, pheid,n_terms), error_callback=print_error)
        p_out.append(last)
        if len(p._cache) > 1e2:
            last.wait()
    p.close()
    p.join()

    for proce in p_out:
        data.append(proce.get())

    out = reduce(lambda left,right:pd.concat([left,right],axis=0,sort=False),data)
    return out

def dynamic_eQTL_main(cov,y,genotype,time_points,n_process,n_terms,cisDist=None,snpspos=None,genepos=None):
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
    cov_new2 = torch.ones((cov_new.shape[1], int(cov_new.shape[0] / time_points), time_points))
    for k in range(cov_new.shape[1]):
        vr = cov_new[:, k]
        vr = torch.reshape(vr, (int(cov_new.shape[0] / time_points), time_points))
        cov_new2[k, :, :] = vr

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
    for dt in pp.part_func(cut, cis_snpgeneid, genotype):
        cis_temp = dt[0]
        genotype_temp = dt[1]
        y2 = y.loc[:, cis_temp.loc[:, 'geneid']]
        pheid = y2.columns.values
        y2 = y2.values
        y2 = torch.log2(torch.from_numpy(y2) + 0.01)
        y3 = torch.ones(y2.shape[1], int(y2.shape[0] / time_points), time_points)
        for n in range(y2.shape[1]):
            vr = y2[:, n]
            vr = torch.reshape(vr, (int(y2.shape[0] / time_points), time_points))
            y3[n, :, :] = vr
        # genotype
        snp = torch.ones(cis_temp.shape[0], int(y2.shape[0] / time_points), 2)

        for g in range(cis_temp.shape[0]):
            snpid = cis_temp.loc[:, 'snpid'].values[g]
            temp = genotype_temp[[snpid]]
            temp.columns = ['a']
            vr = encoder.transform(temp).values
            vr = torch.as_tensor(vr)
            snp[g, :, :] = vr

        snpid = cis_temp.loc[:, 'snpid'].values
        last = p.apply_async(interac, (cov_new2,y3, snp, time_points, snpid, pheid,n_terms), error_callback=print_error)
        p_out.append(last)
        if len(p._cache) > 1e2:
            last.wait()
    p.close()
    p.join()

    for proce in p_out:
        data.append(proce.get())

    out = reduce(lambda left,right:pd.concat([left,right],axis=0,sort=False),data)
    return out
