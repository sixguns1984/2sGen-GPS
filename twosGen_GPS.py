"""
Componets of the 2sGen-GPS
"""
from statsmodels.stats.power import FTestPower
from functools import reduce
import numpy as np
import pandas as pd
import sys
sys.path.append('./StQTL_2sGenGPS')
import torch
from scipy.stats import f
from sklearn.linear_model import LinearRegression
from eli5.sklearn import PermutationImportance
import prepare as pp

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.DoubleTensor)


def inv(A, eps=1e-10):  #Inverse matrix calculation
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
    # [U L^{-1}] -> [I U^{-1}L^{-1}] = [I (LU)^{-1}]
    A_inv = L_inv
    for i in range(n - 1, -1, -1):
        A_inv[:, i:i + 1, :] = A_inv[:, i:i + 1, :] / (U[:, i:i + 1, i:i + 1] + eps)
        U[:, i:i + 1, :] = U[:, i:i + 1, :] / (U[:, i:i + 1, i:i + 1] + eps)
        if i > 0:
            A_inv[:, :i, :] = A_inv[:, :i, :] - U[:, :i, i:i + 1].matmul(A_inv[:, i:i + 1, :])
            U[:, :i, :] = U[:, :i, :] - U[:, :i, i:i + 1].matmul(U[:, i:i + 1, :])
    A_inv_grad = - A_inv.matmul(A).matmul(A_inv)
    return A_inv + A_inv_grad - A_inv_grad.data


class LinearRegression_batch(): #Componet of tensor based linear regression for vector auto regression model
    def __init__(self):
        self.x = None
        self.y = None

    def linear(self, x, y):
        k = x.shape[2]
        n = x.shape[0]
        alpha = torch.ones((n, k, k)).to(device)
        for i in range(k):
            temp = torch.einsum('ijk,ij->ijk', x.double(), x[:, :, i].double())
            temp = torch.einsum('ijk->ik', temp)
            alpha[:, i, :] = temp
        ymat = torch.ones((n, k)).to(device)
        for j in range(k):
            x_temp = x[:, :, j].double()
            temp = torch.matmul(x_temp, y.double())
            ymat[:, j] = temp[:, 0]
        beta = torch.einsum('ijk,ik->ijk', inv(alpha), ymat)
        beta = torch.einsum('ijk->ij', beta)
        return beta

    def vector_ar(self, maxlag, x, ylag, y): #Vector Auto Regression model
        y = y.to(device)
        ylag = ylag.to(device)
        x = x.to(device)
        x_new = torch.ones((x.shape[1], ylag.shape[0], maxlag + ylag.shape[1])).to(device)
        for i in range(x.shape[1]):
            temp = torch.cat((ylag, x[:, i].reshape(maxlag, -1).T), dim=1)
            x_new[i, :, :] = temp
        x_null = x_new[:, :, 0:-maxlag]
        beta0 = self.linear(x_null, y)
        beta = self.linear(x_new, y)
        r0 = torch.einsum('ik,ijk->ij', beta0.double(), x_null.double())
        r1 = torch.einsum('ik,ijk->ij', beta.double(), x_new.double())
        sser = torch.sum((y.double().T - r0) ** 2, dim=1)
        ssef = torch.sum((y.double().T - r1) ** 2, dim=1)
        Fvalue = ((sser - ssef) / (x_new.shape[2] - x_null.shape[2])) / (ssef / (y.shape[0] - x_new.shape[2]))  #
        p = f.sf(Fvalue.cpu(), x_new.shape[2] - x_null.shape[2], y.shape[0] - x_new.shape[2])
        sc = np.log(ssef.cpu() / y.shape[0]) + maxlag * np.log(y.shape[0]) / y.shape[0]
        return Fvalue.cpu(), p, beta.cpu(), sc

    def power_test(self, maxlag, x, ylag, y, p_cutoff):  #Power test for significant causality signal
        y = y.to(device)
        ylag = ylag.to(device)
        x = x.to(device)
        x_new = torch.ones((x.shape[1], ylag.shape[0], maxlag + ylag.shape[1])).to(device)
        for i in range(x.shape[1]):
            temp = torch.cat((ylag, x[:, i].reshape(maxlag, -1).T), dim=1)
            x_new[i, :, :] = temp
        x_null = x_new[:, :, 0:-maxlag]
        beta0 = self.linear(x_null, y)
        beta = self.linear(x_new, y)
        r0 = torch.einsum('ik,ijk->ij', beta0.double(), x_null.double())
        r1 = torch.einsum('ik,ijk->ij', beta.double(), x_new.double())
        sser = torch.sum((y.double().T - r0) ** 2, dim=1)
        ssef = torch.sum((y.double().T - r1) ** 2, dim=1)
        Fvalue = ((sser - ssef) / (x_new.shape[2] - x_null.shape[2])) / (ssef / (y.shape[0] - x_new.shape[2]))  #
        p = f.sf(Fvalue.cpu(), x_new.shape[2] - x_null.shape[2], y.shape[0] - x_new.shape[2])
        # power analysis
        tss = torch.sum((y.double().T - torch.mean(y.double().T, dim=1)) ** 2, dim=1)
        r_square0 = 1 - (sser / tss)
        r_square1 = 1 - (ssef / tss)
        f2 = (r_square1 - r_square0) / (1 - r_square1)
        f2 = f2.cpu()
        f2 = f2.numpy()
        f_sqrt = np.sqrt(f2)
        out = []
        for i in f_sqrt:
            out.append(FTestPower().power(effect_size=i, df_num=y.shape[0], df_denom=maxlag, alpha=p_cutoff))
        return out, p, beta.cpu()

    def pred(self, maxlag, x, ylag, y):
        y = y.to(device)
        ylag = ylag.to(device)
        x = x.to(device)
        x_new = torch.ones((x.shape[1], ylag.shape[0], maxlag + ylag.shape[1])).to(device)
        for i in range(x.shape[1]):
            temp = torch.cat((ylag, x[:, i].reshape(maxlag, -1).T), dim=1)
            x_new[i, :, :] = temp
        beta = self.linear(x_new, y)
        r1 = torch.einsum('ik,ijk->ij', beta.double(), x_new.double())
        return r1

# lag_len: lag order, x: independent variable; y:dependent variable;cov_fixed:fixed covariates ID;cov_long:dynamic covariates
def vector_AR_main(lag_len, x, y,cov_fixed,cov_long):
    qtlid = x.columns
    out = []
    for maxlag in range(lag_len):
        lag_sampleid = pd.DataFrame()
        ylag_all = pd.DataFrame()
        participant_id = y['participant_id'].unique()
        maxlag = maxlag + 1
        xlag = pd.DataFrame()
        for lag in range(maxlag):
            lag = lag + 1
            ylag = []
            y_now = []
            for i in range(participant_id.shape[0]):
                temp = y[y['participant_id'] == participant_id[i]].copy()
                temp.sort_values('time', inplace=True)
                temp.drop(labels=temp.index.values[-lag:], inplace=True)
                temp.drop(labels=temp.index.values[0:maxlag - lag], inplace=True)
                ylag.append(temp)
                temp = y[y['participant_id'] == participant_id[i]].copy()
                temp.sort_values('time', inplace=True)
                temp.drop(labels=temp.index.values[0:lag], inplace=True)
                temp.drop(labels=temp.index.values[0:maxlag - lag], inplace=True)
                y_now.append(temp)
            y_now = reduce(lambda left, right: pd.concat([left, right], axis=0, sort=False), y_now)
            ylag = reduce(lambda left, right: pd.concat([left, right], axis=0, sort=False), ylag)
            ylag.index = range(ylag.shape[0])
            ylag_fixed = ylag.loc[:, cov_fixed]
            xlag = pd.concat((xlag, x.loc[ylag['participant_id'] + '_' + ylag['time'].map(str), :]), axis=0)
            ylag_long = ylag.loc[:, cov_long]
            ylag_all = pd.concat((ylag_all, ylag_long), axis=1)
        ylag_all = pd.concat((ylag_fixed, ylag_all), axis=1)
        y_now = y_now.iloc[:, 2:]
        ylag_all = ylag_all.iloc[:, 1:]
        y_now = y_now.values
        ylag_all = ylag_all.values
        x0 = np.ones(y_now.shape[0])
        ylag_all = np.column_stack((x0, ylag_all))
        ylag_all = torch.as_tensor(ylag_all)
        y_now = torch.as_tensor(y_now[:, 0:1])
        cut = 10000
        for n in range(0, xlag.shape[1], cut):
            start2 = n
            stop2 = n + cut if n + cut <= xlag.shape[1] else xlag.shape[1]
            x_temp = xlag.iloc[:, start2:stop2]
            x_temp = torch.as_tensor(x_temp.values)
            qtlid_temp = qtlid[start2:stop2]
            fvalue, p, beta, sc = LinearRegression_batch().vector_ar(maxlag, x_temp, ylag_all, y_now)
            coef = pd.DataFrame(beta.numpy()[:, -maxlag:])
            coef.columns = np.arange(maxlag) + 1
            coef.columns = 'Coef' + coef.columns.map(str)
            adjp = p * qtlid.shape[0]
            adjp[adjp >= 1] = 1
            temp = {'id': qtlid_temp, 'Pvalue': p, 'adjust Pvalue': adjp, 'SC': sc}
            temp = pd.DataFrame(temp)
            temp['AR'] = 'AR' + str(maxlag)
            temp = pd.concat([temp, coef], axis=1)
            out.append(temp)
    out2 = reduce(lambda left, right: pd.concat([left, right], axis=0, sort=False), out)
    temp = out2['id'].str.split('_', expand=True)
    temp.columns = ['Variant_ID', 'Ensembl_ID']
    out2 = pd.concat([temp,out2], axis=1)
    return out2


def residual_cointest(lag_len, qtlid, x, y,cov_fixed,cov_long):#Co-integration test; qtlid: independent variable ID
    test_p = []
    for maxlag in range(lag_len):
        out = []
        ylag_all = pd.DataFrame()
        participant_id = y['participant_id'].unique()
        maxlag = maxlag + 1
        xlag = pd.DataFrame()
        for lag in range(maxlag):
            lag = lag + 1
            ylag = []
            y_now = []
            for i in range(participant_id.shape[0]):
                temp = y[y['participant_id'] == participant_id[i]].copy()
                temp.sort_values('time', inplace=True)
                temp.drop(labels=temp.index.values[-lag:], inplace=True)
                temp.drop(labels=temp.index.values[0:maxlag - lag], inplace=True)
                ylag.append(temp)
                temp = y[y['participant_id'] == participant_id[i]].copy()
                temp.sort_values('time', inplace=True)
                temp.drop(labels=temp.index.values[0:lag], inplace=True)
                temp.drop(labels=temp.index.values[0:maxlag - lag], inplace=True)
                y_now.append(temp)
            y_now = reduce(lambda left, right: pd.concat([left, right], axis=0, sort=False), y_now)
            ylag = reduce(lambda left, right: pd.concat([left, right], axis=0, sort=False), ylag)
            ylag.index = range(ylag.shape[0])
            ylag_fixed = ylag.loc[:, cov_fixed]
            xlag = pd.concat((xlag, x.loc[ylag['participant_id'] + '_' + ylag['time'].map(str), :]), axis=0)
            ylag_long = ylag.loc[:,cov_long]
            ylag_all = pd.concat((ylag_all, ylag_long), axis=1)
        ylag_all = pd.concat((ylag_fixed, ylag_all), axis=1)
        y_now = y_now.iloc[:, 2:]
        ylag_all = ylag_all.iloc[:, 2:]
        y_now = y_now.values
        ylag_all = ylag_all.values
        x0 = np.ones(y_now.shape[0]) 
        ylag_all = np.column_stack((x0, ylag_all))
        ylag_all = torch.as_tensor(ylag_all)
        y_now = torch.as_tensor(y_now[:, 0:1])
        cut = 10000
        for n in range(0, xlag.shape[1], cut):
            start2 = n
            stop2 = n + cut if n + cut <= xlag.shape[1] else xlag.shape[1]
            x_temp = xlag.iloc[:, start2:stop2]
            x_temp = torch.as_tensor(x_temp.values)
            qtlid_temp = qtlid[start2:stop2]
            y_temp = LinearRegression_batch().pred(maxlag, x_temp, ylag_all, y_now)
            y_temp = y_temp.cpu().numpy()
            temp = pd.DataFrame(y_temp.T)
            temp.columns = qtlid_temp + 'AR' + str(maxlag)
            out.append(temp)
        y_pred = reduce(lambda left, right: pd.concat([left, right], axis=1, sort=False), out)
        rd = np.zeros((y_now.shape[0], x.shape[1]))
        for i in range(x.shape[1]):
            residual = y_now.numpy()[:, 0] - y_pred.values[:, i]
            rd[:, i] = residual
        rd = pd.DataFrame(rd, columns=y_pred.columns, index=ylag_fixed.index)
        rd = pd.concat([ylag_fixed.loc[:, ['participant_id', 'time']], rd], axis=1)
        res = pp.adfuller_batch(rd, regression='ct')
        test_p.append(res)
    out2 = reduce(lambda left, right: pd.concat([left, right], axis=0, sort=False), test_p)
    return out2


def power(lag_len, qtlid, x, y, p_cutoff,cov_fixed,cov_long): # power test for each signal
    out = []
    for maxlag in range(lag_len):
        ylag_all = pd.DataFrame()
        participant_id = y['participant_id'].unique()
        maxlag = maxlag + 1
        xlag = pd.DataFrame()
        for lag in range(maxlag):
            lag = lag + 1
            ylag = []
            y_now = []
            for i in range(participant_id.shape[0]):
                temp = y[y['participant_id'] == participant_id[i]].copy()
                temp.sort_values('time', inplace=True)
                temp.drop(labels=temp.index.values[-lag:], inplace=True)
                temp.drop(labels=temp.index.values[0:maxlag - lag], inplace=True)
                ylag.append(temp)
                temp = y[y['participant_id'] == participant_id[i]].copy()
                temp.sort_values('time', inplace=True)
                temp.drop(labels=temp.index.values[0:lag], inplace=True)
                temp.drop(labels=temp.index.values[0:maxlag - lag], inplace=True)
                y_now.append(temp)
            y_now = reduce(lambda left, right: pd.concat([left, right], axis=0, sort=False), y_now)
            ylag = reduce(lambda left, right: pd.concat([left, right], axis=0, sort=False), ylag)
            ylag.index = range(ylag.shape[0])
            ylag_fixed = ylag.loc[:, cov_fixed]
            xlag = pd.concat((xlag, x.loc[ylag['participant_id'] + '_' + ylag['time'].map(str), :]), axis=0)
            ylag_long = ylag.loc[:, cov_long]
            ylag_all = pd.concat((ylag_all, ylag_long), axis=1)
        ylag_all = pd.concat((ylag_fixed, ylag_all), axis=1)
        y_now = y_now.iloc[:, 2:]
        ylag_all = ylag_all.iloc[:, 2:]
        y_now = y_now.values
        ylag_all = ylag_all.values
        x0 = np.ones(y_now.shape[0])  
        ylag_all = np.column_stack((x0, ylag_all))
        ylag_all = torch.as_tensor(ylag_all)
        y_now = torch.as_tensor(y_now[:, 0:1])
        cut = 10000
        for n in range(0, xlag.shape[1], cut):
            start2 = n
            stop2 = n + cut if n + cut <= xlag.shape[1] else xlag.shape[1]
            x_temp = xlag.iloc[:, start2:stop2]
            x_temp = torch.as_tensor(x_temp.values)
            qtlid_temp = qtlid[start2:stop2]
            pwr, p, beta = LinearRegression_batch().power_test(maxlag, x_temp, ylag_all, y_now, p_cutoff)
            coef = pd.DataFrame(beta.numpy()[:, -maxlag:])
            coef.columns = np.arange(maxlag) + 1
            coef.columns = 'Coef' + coef.columns.map(str)
            temp = {'id': qtlid_temp, 'Pvalue': p, 'Power': pwr}
            temp = pd.DataFrame(temp)
            temp['AR'] = 'AR' + str(maxlag)
            temp = pd.concat([temp, coef], axis=1)
            out.append(temp)
    out2 = reduce(lambda left, right: pd.concat([left, right], axis=0, sort=False), out)
    out2['Power'] = out2['Power'].round(2)
    out2.index = out2['id'] + '_' + out2['AR']
    return out2


def importance(qtlid, x, y):# importance analysis
    order = qtlid['AR'].unique()
    maxlag = order.max()
    participant_id = y['participant_id'].unique()
    xlag = pd.DataFrame()
    for m in order:
        qtlid_temp = qtlid.loc[qtlid['AR'] == m]
        for lag in range(m):
            lag = lag + 1
            ylag = []
            y_now = []
            for i in range(participant_id.shape[0]):
                temp = y[y['participant_id'] == participant_id[i]].copy()
                temp.sort_values('time', inplace=True)
                temp.drop(labels=temp.index.values[-lag:], inplace=True)
                temp.drop(labels=temp.index.values[0:maxlag - lag], inplace=True)
                ylag.append(temp)
                temp = y[y['participant_id'] == participant_id[i]].copy()
                temp.sort_values('time', inplace=True)
                temp.drop(labels=temp.index.values[0:lag], inplace=True)
                temp.drop(labels=temp.index.values[0:maxlag - lag], inplace=True)
                y_now.append(temp)
            y_now = reduce(lambda left, right: pd.concat([left, right], axis=0, sort=False), y_now)
            ylag = reduce(lambda left, right: pd.concat([left, right], axis=0, sort=False), ylag)
            ylag.index = range(ylag.shape[0])
            x_temp = x.loc[ylag['participant_id'] + '_' + ylag['time'].map(str), qtlid_temp['id'].values]
            x_temp.columns = qtlid_temp['id'] + '_lag' + str(lag)
            x_temp.index = range(x_temp.shape[0])
            xlag = pd.concat([xlag, x_temp], axis=1)
    y_now = y_now.iloc[:, 2:]
    # features importance rank
    lr = LinearRegression()
    lr.fit(xlag, y_now.iloc[:, 0])
    perm = PermutationImportance(lr, random_state=1).fit(xlag, y_now.iloc[:, 0])
    imp = {'id': xlag.columns, 'importance': perm.feature_importances_}
    imp = pd.DataFrame(imp)
    imp2 = imp.sort_values('importance', ascending=False)

    temp = imp2['id'].str.split('_', expand=True)
    temp.columns = ['Variant_ID', 'Ensembl_ID', 'Delay']
    imp2 = pd.concat([temp,imp2], axis=1)
    imp2.index = imp2['id'].values
    imp2['id'] = imp2['Variant_ID'] + '_' + imp2['Ensembl_ID']
    return imp2