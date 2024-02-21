"""
Example of Augmented Dickey-Fuller (ADF) unit root test
"""
import pandas as pd
import prepare as pp

path1 = './covariates.csv'
path2 = './longitudinal.gene.expression.csv'

cov = pd.read_csv(path1, header=0, index_col=0)
expression = pd.read_csv(path2, header=0, index_col=0)

cov.index = expression.index
expression = pd.concat([cov.loc[:,['participant_id','time']],expression],axis=1)
res = pp.adfuller_batch(expression)
