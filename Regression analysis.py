import functions as f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns

from dateutil.relativedelta import relativedelta  # working with dates with style
from scipy.optimize import minimize  # for function minimization

import statsmodels.formula.api as smf  # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import datetime
import json
import requests

import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import preprocessing
from itertools import product
from tqdm import tqdm_notebook

dataset = f.openfile('data.h5')


# correlation matrix between variables, seems to work but i cannot isolate the years so it includes all the data
"""corrmat = dataset.corr()
# shape correlation matrix in key-values pairs
corrmat *= np.where(np.tri(*corrmat.shape, k=-1)==0, np.nan, 1)  # puts NaN on upper triangular matrix, including diagonal (k=-1)
corrmat_list=corrmat.unstack().to_frame()
# highlight highest correlations
corrmat_list.columns=['correlation']
corrmat_list['abs_corr']=corrmat_list.correlation.abs()
corrmat_list.sort_values(by=['abs_corr'], ascending=False, na_position='last', inplace=True)
corrmat_list.drop(columns=['abs_corr']).head(10)"""

"""sns.heatmap(corrmat, cmap ="YlGnBu", linewidths = 0.1)
plt.show()"""

# doesnt work with any objects other than float and i cant change the data from datetime.date
"""def box_plot(df, standardize=True):
    fig = plt.figure(figsize=(20, 10))

    if standardize == True:
        # standardize columns for better visualization
        df = pd.DataFrame(preprocessing.StandardScaler().fit_transform(df.values), columns=df.columns)
    fig = sns.boxplot(x='value', y='variable',
                      data=pd.melt(df.reset_index(), id_vars='index', value_vars=list(df.columns)),
                      orient='h')
    fig.tick_params(labelsize=20)
    fig.set_xlabel('')
    fig.set_ylabel('')
    if standardize == True:
        fig.set_title('Note that variables are standardized\nfor better visualization', fontsize=40)
    df = pd.plotting.register_matplotlib_converters()
    plt.show()

Q1 = dataset.quantile(0.2)
Q3 = dataset.quantile(0.8)
IQR = Q3 - Q1

dataset_outlier = dataset[~((dataset < (Q1 - IQR)) |(dataset > (Q3 + IQR))).any(axis=1)]
print('\nData size reduced from {} to {}\n'.format(dataset.shape[0], dataset_outlier.shape[0]))
box_plot(dataset_outlier)"""



""" dt = data.columns.levels"""

"""da = list(data.items())
print(da)

dt = dataset.dtypes
print(dt)"""




""" x = data.xs('tas', level = 1, axis = 1).iloc[0,:].values.reshape(-1,1)
y = data.xs('pr', level = 1, axis = 1).iloc[0,:].values.reshape(-1,1)

plt.scatter(x, y, marker='o')
plt.title('pr vs tas 1960')
plt.xlabel('tas')
plt.ylabel('pr')
plt.show() """

"""tas = data.xs('tas', level = 1, axis = 1).iloc[0:4,:]
yrs = data.iloc[0:4,:]

print(tas)"""



"""plt.scatter(x, y, marker='o')
plt.title('pr vs tas')
plt.xlabel('tas')
plt.ylabel('pr')
plt.show()"""

"""X = dataset.drop(columns=['Net migration', 'year', 'country'])
y = dataset['Net migration'].values
print(X.shape)
print(y.shape)

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(10,20), constrained_layout=True)
spec = gridspec.GridSpec(nrows=X.shape[1],ncols=2, figure=fig)  # allows to use grid location for subplots

for var_index, var in enumerate(X.columns):
    ax_left = fig.add_subplot(spec[var_index, 0])
    sns.distplot(X[var], ax=ax_left)
    ax_left.set_title(var + ' distribution', fontsize=15)
    ax_right = fig.add_subplot(spec[var_index, 1])
    ax_right.scatter(X[var], y, marker='o')
    ax_right.set_title('Net migration vs ' + var + ' - corr: ' +\
                       str(round(corrmat_list.loc[(var, 'Net migration'),'correlation']*100)) + '%', fontsize=15)
    ax_right.set_xlabel(var)
    ax_right.set_ylabel('Net migration')

plt.show()"""