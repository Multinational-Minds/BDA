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
from itertools import product                    # some useful functions
from tqdm import tqdm_notebook
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import preprocessing
from itertools import product
from tqdm import tqdm_notebook


df = f.openfile('data.h5')

"""df_1960 = df[df['year'] == '1960-01-01']
df_num = df_1960.select_dtypes(include=['float64'])"""

"""data selection of specific years"""
df['year'] = pd.to_datetime(df['year'])
df_6065 = (df['year'] >= '1960-01-01') & (df['year'] < '1970-01-01')
df1960 = df.loc[df_6065]
df60 = df.loc[df_6065].iloc[:,2:7]


# corr matrix for 1960/65, can loop to make for other years? otherwise manual
corrmat = df1960.corr()
# shape correlation matrix in key-values pairs
corrmat *= np.where(np.tri(*corrmat.shape, k=-1)==0, np.nan, 1)  # puts NaN on upper triangular matrix, including diagonal (k=-1)
corrmat_list=corrmat.unstack().to_frame()
corrmat_list.columns=['correlation']
corrmat_list['abs_corr']=corrmat_list.correlation.abs()
corrmat_list.sort_values(by=['abs_corr'], ascending=False, na_position='last', inplace=True)
corrmat_list.drop(columns=['abs_corr']).head(10)

sns.heatmap(corrmat, cmap ="YlGnBu", linewidths = 0.1)
plt.show()


# boxplot showing variability for each variable for 1960/65 and is standardized.
# Again, we can make a loop to run this for dif time periods
"""def box_plot(df60, standardize=True):
    fig = plt.figure(figsize=(20, 10))

    if standardize == True:
        # standardize columns for better visualization
        df60 = pd.DataFrame(preprocessing.StandardScaler().fit_transform(df60.values), columns=df60.columns)
    fig = sns.boxplot(x='value', y='variable',
                      data=pd.melt(df60.reset_index(), id_vars='index', value_vars=list(df60.columns)),
                      orient='h')
    fig.tick_params(labelsize=20)
    fig.set_xlabel('')
    fig.set_ylabel('')
    if standardize == True:
        fig.set_title('Standardized Variable Distribution\nfor better visualization', fontsize=40)
    df60 = pd.plotting.register_matplotlib_converters()
    plt.show()

Q1 = df60.quantile(0.2)
Q3 = df60.quantile(0.8)
IQR = Q3 - Q1

dataset_outlier = df60[~((df60 < (Q1 - IQR)) |(df60 > (Q3 + IQR))).any(axis=1)]
print('\nData size reduced from {} to {}\n'.format(df60.shape[0], dataset_outlier.shape[0]))
box_plot(dataset_outlier)"""

# simple scatter plot comparing migration and tas 1960/65
# can be changed to show other 1 on 1 relationships
"""x = df60.iloc[:,0].values.reshape(-1,1)
y = df60.iloc[:,1].values.reshape(-1,1)

plt.scatter(x, y, marker='o')
plt.title('Net migration vs tas 1960')
plt.xlabel('Net migration')
plt.ylabel('tas')
plt.show()"""


# variable distribution and relation to net migration
# all for 1960/65, can be changed (looped) for other years
"""X = df60.drop(columns=['Net migration'])
y = df60['Net migration'].values

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(10,20), constrained_layout=True)
spec = gridspec.GridSpec(nrows=X.shape[1],ncols=2, figure=fig)  # allows to use grid location for subplots

for var_index, var in enumerate(X.columns):
    ax_left = fig.add_subplot(spec[var_index, 0])
    sns.distplot(X[var], ax=ax_left)
    ax_left.set_title(var + ' distribution', fontsize=10)
    ax_right = fig.add_subplot(spec[var_index, 1])
    ax_right.scatter(X[var], y, marker='o')
    ax_right.set_title('Net migration vs ' + var, fontsize=10)
    ax_right.set_xlabel(var)
    ax_right.set_ylabel('Net migration')

plt.show()"""