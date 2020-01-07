import functions as f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns


from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize

import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import datetime
import json
import requests
from itertools import product
from tqdm import tqdm_notebook
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import KFold


df = f.openfile('data.h5')

"""df_1960 = df[df['year'] == '1960-01-01']
df_num = df_1960.select_dtypes(include=['float64'])"""

# data selection of specific years
df['year'] = pd.to_datetime(df['year'])
df_6065 = (df['year'] >= '1960-01-01') & (df['year'] < '1970-01-01')
df1960 = df.loc[df_6065]
df60 = df.loc[df_6065].iloc[:,2:7]

# all data , remove column year and country
df_num = df.drop(columns=['year', 'country'])

# time series data, without countries
df_time = df.drop(columns=['country'])

# data in cronological order 1960-2010
cron = df.sort_values(by='year')


# corr matrix for 1960/65, can loop to make for other years? otherwise manual
"""corrmat = df1960.corr()
# shape correlation matrix in key-values pairs
corrmat *= np.where(np.tri(*corrmat.shape, k=-1)==0, np.nan, 1)  # puts NaN on upper triangular matrix, including diagonal (k=-1)
corrmat_list=corrmat.unstack().to_frame()
corrmat_list.columns=['correlation']
corrmat_list['abs_corr']=corrmat_list.correlation.abs()
corrmat_list.sort_values(by=['abs_corr'], ascending=False, na_position='last', inplace=True)
corrmat_list.drop(columns=['abs_corr']).head(10)

sns.heatmap(corrmat, cmap ="YlGnBu", linewidths = 0.1)
plt.show()"""


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


# K-fold Cross-Validation
# Runs but does not account for time series, uses all data for kfold test/split so is not accurate :(
# useless and bad results but might be able to apply it to useful model

"""X = cron.drop(columns=['year', 'country', 'Net migration'])
y = cron['Net migration'].to_frame()

model = LinearRegression()
n_fold = 5


def kFold_CV(X, y, model, n_fold, _display=True):
    # generate folds
    folds = KFold(n_splits=n_fold, random_state=0, shuffle=True)

    # fit model on each k-1 fold and evaluate performances (errors)
    results = pd.DataFrame(columns=['Split', 'Train size', 'Test size', 'Train R^2', 'Train RMSE', 'Test RMSE'],
                           dtype=float).fillna(0)

    fig = plt.figure(figsize=(10, 1.5 * n_fold))
    plot_count = 1
    split_count = 1
    model_list = {}
    for train_index, test_index in folds.split(X, y):
        # define train and test (validation) set
        X_split_train = X.iloc[train_index, :]
        X_split_test = X.iloc[test_index, :]
        y_split_train = y.iloc[train_index, :]
        y_split_test = y.iloc[test_index, :]

        # plot target variable distribution comparison between split_train and split_test set
        ax = fig.add_subplot(math.ceil(n_fold / 3), 3, plot_count)
        sns.distplot(y_split_train, label='train', ax=ax)
        sns.distplot(y_split_test, label='test', ax=ax)
        ax.set_title('Target variable distribution\nsplit ' + str(split_count), fontsize=12)
        ax.legend(fontsize=8)

        # fit model on train set and get performances on train set
        model_fit = model.fit(X_split_train, y_split_train.values.ravel())
        y_train_predicted = model.predict(X_split_train)
        R2_train = metrics.r2_score(y_split_train, y_train_predicted)
        RMSE_train = np.sqrt(metrics.mean_squared_error(y_split_train, y_train_predicted))
        model_list['split_' + str(split_count)] = model_fit

        # get performance on test set
        y_test_predicted = model.predict(X_split_test)
        RMSE_test = np.sqrt(metrics.mean_squared_error(y_split_test, y_test_predicted))

        # append results
        results = results.append(pd.DataFrame([[split_count, X_split_train.shape[0], X_split_test.shape[0], R2_train,
                                                RMSE_train, RMSE_test]],
                                              columns=results.columns))
        split_count += 1
        plot_count += 1

    results['Split'] = results['Split'].astype(int)
    results['Train size'] = results['Train size'].astype(int)
    results['Test size'] = results['Test size'].astype(int)

    if _display == True:
        plt.tight_layout()
        fig.subplots_adjust(top=0.88)
        plt.show()
        print(results)
    else:
        plt.close()

    return results, model_list


model = LinearRegression()
results, model_list = kFold_CV(X, y, model, n_fold=5)"""


# Time series train test split, only runs for one variable
"""X = cron['Net migration']
splits = TimeSeriesSplit(n_splits=3)
plt.figure(1)
index = 1
for train_index, test_index in splits.split(X):
	train = X[train_index]
	test = X[test_index]
	print('Observations: %d' % (len(train) + len(test)))
	print('Training Observations: %d' % (len(train)))
	print('Testing Observations: %d' % (len(test)))
	plt.subplot(310 + index)
	plt.plot(train)
	plt.plot([None for i in train] + [x for x in test])
	index += 1
plt.show()"""