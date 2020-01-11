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
# from tqdm import tqdm_notebook
import math
import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import KFold
from pylab import rcParams
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

df = f.openfile('data.h5')


'''I thought a dict with all the data per decade would be more useful for you so I went ahead and created that for you'''

df['year'] = df['year'].apply(lambda x: int(x.year))

decades = df['year'].unique()
dataframes = {}
for decade in decades:
    data = df.loc[(df['year'] == decade)]
    dataframes.update({decade: data})

# all data , remove column year and country
df_num = df.drop(columns=['year', 'country'])

# time series data, without countries
df_time = df.drop(columns=['country'])
AFG = df.loc[(df['country'] == 'AFG')].sort_values(
    by='year')
cron = df.sort_values(by='year')

'''We now needto check if there is any autocorrelation (due to this being a time series) left in our data 
if so we will need to use alternative regression methods in order to account for this'''



# VAR TESTING START - Vector Auto regression
# https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/
# Plot 1 line per country

fig, axes = plt.subplots(nrows=3, ncols=2, dpi=120, figsize=(10, 6))
for i, ax in enumerate(axes.flatten()):
    data = df[df.columns[i]]
    ax.plot(data, color='red', linewidth=1)
    # Decorations
    ax.set_title(df.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()

# splits  data for train and test (all data)
# make country specific by opening line #58, and changing nobe line#85 to 2
# again, issue is it looks at everything, not on a country by country basis
nobs = 188
df_train, df_test = df[0:-nobs], df[-nobs:]


def adfuller_test(series, signif=0.05, name='', verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic': round(r[0], 4), 'pvalue': round(r[1], 4), 'n_lags': round(r[2], 4), 'n_obs': r[3]}
    p_value = output['pvalue']

    def adjust(val, length=6):
        return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-' * 47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key, val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")


# runs adfuller test for each variable (all data), returns all data stationary
# if its ran for each country individually, data is non-stationary...
for name, column in df_train.drop(columns=['country']).iteritems():
    adfuller_test(column, name=name)
    print('\n')

model = VAR(pd.get_dummies(df_train))
for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')

model_fitted = model.fit(1)
# print(model_fitted.summary())

# Get the lag order
lag_order = model_fitted.k_ar
print(lag_order)  # > 1

# Input data for forecasting
forecast_input = df_train.values[-lag_order:]

fc = model_fitted.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=df.index[-nobs:], columns=df.columns + '_2d')



def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col) + '_1d'] = (df_train[col].iloc[-1] - df_train[col].iloc[-2]) + df_fc[
                str(col) + '_2d'].cumsum()
        # Roll back 1st Diff
        df_fc[str(col) + '_forecast'] = df_train[col].iloc[-1] + df_fc[str(col) + '_1d'].cumsum()
    return df_fc


df_results = invert_transformation(df_train, df_forecast, second_diff=True)

fig, axes = plt.subplots(nrows=int(len(df.columns) / 2), ncols=2, dpi=150, figsize=(10, 10))
for i, (col, ax) in enumerate(zip(df.columns, axes.flatten())):
    df_results[col + '_forecast'].plot(legend=True, ax=ax).autoscale(axis='x', tight=True)
    df_test[col][-nobs:].plot(legend=True, ax=ax);
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()

# END OF VAR TESTING


# corr matrix and regression analysis for 1960, can loop to make for other years? otherwise manual
df60 = dataframes[1960]
corrmat_partial = df60.drop(columns=['country', 'year']).corr()

corrmat_partial *= np.where(np.tri(*corrmat_partial.shape, k=-1) == 0, np.nan,
                            1)  # puts NaN on upper triangular matrix, including diagonal (k=-1)
corrmat = pd.get_dummies(df60).drop(columns=['year']).corr()
corrmat_list=corrmat.unstack().to_frame()
corrmat_list.columns=['correlation']
corrmat_list['abs_corr']=corrmat_list.correlation.abs()
corrmat_list.sort_values(by=['abs_corr'], ascending=False, na_position='last', inplace=True)
# corrmat_list.drop(columns=['abs_corr']).head(10)

sns.heatmap(corrmat_partial, cmap="YlGnBu", linewidths=0.1)
plt.show()

# regression analysis
# variable distribution and relation to net migration
# all for 1960, can be changed (looped) for other years

X = pd.get_dummies(df60.drop(columns=['Net migration', 'year']))
y = df60['Net migration'].values

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(10,20), constrained_layout=True)
spec = gridspec.GridSpec(nrows=X.shape[1],ncols=2, figure=fig)
'''
for var_index, var in enumerate(X.columns):
    ax_left = fig.add_subplot(spec[var_index, 0])
    sns.distplot(X[var], ax=ax_left)
    ax_left.set_title(var + ' distribution', fontsize=10)
    ax_right = fig.add_subplot(spec[var_index, 1])
    ax_right.scatter(X[var], y, marker='o')
    ax_right.set_title('Net migration vs ' + var, fontsize=10)
    ax_right.set_xlabel(var)
    ax_right.set_ylabel('Net migration')

plt.show()
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)

plt.figure(figsize=(10,6))
sns.distplot(y_train, label='train')
sns.distplot(y_test, label='test')
plt.title('Target variable distribution', fontsize=20)
plt.legend(fontsize=15)
plt.show()

model = LinearRegression()
model.fit(X_train, y_train)

# get coefficients
print('Intercept:', model.intercept_)
print('Slope:', model.coef_)
print(pd.DataFrame({'Variable': ['intercept'] + list(X.columns), 'Coefficient': ["{0:.5f}".format(v) for v in np.append(model.intercept_, model.coef_.flatten()).round(6)]}))

# get fitted value on training set
y_train_predicted = model.predict(X_train)

# compare predictions
print(pd.DataFrame({'True': y_train.flatten(), 'Predicted': y_train_predicted.flatten()}))


# plot marginal models
fig, ax = plt.subplots(math.ceil(X_train.shape[1] / 3), 3, figsize=(20,10), constrained_layout=True)
ax=ax.flatten()

for i, var in enumerate(X_train.columns):
    ax[i].scatter(X_train[var], y_train,  color='gray')
    X_train_univariate=pd.DataFrame(np.zeros(X_train.shape), columns=X_train.columns, index=X_train.index)
    X_train_univariate[var]=X_train[var]
    y_train_predicted_univariate=model.predict(X_train_univariate)
    ax[i].plot(X_train[var], y_train_predicted_univariate + (y_train.mean()-y_train_predicted_univariate.mean()),
               color='red', linewidth=2)
    # y_train.mean()-y_train_predicted_univariate.mean() has been added only to center the line on the points
    # what matters is the slope of the line as the intercept term cannot be "shared" among all univariate variables
    ax[i].set_title('Net migration vs ' + var + ' - corr: ' +\
                       str(round(corrmat_list.loc[(var, 'Net migration'),'correlation']*100)) + '%', fontsize=15)
    ax[i].set_xlabel(var)
    ax[i].set_ylabel('Net migration')
plt.show()

# boxplot showing variability for each variable for 1960/65 and is standardized.
# Again, we can make a loop to run this for dif time periods
def box_plot(df60, standardize=True):
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
box_plot(dataset_outlier)

# simple scatter plot comparing migration and tas 1960/65
# can be changed to show other 1 on 1 relationships
"""x = df60.iloc[:,0].values.reshape(-1,1)
y = df60.iloc[:,1].values.reshape(-1,1)

plt.scatter(x, y, marker='o')
plt.title('Net migration vs tas 1960')
plt.xlabel('Net migration')
plt.ylabel('tas')
plt.show()"""

# K-fold Cross-Validation
# Runs but does not account for time series, uses all data for kfold test/split so is not accurate :(
# useless and bad results but might be able to apply it to useful model
X = cron.drop(columns=['year', 'country', 'Net migration'])
y = cron['Net migration'].to_frame()



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
results, model_list = kFold_CV(X, y, model, n_fold=10)

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

# ARIMA regression model - cant make this one work
"""
lag_acf = acf(df_mig, nlags=20)
lag_pacf = pacf(df_mig, nlags=20, method='ols')

#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_mig)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_mig)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(df_mig)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(df_mig)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
# plt.show()

# AR Model
model = ARIMA(df_mig, order=(2, 1, 0))
results_AR = model.fit(disp=-1)
plt.plot(df_mig)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-df_mig)**2))
plt.show()
"""
