import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from sklearn import metrics
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import functions as f

register_matplotlib_converters()
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

def time_series_cv(X, y, model, date_column, _display=True):
    """Time series nested CV using day forward chaining
    https://towardsdatascience.com/time-series-nested-cross-validation-76adba623eb9"""
    periods = X[date_column].unique()
    columns = pd.get_dummies(X.drop(columns=[date_column])).columns
    performance = list()
    R2_list = list()
    # making train, validate and prediction sets based on subsequent time differences
    for count in range(1, len(periods) - 1):
        X_train = pd.get_dummies(X.loc[X[date_column] <= periods[count - 1]].drop(columns=[date_column])).reindex(
            columns=columns, fill_value=0)
        X_test = pd.get_dummies(X.loc[X[date_column] == periods[count]].drop(columns=[date_column])).reindex(
            columns=columns, fill_value=0)
        y_train = y.iloc[X_train.index]
        y_test = y.iloc[X_test.index]

        # fit model on train set and measure performance
        model.fit(X_train, y_train.values.ravel())
        y_train_predicted = model.predict(X_train)
        R2_train = metrics.r2_score(y_train, y_train_predicted)

        # get performance on test set
        y_test_predicted = model.predict(X_test)
        RMSE_test = np.sqrt(metrics.mean_squared_error(y_test, y_test_predicted))

        # store values to determine performance later
        performance.append(RMSE_test)
        R2_list.append(R2_train)
    # take mean of all the RMSE to obtain almost unbiased test RMSE
    RMSE = sum(performance) / len(performance)
    R2 = sum(R2_list) / len(R2_list)
    result = pd.Series({'Train R^2': R2, 'RMSE model': RMSE})

    if _display:
        print(result)
    return result


def adfuller_test(series, signif=0.05, name='', verbose=False, print=True):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic': round(r[0], 4), 'pvalue': round(r[1], 4), 'n_lags': round(r[2], 4), 'n_obs': r[3]}
    p_value = output['pvalue']

    def adjust(val, length=6):
        return str(val).ljust(length)

    if print:
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
            return ('stat', p_value)
        else:
            print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
            print(f" => Series is Non-Stationary.")
            print('\n')
            return ('non-stat', p_value)
    else:
        if p_value <= signif:
            return 'stat'
        else:
            return 'non-stat', output['test_statistic']


def box_plot(dataframe, standardize=True):
    fig = plt.figure(figsize=(20, 10))

    if standardize == True:
        # standardize columns for better visualization
        idx = dataframe.applymap(lambda x: isinstance(x, str)).all(0)
        dataframe_num = dataframe[dataframe.columns[~idx]]
        dataframe = pd.DataFrame(preprocessing.StandardScaler().fit_transform(dataframe_num.values),
                                 columns=dataframe_num.columns)

    fig = sns.boxplot(x='value', y='variable',
                      data=pd.melt(dataframe.reset_index(), id_vars='index', value_vars=list(dataframe.columns)),
                      orient='h')
    fig.tick_params(labelsize=20)
    fig.set_xlabel('')
    fig.set_ylabel('')
    if standardize == True:
        fig.set_title('Standardized Variable Distribution\nfor better visualization', fontsize=40)
    dataframe = pd.plotting.register_matplotlib_converters()
    plt.show()


def kFold_CV(X, y, model, n_fold, _display=True):
    # generate folds
    folds = KFold(n_splits=n_fold, random_state=0, shuffle=True)

    # fit model on each k-1 fold and evaluate performances (errors)
    columns = ['Split', 'Train size', 'Test size', 'Train R^2', 'Train RMSE', 'Test RMSE']
    results = pd.DataFrame()

    plot_count = 1
    split_count = 1
    model_list = {}
    for train_index, test_index in folds.split(X, y):
        # define train and test (validation) set
        X_split_train = X.iloc[train_index, :]
        X_split_test = X.iloc[test_index, :]
        y_split_train = y.iloc[train_index]
        y_split_test = y.iloc[test_index]

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
        to_append = pd.DataFrame([[split_count, X_split_train.shape[0], X_split_test.shape[0], R2_train,
                                   RMSE_train, RMSE_test]],
                                 columns=columns)
        results = results.append(to_append)
        split_count += 1
        plot_count += 1

    results['Split'] = results['Split'].astype(int)
    results['Train size'] = results['Train size'].astype(int)
    results['Test size'] = results['Test size'].astype(int)
    results = results.reset_index(drop=True)
    mean = results[['Train R^2', 'Train RMSE', 'Test RMSE']].mean()
    mean = mean.append(results[['Split', 'Train size', 'Test size']].iloc[-1])

    if _display == True:
        print(mean)

    return mean, model_list


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


def vif(exogs, data):
    vif_dict, tolerance_dict = {}, {}

    # form input data for each exogenous variable
    for exog in exogs:
        not_exog = [i for i in exogs if i != exog]
        X, y = data[not_exog], data[exog]

        # extract r-squared from the fit
        r_squared = LinearRegression().fit(X, y).score(X, y)

        # calculate VIF
        vif = 1 / (1 - r_squared)
        vif_dict[exog] = vif

        # calculate tolerance
        tolerance = 1 - r_squared
        tolerance_dict[exog] = tolerance

    # return VIF DataFrame
    df_vif = pd.DataFrame({'VIF': vif_dict, 'Tolerance': tolerance_dict})

    return df_vif


df = f.openfile('data.h5')
df['year'] = df['year'].apply(lambda x: int(x.year))
decades = df['year'].unique()
dataframes = {}
for decade in decades:
    data = df.loc[(df['year'] == decade)]
    dataframes.update({decade: data})

# time series data, without countries

cron = df.sort_values(by='year')
cron['year'] = cron['year'].apply(lambda x: str(x))
df['year'] = df['year'].apply(lambda x: str(x))

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
data = cron[cron.country != 'SRB']
data = pd.get_dummies(data.drop(columns=['year']))
df_train, df_test = data[0:-nobs], data[-nobs:]

# vif testing
variables = data.columns
vif_df = vif(variables, data)
print(vif_df)

# runs adfuller test for each variable (all data), returns all data stationary

checkup = list()
non_stationaries = list()
diff = 0

for name, column in df_train.iteritems():
    res = adfuller_test(column, name=name, print=False)
    if res[0] == 'non-stat':
        non_stationaries.append({(name + str(diff)): res[1]})

df_differenced = df_train

while len(non_stationaries) > 0:
    checkup.append(non_stationaries)
    non_stationaries = list()
    diff = diff + 1
    df_differenced = df_differenced.diff().dropna()
    for name, column in df_differenced.iteritems():
        res = adfuller_test(column, name=name, print=False)
        if res[0] == 'non-stat':
            non_stationaries.append({(name + str(diff)): res[1]})

print('final order of differences: ', diff)

model = VAR(df_train)
try:
    for i in range(1, 9):
        result = model.fit(i)
        print('Lag Order =', i)
        print('AIC : ', result.aic)
        print('BIC : ', result.bic)
        print('FPE : ', result.fpe)
        print('HQIC: ', result.hqic, '\n')
except OverflowError:
    print('FPE too large, did not want to mess with code in package')

model_fitted = model.fit(4)
# print(model_fitted.summary())

# Get the lag order
lag_order = model_fitted.k_ar
print(lag_order)  # > 1

# Input data for forecasting
forecast_input = df_differenced.values[-lag_order:]

fc = model_fitted.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=data.index[-nobs:], columns=data.columns + '_1d')

df_results = invert_transformation(df_train, df_forecast)

fig, axes = plt.subplots(nrows=int(len(data.columns) / 2), ncols=2, dpi=150, figsize=(10, 10))
for i, (col, ax) in enumerate(zip(data.columns, axes.flatten())):
    df_results[col + '_forecast'].plot(legend=True, ax=ax).autoscale(axis='x', tight=True)
    df_test[col][-nobs:].plot(legend=True, ax=ax)
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()
plt.show()

time_series_cv(model_fitted.exog, model_fitted.endog, model_fitted)
sys.exit(0)

# END OF VAR TESTING

# corr matrix and regression analysis
df_linreg = df
corrmat_partial = df_linreg.drop(columns=['country', 'year']).corr()

corrmat_partial *= np.where(np.tri(*corrmat_partial.shape, k=-1) == 0, np.nan,
                            1)  # puts NaN on upper triangular matrix, including diagonal (k=-1)
corrmat = pd.get_dummies(df_linreg).corr()
corrmat_list = corrmat.unstack().to_frame()
corrmat_list.columns = ['correlation']
corrmat_list['abs_corr'] = corrmat_list.correlation.abs()
corrmat_list.sort_values(by=['abs_corr'], ascending=False, na_position='last', inplace=True)

sns.heatmap(corrmat_partial, cmap="YlGnBu", linewidths=0.1)
plt.show()

# regression analysis
# variable distribution and relation to net migration
# all for 1960, can be changed (looped) for other years

X = pd.get_dummies(df_linreg.drop(columns=['Net migration']))
y = df_linreg['Net migration']

fig = plt.figure(figsize=(10, 20), constrained_layout=True)
spec = gridspec.GridSpec(nrows=X.shape[1], ncols=2, figure=fig)

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)

plt.figure(figsize=(10, 6))
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
print(pd.DataFrame({'Variable': ['intercept'] + list(X.columns), 'Coefficient': ["{0:.5f}".format(v) for v in
                                                                                 np.append(model.intercept_,
                                                                                           model.coef_.flatten()).round(
                                                                                     6)]}))

# get fitted value on training set
y_train_predicted = model.predict(X_train)

# compare predictions
print(pd.DataFrame({'True': y_train.ravel(), 'Predicted': y_train_predicted.ravel()}))

# deleting outliers
Q1 = df_linreg.quantile(0.2)
Q3 = df_linreg.quantile(0.8)
IQR = Q3 - Q1

dataset_outlier = df_linreg[~((df_linreg < (Q1 - IQR)) | (df_linreg > (Q3 + IQR))).any(axis=1)]
print('\nData size reduced from {} to {}\n'.format(df_linreg.shape[0], dataset_outlier.shape[0]))
box_plot(dataset_outlier)

#applying the linear regression model
model = LinearRegression()
X = df_linreg.drop(columns=['Net migration'])
y = df_linreg['Net migration']

#CV in order to obtain performance metrics
time_series_cv(X, y, model, date_column='year')
