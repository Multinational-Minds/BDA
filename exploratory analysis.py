import functions as f
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from scipy import stats


data = f.openfile('data.h5')
f.savefile(data, "data")
df = pd.read_csv('data.csv')
print(df)
print(df.columns)
print(df.dtypes)

# remove outliers for future plots
x_nm = np.array(df['Net migration'])
q1_nm = np.quantile(x_nm, 0.25, interpolation='midpoint')
q3_nm = np.quantile(x_nm, 0.75, interpolation='midpoint')
IQR_nm = (q3_nm - q1_nm)

x_tas = np.array(df['tas'])
q1_tas = np.quantile(x_tas, 0.25, interpolation='midpoint')
q3_tas = np.quantile(x_tas, 0.75, interpolation='midpoint')
IQR_tas = (q3_tas - q1_tas)

x_pr = np.array(df['pr'])
q1_pr = np.quantile(x_pr, 0.25, interpolation='midpoint')
q3_pr = np.quantile(x_pr, 0.75, interpolation='midpoint')
IQR_pr = (q3_pr - q1_pr)

x_al = np.array(df['Arable land (%25 of land area)'])
q1_al = np.quantile(x_al, 0.25, interpolation='midpoint')
q3_al = np.quantile(x_al, 0.75, interpolation='midpoint')
IQR_al = (q3_al - q1_al)

x_pg = np.array(df['Population growth (annual %25)'])
q1_pg = np.quantile(x_pg, 0.25, interpolation='midpoint')
q3_pg = np.quantile(x_pg, 0.75, interpolation='midpoint')
IQR_pg = (q3_pg - q1_pg)

x_pt = np.array(df['Population, total'])
q1_pt = np.quantile(x_pt, 0.25, interpolation='midpoint')
q3_pt = np.quantile(x_pt, 0.75, interpolation='midpoint')
IQR_pt = (q3_pt - q1_pt)

df_no_tas = df.loc[(df['Net migration'] < 1.5*IQR_nm) & (df['tas'] < 1.5*IQR_tas)]
df_no_pr = df.loc[(df['Net migration'] < 1.5*IQR_nm) & (df['pr'] < 1.5*IQR_pr)]
df_no_al = df.loc[(df['Net migration'] < 1.5*IQR_nm) & (df['Arable land (%25 of land area)'] < 1.5*IQR_al)]
df_no_pg = df.loc[(df['Net migration'] < 1.5*IQR_nm) & (df['Population growth (annual %25)'] < 1.5*IQR_pg)]
df_no_pt = df.loc[(df['Net migration'] < 1.5*IQR_nm) & (df['Population, total'] < 1.5*IQR_pt)]

periods = df.year.unique()

# plot net immigration vs tas for every period
fig = plt.figure(figsize=(20, 20))

for c, num in zip(periods, range(1, 12)):
    df0 = df_no_tas[df_no_tas['year'] == c]
    ax1 = fig.add_subplot(5, 3, num)
    ax1 = fig.add_subplot(5, 3, num, sharex=ax1, sharey=ax1, xlabel='Temperature', ylabel='Net migration')

    x = df0['tas']
    y = df0['Net migration']

    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    ax1.scatter(x, y, c=z)
    ax1.set_title(c)

plt.tight_layout()
fig.savefig("TempvsNetMigr_AllPeriods.png")

# plot net immigration vs pr for every period
fig = plt.figure(figsize=(20, 20))

for c, num in zip(periods, range(1, 12)):
    df0 = df_no_pr[df_no_pr['year'] == c]
    ax2 = fig.add_subplot(5, 3, num)
    ax2 = fig.add_subplot(5, 3, num, sharex=ax2, sharey=ax2, xlabel='Rain', ylabel='Net migration')

    x = df0['pr']
    y = df0['Net migration']

    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    ax2.scatter(x, y, c=z)
    ax2.set_title(c)

fig.tight_layout()
fig.savefig("RainvsNetMigr_AllPeriods.png")

# plot net immigration vs arable land for every period
fig = plt.figure(figsize=(20, 20))

for c, num in zip(periods, range(1, 12)):
    df0 = df_no_al[df_no_al['year'] == c]
    ax3 = fig.add_subplot(5, 3, num)
    ax3 = fig.add_subplot(5, 3, num, sharex=ax3, sharey=ax3, xlabel='Arable land', ylabel='Net migration')

    x = df0['Arable land (%25 of land area)']
    y = df0['Net migration']

    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    ax3.scatter(x, y, c=z)
    ax3.set_title(c)

plt.tight_layout()
fig.savefig("ArLandvsNetMigr_AllPeriods.png")

# plot net immigration vs population growth for every period
fig = plt.figure(figsize=(20, 20))

for c, num in zip(periods, range(1, 12)):
    df0 = df_no_pg[df_no_pg['year'] == c]
    ax4 = fig.add_subplot(5, 3, num)
    ax4 = fig.add_subplot(5, 3, num, sharex=ax4, sharey=ax4, xlabel='Population growth', ylabel='Net migration')

    x = df0['Population growth (annual %25)']
    y = df0['Net migration']

    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    ax4.scatter(x, y, c=z)
    ax4.set_title(c)

plt.tight_layout()
fig.savefig("PopGrowthvsNetMigr_AllPeriods.png")

# plot net immigration vs total population for every year
fig = plt.figure(figsize=(20, 20))

for c, num in zip(periods, range(1, 12)):
    df0 = df_no_pt[df_no_pt['year'] == c]
    ax5 = fig.add_subplot(5, 3, num)
    ax5 = fig.add_subplot(5, 3, num, sharex=ax5, sharey=ax5, xlabel='Total population', ylabel='Net migration')

    x = df0['Population, total']
    y = df0['Net migration']

    # Calculate the point density
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    ax5.scatter(x, y, c=z)
    ax5.set_title(c)

plt.tight_layout()
fig.savefig("TotPopvsNetMigr_AllPeriods.png")

df.head()

# Change column names for layout purposes
df.columns = ['Unnamed: 0', 'year', 'country', 'Net migration', 'temp', 'rain', 'arable', 'pop growth', 'pop total']
df.head(1)

# correlation heatmap of all periods
fig = plt.figure(figsize=(10, 10))
ax = sns.heatmap(df.drop(['Unnamed: 0'], axis=1).corr(), annot=True, cmap='coolwarm', square=True)
fig.savefig("CorrHeatmap_OverAllPeriods.png")


# correlation heatmap for every period

df_num = df.drop(['Unnamed: 0'], axis=1)
fig = plt.figure(figsize=(45, 60))

for c, num in zip(periods, range(1, 12)):
    df0 = df[df['year'] == c]
    ax = fig.add_subplot(5, 3, num)
    ax = sns.heatmap(df_num[df_num['year'] == c].corr(),
                     annot=True, annot_kws={"size": 25}, cmap='coolwarm')
    ax.set_title(c, fontsize=36)
    for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(20)
                tick.label.set_rotation(90)
    for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(20)
                tick.label.set_rotation(0)

fig.tight_layout()
fig.subplots_adjust(wspace=0.2, hspace=1)
fig.savefig("CorrHeatmap_ForEveryPeriod.png")

# check normal distribution of Net migration for every period
df_NetMigr = df[['year', 'Net migration']]

for period in periods:
    df_NetMigr_period = df_NetMigr[df_NetMigr['year'] == str(period)]
    print(df_NetMigr_period.describe())
    fig = plt.figure(figsize=(10, 10))
    sns.distplot(df_NetMigr_period['Net migration'], bins=100)
    fig.savefig("NormDistrNetMigr_"+str(period)+".png")

    stat, p = stats.shapiro(df_NetMigr_period['Net migration'].values)
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

    # histograms for all variables for every period
    df_period = df[df['year'] == str(period)]

    df_num_period = df_period.select_dtypes(include=['float64'])
    fig, ax = plt.subplots()
    df_num_period.hist(figsize=(20, 20), bins=100, ax=ax)
    plt.tight_layout()
    fig.savefig("HistAllVar_"+str(period)+".png")

    # highest correlations in descending order for every period
    df_num_corr = df_num_period.corr()['Net migration'][1:5]
    golden_features_list = df_num_corr[abs(df_num_corr) > 0.3].sort_values(ascending=False)
    print("There is {} strongly correlated values with Net migration:\n{}".format(len(golden_features_list),
                                                                                  golden_features_list))

    # all variables plotted against Net migration
    for i in range(1, len(df_num_period.columns), 5):
        fig = sns.pairplot(data=df_num_period, x_vars=df_num_period.columns[i:i + 5], y_vars=['Net migration'], height=8,
                    aspect=0.7)
        fig.savefig("AllVarvsNetMigr_"+str(period)+".png")

    # variables plotted against Net migration plus trend line
    fig, ax = plt.subplots(5, figsize=(10, 20))
    for i, ax in enumerate(fig.axes):
        if i < len(df_num_period.columns) - 1:
            sns.regplot(x=df_num_period.columns[i + 1], y='Net migration', data=df_num, ax=ax)
            fig.savefig("TrendAllVarvsNetMigr_"+str(period)+".png")
