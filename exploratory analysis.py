import functions as f
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = f.openfile('data.h5')
f.savefile(data, "data")
df = pd.read_csv('data.csv')
print(df)
print(type(df))
print(type(df.index))
print(type(df.columns))
print(type(df.values))

df.head()
df.info()

df_NetMigr = df[['year', 'Net migration']]
df_NetMigr1960 = df_NetMigr[df_NetMigr['year'] == '1960-01-01']
print(df_NetMigr1960.describe())
plt.figure(figsize=(10, 10))
sns.distplot(df_NetMigr1960['Net migration'], bins=100)
plt.show()

df_1960 = df[df['year'] == '1960-01-01']

print(df_1960.dtypes)

df_num = df_1960.select_dtypes(include=['float64'])
print(df_num)

df_num.hist(figsize=(20, 20), bins=100)
plt.show()

df_num_corr = df_num.corr()['Net migration'][1:5]
golden_features_list = df_num_corr[abs(df_num_corr) > 0.5].sort_values(ascending=False)
print("There is {} strongly correlated values with Net migration:\n{}".format(len(golden_features_list), golden_features_list))

for i in range(1, len(df_num.columns), 5):
    sns.pairplot(data=df_num, x_vars=df_num.columns[i:i+5], y_vars=['Net migration'])
    plt.show()

fig, ax = plt.subplots(5, figsize=(10, 20))
for i, ax in enumerate(fig.axes):
    if i < len(df_num.columns) - 1:
        sns.regplot(x=df_num.columns[i+1], y='Net migration', data=df_num, ax=ax)

plt.show()


