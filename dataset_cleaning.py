'''This file should be used for cleaning datasets and joining them to make one large pandas dataframe'''
import datetime

import pandas as pd
import numpy as np
import pandas as pd
import functions as f
import seaborn as sns
import matplotlib.pyplot as plt

arable = f.openfile("arable land.h5")
migration = f.openfile("migration.h5")
pop_growth = f.openfile("population growth.h5")
rain = f.openfile("rain.h5")
temperature = f.openfile("temperature.h5")
total_pop = f.openfile("total population.h5")
dataset = pd.concat(
    [temperature, rain, migration, arable, pop_growth, total_pop], sort=True).sort_index(axis=0)

dataset.drop(dataset.iloc[:, 0:59], inplace=True, axis=1)
year2007 = datetime.date(2007, 1, 1)
dataset[year2007].replace('', np.nan, inplace=True)
dataset.dropna(subset=[year2007], inplace=True)
for index, df in dataset.groupby(level=0):
    if len(df.index) < 5:
        dataset = dataset.drop(index, level=0)

idx = dataset.index.levels[1].str.replace('%', '%25')
dataset.index = dataset.index.set_levels(idx, level=1)
dataset = dataset.transpose()

rownames = list(dataset.index.values)
rownames = rownames[0::5]
count = 5
previouscount = 0
data = pd.DataFrame()
for name in rownames:
    series = dataset[previouscount:count]
    count = count + 5
    previouscount = previouscount + 5
    series = series.mean()
    frame = {name: series}
    tempdf = pd.DataFrame(frame)
    data = pd.concat([data, tempdf], axis=1)

data.replace('', np.nan, inplace=True)
data = data.dropna()
for index, df in data.groupby(level=0):
    if len(df.index) < 6:
        data = data.drop(index, level=0)

data = data.transpose()
f.savefile(data, "data", csv=False)


