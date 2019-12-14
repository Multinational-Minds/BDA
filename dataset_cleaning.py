'''This file should be used for cleaning datasets and joining them to make one large pandas dataframe'''
import datetime

from numpy.ma import count

import pandas as pd
import numpy as np
import functions as f

arable = f.openfile("arable land.h5")
forest = f.openfile("forest.h5")
greenhouse_gasses = f.openfile("greenhouse gasses.h5")
migration = f.openfile("migration.h5")
pop_growth = f.openfile("population growth.h5")
rain = f.openfile("rain.h5")
temperature = f.openfile("temperature.h5")
total_pop = f.openfile("total population.h5")
dataset = pd.concat(
    [temperature, rain, migration, arable, pop_growth, total_pop]).sort_index(axis=0)
"f.season(arable.transpose())"
'''f.savefile(arable,"arrrrrable", csv=True)
sm.graphics.tsa.plot_acf(arable, lags=40)
plot_acf(arable.transpose())  '''
dataset.drop(dataset.iloc[:, 0:59], inplace=True, axis=1)
year2007 = datetime.date(2007, 1, 1)
dataset[year2007].replace('', np.nan, inplace=True)
dataset.dropna(subset=[year2007], inplace=True)
for index, df in dataset.groupby(level=0):
    if len(df.index) < 5:
        dataset = dataset.drop(index, level=0)
f.savefile(dataset, "test data", csv=True)
