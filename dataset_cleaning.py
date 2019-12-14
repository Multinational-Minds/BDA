'''This file should be used for cleaning datasets and joining them to make one large pandas dataframe'''
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm

import pandas as pd
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
    [temperature, rain, migration, arable, forest, pop_growth, total_pop, greenhouse_gasses]).sort_index(axis=0)
"f.season(arable.transpose())"
'''f.savefile(arable,"arrrrrable", csv=True)
sm.graphics.tsa.plot_acf(arable, lags=40)
plot_acf(arable.transpose())  '''

dataset1 = dataset.drop(dataset.iloc[:,0:59], inplace = True, axis = 1)
f.savefile(dataset, "1960onwards data", csv = True)




