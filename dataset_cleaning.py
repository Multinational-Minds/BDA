'''This file should be used for cleaning datasets and joining them to make one large pandas dataframe'''
import datetime

import numpy as np
import pandas as pd
import functions as f

countries = f.openfile("iso3.txt")
iso3 = []
for country in countries:
    append = dict.get(country, 'alpha-3')
    iso3.append(append)

arable = f.openfile("arable land.h5")
migration = f.openfile("migration.h5")
pop_growth = f.openfile("population growth.h5")
rain = f.openfile("rain.h5")
temperature = f.openfile("temperature.h5")
total_pop = f.openfile("total population.h5")

varlist = [arable, temperature, pop_growth, rain, total_pop]
dataset = migration
for var in varlist:
    dataset = pd.merge(dataset, var, on=['year', 'country'])
dataset.columns = dataset.columns.str.replace('%', '%25')

data = pd.DataFrame()
for country in iso3:
    append = dataset.loc[dataset['country'] == str(country)]
    append = append.sort_values(by='year')
    append = append.groupby(np.arange(len(append)) // 5).agg(
        {'year': 'first', 'country': 'last', 'Net migration': 'mean', 'tas': 'mean', 'pr': 'mean',
         'Arable land (%25 of land area)': 'mean', 'Population growth (annual %25)': 'mean',
         'Population, total': 'mean'})
    append = append.dropna()
    data = data.append(append, ignore_index=True)

f.savefile(data, "data", csv=False)
