'''This file should be used for cleaning datasets and joining them to make one large pandas dataframe'''
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


