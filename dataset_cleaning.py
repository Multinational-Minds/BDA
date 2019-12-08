'''This file should be used for cleaning datasets and joining them to make one large pandas dataframe'''

import functions as f

arable = f.openfile("arable land.csv")
below5 = f.openfile("below5.csv")
forest = f.openfile("forest.csv")
greenhouse_gasses = f.openfile("greenhouse gasses.csv")
migration = f.openfile("migration.csv")
pop_growth = f.openfile("population growth.csv")
rain = f.openfile("rain.csv")
temp = f.openfile("temp.csv")
total_pop = f.openfile("total population.csv")

f.season(rain)

