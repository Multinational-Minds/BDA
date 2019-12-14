'''this file is used to query the API for the needed data'''

import functions as f
import pandas as pd

countries = f.openfile("iso3.txt")
iso3 = []

for country in countries:
    append = dict.get(country, 'alpha-3')
    iso3.append(append)

temperature = f.wbclimate("tas", "year", iso3, export=True, name='temperature')
rain = f.wbclimate("pr", "year", iso3, export=True, name='rain')
migration = f.wbdataset('SM.POP.NETM', iso3, 1960, 2012, export=True, name='migration')
arable_land = f.wbdataset('AG.LND.ARBL.ZS', iso3, 1900, 2012, export=True, name='arable land')
forest = f.wbdataset('AG.LND.FRST.ZS', iso3, 1900, 2012, export=True, name='forest')
pop_growth = f.wbdataset('SP.POP.GROW', iso3, 1900, 2012, export=True, name='population growth')
total_pop = f.wbdataset('SP.POP.TOTL', iso3, 1900, 2012, export=True, name='total population')
greenhouse_gasses = f.wbdataset('EN.ATM.GHGT.KT.CE', iso3, 1900, 2012, export=True, name='greenhouse gasses')
dataset = pd.concat(
    [temperature, rain, migration, arable_land, forest, pop_growth, total_pop, greenhouse_gasses]).sort_index(axis=0)

