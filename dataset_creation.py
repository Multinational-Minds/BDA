'''this file is used to query the API for the needed data'''

import functions as f

countries = f.openfile("iso3.txt")
iso3 = []

for country in countries:
    append = dict.get(country, 'alpha-3')
    iso3.append(append)

temperature = f.wbclimate("tas", "year", iso3, export=True, name='temperature')
rain = f.wbclimate("pr", "year", iso3, export=True, name='rain')
migration = f.wbdataset('SM.POP.NETM', iso3, 1960, 2012, export=True, name='migration')
arable_land = f.wbdataset('AG.LND.ARBL.ZS', iso3, 1900, 2012, export=True, name='arable land')
pop_growth = f.wbdataset('SP.POP.GROW', iso3, 1900, 2012, export=True, name='population growth')
total_pop = f.wbdataset('SP.POP.TOTL', iso3, 1900, 2012, export=True, name='total population')

