'''this file is used to query the API for the needed data'''

import functions as f

countries = f.openfile("iso3.txt")
iso3 = []

for country in countries:
    append = dict.get(country, 'alpha-3')
    iso3.append(append)

#temperature = f.wbclimate("tas", "year", iso3, export=True)
#rain = f.wbclimate("pr", "year", iso3, export=True)
migration = f.wbdataset('SM.POP.NETM', iso3, 1900, 2019, export=True)
arable_land = f.wbdataset('AG.LND.ARBL.ZS', iso3, 1900, 2019, export=True)
forest = f.wbdataset('AG.LND.FRST.ZS', iso3, 1900, 2019, export=True)
pop_growth = f.wbdataset('SP.POP.GROW', iso3, 1900, 2019, export=True)
pop_total = f.wbdataset('SP.POP.TOTL', iso3, 1900, 2019, export=True)
greenhouse_gasses = f.wbdataset('EN.ATM.GHGT.KT.CE', iso3, 1900, 2019, export=True)
