import functions as f

countries = f.openfile("iso3.txt")
iso3 = []

for country in countries:
    append = dict.get(country, 'alpha-3')
    iso3.append(append)


data_temperature = f.wbclimate("tas", "year", iso3, export=True)
data_rain = f.wbclimate("pr", "year", iso3, export=True)
data_migration = f.wbdataset('SM.POP.NETM', iso3, 1900, 2019, export=True)
data_arable_land = f.wbdataset('AG.LND.ARBL.ZS',1900,2019, export =True)
data_forest = f.wbdataset('AG.LND.FRST.ZS',1900,2019, export =True)
data_below5_metres = f.wbdataset('AG.LND.EL5M.ZS',1900,2019, export =True)
data_pop_growth = f.wbdataset('SP.POP.GROW',1900,2019, export =True)
data_pop_total = f.wbdataset('SP.POP.TOTL',1900,2019, export =True)
data_greenhouse_gasses = f.wbdataset('EN.ATM.GHGT.KT.CE',1900,2019, export =True)
