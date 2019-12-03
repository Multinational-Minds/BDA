import functions as f

countries = f.openfile("iso3.txt")
iso3 = []

for country in countries:
    append = dict.get(country, 'alpha-3')
    iso3.append(append)


#data = f.wbclimate("tas", "year", iso3, export=True)

#data2 = f.wbdataset('SM.POP.NETM', iso3, 1960, 2012, export=True)


