'''This file is used to define custom functions'''
import datetime
import json
import pandas as pd
import requests
import statsmodels.api as sm


def wbclimate(variable, timescale, countriesList, export=False, name=''):
    """ retrieve historical climate data

    variable is either tas or pr
    timescale is decade, year or month
    countriesList is a list of iso 3 codes for countries you want to look up
    export = True will export the retrieved data as a csv and will ask for a name in the console
    """
    dataset = pd.DataFrame()
    countries_done = []
    countries = countriesList
    url_base = "http://climatedataapi.worldbank.org/climateweb/rest/v1/country/cru/" + variable + "/" + timescale + "/"
    while len(countries) > 0:
        try:
            for country in countries:
                url = url_base + country
                response = requests.get(url)
                if response.ok:
                    data = json.loads(response.content)
                    templist = []
                    columnlist = []
                    for entry in data:
                        templist.append(float(entry.get("data")))
                        date = datetime.date(year=int(entry.get("year")), month=1, day=1)
                        columnlist.append(date)
                    newdf = pd.DataFrame(
                        {str(variable): templist, 'year': columnlist, 'country': [country] * len(templist)})
                    dataset = dataset.append(newdf)
                    countries_done.append(country)
                    remaining = len(countriesList) - len(countries_done)
                    print("done: ", country, " remaining: ", remaining)
                else:
                    response.raise_for_status()
            countries = [x for x in countriesList if x not in countries_done]
        except requests.exceptions.HTTPError:
            countries = [x for x in countriesList if x not in countries_done]
    else:

        if export:
            if len(name) == 0:
                name = input('file name: ')
            name = name + ".h5"
            dataset.to_hdf(name, key=str(variable), mode='a')
        return dataset


def wbdataset(topic, countriesList="all", startdate=None, enddate=None, export=False, name=''):
    """ retrieve data from world bank

        topic is available on the data viewer of the world bank (e.g. SM.POP.NETM)
        countriesList is a list of iso 3 codes for countries you want to look up
        startdate and enddate are the years in which you want to start retrieving data (e.g. 1960)
        export = True will export the retrieved data as a csv and will ask for a name in the console
        """
    dataset = pd.DataFrame()
    varname = None
    countries_done = []
    countries = countriesList
    url_base = 'http://api.worldbank.org/countries/'
    if startdate is None and enddate is None:
        url_extension = '/indicators/' + topic + "?format=json&per_page=1000"
    elif startdate == enddate:
        url_extension = '/indicators/' + topic + "?date=" + startdate + "&format=json&per_page=1000"
    else:
        url_extension = '/indicators/' + topic + "?date=" + str(startdate) + ":" + str(
            enddate) + "&format=json&per_page=1000"

    while len(countries) > 0:
        for country in countries:
            try:
                url = url_base + country + url_extension
                response = requests.get(url)
                if response.ok:
                    data = json.loads(response.content)
                    data.remove(data[0])

                    templist = []
                    columnlist = []
                    if len(data) > 0 and data[0] is not None:
                        varname = data[0][1].get('indicator').get('value')
                        for list in data:
                            for entry in list:
                                value = entry.get("value")
                                if value is not None:
                                    templist.append(float(value))
                                else:
                                    templist.append(None)
                                date = datetime.date(year=int(entry.get("date")), month=1, day=1)
                                columnlist.append(date)
                    newdf = pd.DataFrame(
                        {str(varname): templist, 'year': columnlist, 'country': [country] * len(templist)})
                    dataset = dataset.append(newdf)
                    countries_done.append(country)
                    remaining = len(countriesList) - len(countries_done)
                    print("done: ", country, " remaining: ", remaining)
                else:
                    response.raise_for_status()
                countries = [x for x in countriesList if x not in countries_done]
            except requests.exceptions.HTTPError:
                countries = [x for x in countriesList if x not in countries_done]
    else:
        if export:
            if len(name) == 0:
                name = input('file name: ')
            name = name + ".h5"
            dataset.to_hdf(name, key=str(varname), mode='a')
        return dataset


def openfile(name):
    if str(name).lower().endswith('.json'):
        with open(name, "r") as file:
            return json.load(file)
    elif str(name).lower().endswith('.csv'):
        with open(name, "r") as file:
            data = pd.read_csv(file)
            return data
    elif str(name).lower().endswith('.txt'):
        with open(name, "r") as file:
            return json.load(file)
    elif str(name).lower().endswith('.h5'):
        data = pd.read_hdf(name)
        return data


def savefile(data, name, csv=True):
    if csv:
        name = name + ".csv"
        data.to_csv(name)
    else:
        name = name + ".h5"
        data.to_hdf(name, key=str(name), mode='a')


def season(data, variable):
    series = data.xs(variable, level=1, axis=1).mean(axis=1)
    idx = series.index.values
    idx = pd.DatetimeIndex(idx)
    frame = {"worldwide " + ': ' + variable: series}
    data = pd.DataFrame(frame)
    data = data.set_index(idx)

    decomposition = sm.tsa.seasonal_decompose(data, model='additive')
    fig = decomposition.plot()
    fig.suptitle("Seasonality: worldwide " + variable)

    fig.savefig('seasonality.png')
