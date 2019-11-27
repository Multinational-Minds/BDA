import json
import requests
import pandas as pd



def wbclimate(variable, timescale, countriesList, export=False):
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
                        templist.append(entry.get("data"))
                        columnlist.append(entry.get("year"))
                    newdf = pd.DataFrame([templist], columns=columnlist, index=[country])
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
            name = input('file name: ') + ".csv"
            dataset.to_csv(name)
        return dataset


def wbdataset(topic, countriesList="all", startdate=None, enddate=None):
    dataset = pd.DataFrame()
    countries_done = []
    countries = countriesList
    url = 'http://api.worldbank.org/countries/'
    if startdate is None and enddate is None:
        url_extension = '/indicators/' + topic + "?format=json&per_page=1000"
    elif startdate == enddate:
        url_extension = '/indicators/' + topic + "?date=" + startdate + "&format=json&per_page=1000"
    else:
        url_extension = '/indicators/' + topic + "?date=" + str(startdate) + ":" + str(enddate) + "&format=json&per_page=1000"

    while len(countries) > 0:
        for country in countries:
            try:
                url = url + country+url_extension
                response = requests.get(url)
                if response.ok:
                    data = json.loads(response.content)
                    templist = []
                    columnlist = []
                    for entry in data:
                        templist.append(entry.get("value"))
                        columnlist.append(entry.get("date"))
                    newdf = pd.DataFrame([templist], columns=columnlist, index=[country])
                    dataset = dataset.append(newdf)
                    countries_done.append(country)
                    remaining = len(countriesList) - len(countries_done)
                    print("done: ", country, " remaining: ", remaining)
                else:
                    response.raise_for_status()
                countries = [x for x in countriesList if x not in countries_done]
            except requests.exceptions.HTTPError:
                countries = [x for x in countriesList if x not in countries_done]
    dataset = dataset.transpose()
    return dataset


def openfile(name):
    with open(name, "r") as file:
        return json.load(file)


def savefile(data, name):
    with open(name, "w") as file:
        json.dump(data, file)
