import json
import requests


def wbclimate(variable, timescale, countriesList, export= False):
    dataset = []
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
                    entry = dict(country=country, observations=data)
                    dataset.append(entry)
                    countries_done.append(country)
                    remaining = len(countriesList) - len(countries_done)
                    print("done: " + country + " remaining: " + remaining)
                else:
                    response.raise_for_status()
            countries = [x for x in countriesList if x not in countries_done]
        except requests.exceptions.HTTPError:
            countries = [x for x in countriesList if x not in countries_done]
    else:
        if export:
            savefile(dataset, "dataset.json")
        return dataset



def openfile(name):
    with open(name, "r") as file:
        return json.load(file)


def savefile(data, name):
    with open(name, "w") as file:
        json.dump(data, file)
