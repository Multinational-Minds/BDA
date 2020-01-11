import pandas as pd
import functions as f
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# import data

data = f.openfile("data.h5")
print(data)
f.savefile(data, "data")

df = pd.read_csv('data.csv')

periods = df.year.unique()

fig = plt.figure(figsize=(20, 20))

for c, num in zip(periods, range(1, 12)):

    # subset every period
    df_year = df[df['year'] == c]
    # select only numeric columns
    features = df_year.iloc[:, 3:9].values

    ax = fig.add_subplot(5, 3, num)

    # perform elbow method
    Error = []
    for i in range(1, 15):
        kmeans = KMeans(n_clusters=i).fit(features)
        kmeans.fit(features)
        Error.append(kmeans.inertia_)

    ax.plot(range(1, 15), Error)

plt.tight_layout()
plt.show()

# It appears that the optimal number of clusters for every year is around 3

# perform 3-means clustering for every period

kmeans3 = KMeans(n_clusters=3)

for c in periods:
    df_year = df[df['year'] == c]
    features = df_year.iloc[:, 3:9].values
    y_kmeans3 = kmeans3.fit_predict(features)

    print(c)
    print(y_kmeans3)
    print()