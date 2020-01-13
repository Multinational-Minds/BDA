import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import functions as f


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
fig.savefig("ElbowMethod.png")

# It appears that the optimal number of clusters for every period is around 3

# perform 3-means clustering for every period

kmeans3 = KMeans(n_clusters=3)

for c in periods:
    df_year = df[df['year'] == c]
    features = df_year.iloc[:, 3:9].values
    y_kmeans3 = kmeans3.fit_predict(features)

    print(c)
    print(y_kmeans3)
    print()

# Kmeans Clustering for every period
fig = plt.figure(figsize=(45, 60))

for c, num in zip(periods, range(1, 12)):
    df0 = df[df['year'] == c]
    ax = fig.add_subplot(5, 3, num)
    scatter = ax.scatter(df_year['Population, total'], df_year['Net migration'],
                         c=y_kmeans3, s=50)
    ax.set_title('K-Means Clustering ' + str(c))
    ax.set_xlabel('Total population')
    ax.set_ylabel('Net migration')
    plt.colorbar(scatter)

fig.tight_layout()
fig.savefig("KMeansClustering.png")


