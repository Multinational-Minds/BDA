import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import functions as f


# import data
df = f.openfile("data.h5")



periods = df.year.unique()

fig = plt.figure(figsize=(20, 20))

for c, num in zip(periods, range(1, 12)):

    # subset every period
    features = df[df['year'] == c].drop(columns=['year', 'country'])

    ax = fig.add_subplot(5, 3, num)

    # perform elbow method
    Error = []
    for i in range(1, 15):
        kmeans = KMeans(n_clusters=i).fit(features)
        Error.append(kmeans.inertia_)

    ax.plot(range(1, 15), Error)
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')

plt.tight_layout()
fig.savefig("ElbowMethod.png")

# It appears that the optimal number of clusters for every period is around 3

# perform 3-means clustering for every period

kmeans3 = KMeans(n_clusters=3)


# Kmeans Clustering for every period
fig = plt.figure(figsize=(45, 60))

for c, num in zip(periods, range(1, 12)):
    features = df[df['year'] == c].drop(columns=['year', 'country'])
    y_kmeans3 = kmeans3.fit_predict(features)

    print(c)
    print(y_kmeans3)
    print()

    ax = fig.add_subplot(5, 3, num)
    scatter = ax.scatter(features['Population, total'], features['Net migration'],
                         c=y_kmeans3, s=50)
    ax.set_title('K-Means Clustering ' + str(c))
    ax.set_xlabel('Total population')
    ax.set_ylabel('Net migration')
    plt.colorbar(scatter)

fig.tight_layout()
fig.savefig("KMeansClustering.png")


