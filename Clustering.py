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
plt.show()

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

    # K means Clustering
    def doKmeans(X, nclust=3):
        model = KMeans(nclust)
        model.fit(X)
        clust_labels = model.predict(X)
        cent = model.cluster_centers_
        return (clust_labels, cent)


    clust_labels, cent = doKmeans(features, 3)
    kmeans = pd.DataFrame(clust_labels)
    df_year.insert((df_year.shape[1]), 'kmeans', kmeans)

    # Plot the clusters obtained using k means
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(df_year['tas'], df_year['Net migration'],
                         c=kmeans[0], s=50)
    ax.set_title('K-Means Clustering ' + str(c))
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Net migration')
    plt.colorbar(scatter)
    plt.show()
