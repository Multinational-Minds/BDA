import functions as f
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = f.openfile('data.h5')

'''reading in data, setting target values as y and variables as x
applying the string conversion of years (as they are more categorical rather than numerical in this instance)
creating dummy variables and standardising the independent variables (aka x) and dependent or target variable y'''

y = np.array(data['Net migration'])
x = data.drop('Net migration', axis=1)
x['year'] = x['year'].apply(lambda x: str(x.year))
x = pd.get_dummies(x)
features_list = list(x.columns)
x = StandardScaler().fit_transform(x)
y = y.reshape(-1, 1)
y = StandardScaler().fit_transform(y)

'''Next we define the pca parameters and apply these to the data
initially I chose to asses the two main principal components to plot these on a graph
we store these principal components in the principals dataframe to which we then add the target variable
to obtain the results dataframe which will allow for plotting'''
pca_init = PCA(n_components=2)
pca = pca_init.fit_transform(x)
principals = pd.DataFrame(data=pca, columns=['Principal component 1', 'Principal component 2'])
target = pd.DataFrame(data=y, columns=['Target'])
result = pd.concat([principals, target], axis=1)

'''below we visualise the result dataframe to get a better understanding of our components'''
result.plot.scatter(x='Principal component 1', y='Principal component 2', c='Target', colormap='Blues_r')
plt.show()

'''lastly we print out how much variance is explained by each of the components'''
print('the total explained variance is: ',
      (pca_init.explained_variance_ratio_[0] + pca_init.explained_variance_ratio_[1]) * 100, '%')
print('var PC1: ', pca_init.explained_variance_ratio_[0])

'''As we can see the first two principal components are not able to capture a lot of the variation in the data, neither is the first component so this will be of limited use'''

# Fitting the PCA algorithm with our Data
pca = PCA().fit(x)
# Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')  # for each component
plt.title('Dataset Explained Variance')
plt.show()
