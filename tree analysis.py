import numpy as numpy

import functions as f
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

data = f.openfile('data.h5')
labels = np.array(data['Net Migration'])
features = data.drop('Net Migration', axis=1)
features_list = list(features.columns)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                            random_state=42)
rf = RandomForestRegressor(n=200)
rf.fit(data)
