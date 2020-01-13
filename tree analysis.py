import os
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydot
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz

import functions as f

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz/release/bin/'

data = f.openfile('data.h5')
copy = data.copy()
data = data.sort_values(by='year').reset_index()
data['year'] = data['year'].apply(lambda x: str(x.year))
data = pd.get_dummies(data)

labels = np.array(data['Net migration'])
features = data.drop('Net migration', axis=1)

features_list = list(features.columns)

copy['year'] = copy['year'].apply(lambda x: str(x.year))
copy['key'] = copy[['country', 'year']].apply(lambda x: ''.join(x), axis=1)

nobs = int(len(data.index) * 3 / 4)
train_features, test_features = features[0:nobs], features[nobs:]
train_labels, test_labels = labels[0:nobs], labels[nobs:]
idx = test_features.index
keys = copy['key'].iloc[idx]

rf = RandomForestRegressor(n_estimators=1000)
fit = rf.fit(train_features, train_labels)

predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
RMSE = sqrt((errors ** 2).mean())
R2 = fit.score(test_features, test_labels)
print('RMSE model full:' + str(RMSE))
print('R2 model full: ', R2)

rf_small = RandomForestRegressor(n_estimators=10, max_depth=3)
rf_small.fit(train_features, train_labels)
tree_small = rf_small.estimators_[5]
export_graphviz(tree_small, out_file='small_tree.dot', feature_names=features_list, rounded=True, precision=1)
(graph,) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png')

feature_importances = pd.DataFrame(rf.feature_importances_, index=features.columns, columns=['importance']).sort_values(
    'importance', ascending=False)
importances = feature_importances[:15]
f.savefile(feature_importances, "rf_feature importances")

fig = importances.plot(kind='barh', y='importance')
plt.show()

rf_most_important = RandomForestRegressor(n_estimators=1000)
important_indices = list(importances[:9].index)
train_important = train_features[important_indices]
test_important = test_features[important_indices]

rf_most_important.fit(train_important, train_labels)
predictions_important = rf_most_important.predict(test_important)
errors_reduced = (predictions_important - test_labels)
RMSE_reduced = sqrt((errors_reduced ** 2).mean())
R2_reduced = fit.score(test_features, test_labels)
print('RMSE model reduced:' + str(RMSE_reduced))
print('R2 model reduced: ', R2_reduced)
print('difference between the two RMSE: ', str(RMSE_reduced - RMSE))

predictions_data = pd.DataFrame({'prediction': predictions, 'key': keys})
true_data = pd.DataFrame({'actual': test_labels, 'key': keys})

plt.plot(true_data['key'], true_data['actual'], 'b-', label='actual')
plt.plot(predictions_data['key'], predictions_data['prediction'], 'ro', label='prediction')

plt.xticks(rotation='60')
plt.legend()
plt.xlabel('key')
plt.ylabel('Net Migration')
plt.title('Actual and Predicted Values')
plt.gca().axes.get_xaxis().set_visible(False)
plt.show()
