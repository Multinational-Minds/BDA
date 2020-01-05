import functions as f
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.tree import export_graphviz
import pydot
import matplotlib.pyplot as plt
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/graphviz/release/bin/'

data = f.openfile('data.h5')

labels = np.array(data['Net migration'])
features = data.drop('Net migration', axis=1)
features['year'] = features['year'].apply(lambda x: str(x.year))
features = pd.get_dummies(features)
features_list = list(features.columns)

data['year'] = data['year'].apply(lambda x: str(x.year))
data['key'] = data[['country', 'year']].apply(lambda x: ''.join(x), axis=1)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25)
idx = test_features.index
keys = data['key'].iloc[idx]

rf = RandomForestRegressor(n_estimators=1000)
fit = rf.fit(train_features, train_labels)

predictions = rf.predict(test_features)
errors = abs(predictions - test_labels)
MSE = (errors ** 2).mean()
print('Mean Square Error model full:' + str(MSE))

'''
ONLY LET THIS RUN ONCE, IT'S SLOW AS FUCK
tree = rf.estimators_[5]
export_graphviz(tree, out_file='tree.dot', feature_names=features_list, rounded=True, precision=1)
(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')'''

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
lt.show()

rf_most_important = RandomForestRegressor(n_estimators=1000)
important_indices = list(importances[:9].index)
train_important = train_features[important_indices]
test_important = test_features[important_indices]

rf_most_important.fit(train_important, train_labels)
predictions_important = rf_most_important.predict(test_important)
errors_reduced = (predictions_important - test_labels)
MSE_reduced = (errors_reduced ** 2).mean()
print('Mean Square Error model reduced:' + str(MSE_reduced))
print('difference between the two MSE: ', str(MSE - MSE_reduced))

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
