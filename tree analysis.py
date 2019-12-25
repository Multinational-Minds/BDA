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
features['year'] = features['year'].apply(lambda x: x.year)
features = pd.get_dummies(features)
features_list = list(features.columns)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25)

rf = RandomForestRegressor(n_estimators=1000)
fit = rf.fit(train_features, train_labels)

'''
ONLY LET THIS RUN ONCE, IT'S SLOW AS FUCK
tree = rf.estimators_[5]
export_graphviz(tree, out_file='tree.dot', feature_names=features_list, rounded=True, precision=1)
(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')

rf_small = RandomForestRegressor(n_estimators=10, max_depth=3)
rf_small.fit(train_features, train_labels)
tree_small = rf_small.estimators_[5]
export_graphviz(tree_small, out_file='small_tree.dot', feature_names=features_list, rounded=True, precision=1)
(graph,) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png')'''

feature_importances = pd.DataFrame(rf.feature_importances_, index=features.columns, columns=['importance']).sort_values(
    'importance', ascending=False)
importances = feature_importances[:15]
fig = importances.plot(kind='barh', y='importance')
plt.savefig('random_forest features.png')
plt.show()
