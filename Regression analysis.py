import functions as f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt  # plots
import seaborn as sns  # more plots

from dateutil.relativedelta import relativedelta  # working with dates with style
from scipy.optimize import minimize  # for function minimization

import statsmodels.formula.api as smf  # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs
import datetime
import json
import requests

from itertools import product  # some useful functions
from tqdm import tqdm_notebook

data = f.openfile('data.h5')
dt = data.columns.levels

""" da = list(data.items())
print(da)

dt = data.describe()
print(dt) """





x = data.xs('tas', level = 1, axis = 1).iloc[0,:].values.reshape(-1,1)
y = data.xs('pr', level = 1, axis = 1).iloc[0,:].values.reshape(-1,1)

plt.scatter(x, y, marker='o')
plt.title('pr vs tas 1960')
plt.xlabel('tas')
plt.ylabel('pr')
plt.show()

"""tas = data.xs('tas', level = 1, axis = 1).iloc[0:4,:]
yrs = data.iloc[0:4,:]

print(tas)"""



"""plt.scatter(x, y, marker='o')
plt.title('pr vs tas')
plt.xlabel('tas')
plt.ylabel('pr')
plt.show()"""