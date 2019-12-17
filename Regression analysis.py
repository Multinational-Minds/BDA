import functions as f
import pandas as pd

data = f.openfile('data.csv')

stacked = data.stack()

f.savefile(stacked, 'test', csv=True)