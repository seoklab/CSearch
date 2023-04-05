import os,sys
import pandas as pd
df1 = pd.read_csv('energy.csv')
df1 = df1.sort_values(by=['energy'])
df1.to_csv('energy_sorted.csv', index=False)
