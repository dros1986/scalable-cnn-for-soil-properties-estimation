import os
import pandas as pd

# read csv files
ir = pd.read_csv("data/ir.csv", sep=';')
map = pd.read_csv("data/map.csv", sep=';')
per_informatici = pd.read_csv("data/per_informatici.csv", sep=';')
resistivita = pd.read_csv("data/resistivita.csv", sep=';')

# join them
tot = ir.join(map.set_index('lab'), on='sample')
tot = tot.join(per_informatici.set_index('SAMPLE'), on='sample')
tot = tot.join(resistivita.set_index('Punto'), on='campo')

# drop unnamed cols
tot.drop(tot.columns[tot.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)

# save in data
tot.to_csv('data/tot.csv', sep=';', index=False)
