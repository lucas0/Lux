import pandas as pd
import os, sys

cwd = os.path.abspath(__file__+"/..")

d1 = pd.read_csv(cwd+"/veritas3.csv")
d2 = pd.read_csv(cwd+"/pos_samples.csv", sep='\t')
d3 = pd.read_csv(cwd+"/good_a3.csv")

cols = list(set(d1.columns) & set(d2.columns) & set(d3.columns))
print(cols)

d1 = d1.loc[:,cols]
d2 = d2.loc[:,cols]
d3 = d3.loc[:,cols]

c = pd.concat([d1,d2,d3])

c.to_csv("dataset.csv", index=False, sep=',')
