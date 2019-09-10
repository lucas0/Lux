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

c['verdict'] = c['verdict'].str.lower()

c.loc[c['verdict'] == "legend", 'verdict'] = 'false'
c.loc[c['verdict'] == "false.", 'verdict'] = 'false'
c.loc[c['verdict'] == "mfalse", 'verdict'] = 'false'
c.loc[c['verdict'] == "mostly false", 'verdict'] = 'false'
c.loc[c['verdict'] == "true.", 'verdict'] = 'true'
c.loc[c['verdict'] == "mtrue", 'verdict'] = 'true'
c.loc[c['verdict'] == "mostly true", 'verdict'] = 'true'

print(c.groupby('verdict').count())

c.to_csv("dataset.csv", index=False, sep=',')
