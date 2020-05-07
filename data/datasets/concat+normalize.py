import pandas as pd
import os, sys

cwd = os.path.abspath(__file__+"/..")


d1 = pd.read_csv(cwd+"/veritas3.csv")
print("size of veritas3.csv:", len(d1))
d2 = pd.read_csv(cwd+"/pos_samples.csv", sep='\t')
print("size of pos_samples.csv:", len(d2))
d3 = pd.read_csv(cwd+"/good_a3.csv")
print("size of good_a3.csv:", len(d3))
d4 = pd.read_csv(cwd+"/veritas4.csv", sep="\t")
print("size of veritas4.csv:", len(d4))
d5 = pd.read_csv(cwd+"/emergent_gold.csv", sep="\t")
print("size of emergent_gold.csv:", len(d5))

print(d1.columns)
print(d2.columns)
print(d3.columns)
print(d4.columns)
print(d5.columns)

cols = list(set(d1.columns) & set(d2.columns) & set(d3.columns) & set(d4.columns) & set(d5.columns))
#cols = list(set(d1.columns) & set(d2.columns) & set(d3.columns) & set(d4.columns))
#cols = list(set(d1.columns) & set(d2.columns) & set(d3.columns))
print(cols)

d1 = d1.loc[:,cols]
d2 = d2.loc[:,cols]
d3 = d3.loc[:,cols]
d4 = d4.loc[:,cols]
d5 = d5.loc[:,cols]

c = pd.concat([d1,d2,d3,d4,d5]).sample(frac=1)
#c = pd.concat([d1,d2,d3,d4]).sample(frac=1)
#c = pd.concat([d1,d2,d3]).sample(frac=1)
#c = pd.concat([d1,d2,d3]).sample(frac=1)

c['verdict'] = c['verdict'].str.lower()

c.loc[c['verdict'] == "legend", 'verdict'] = 'false'
c.loc[c['verdict'] == "false.", 'verdict'] = 'false'
c.loc[c['verdict'] == "mfalse", 'verdict'] = 'false'
c.loc[c['verdict'] == "mostly false", 'verdict'] = 'false'
c.loc[c['verdict'] == "true.", 'verdict'] = 'true'
c.loc[c['verdict'] == "mtrue", 'verdict'] = 'true'
c.loc[c['verdict'] == "mostly true", 'verdict'] = 'true'
c.loc[c['verdict'].str.startswith("true"), 'verdict'] = 'true'
c.loc[c['verdict'].str.startswith("un"), 'verdict'] = 'unverified'

print("size of concatenated \"dataset\":",len(c))
c.to_csv("dataset.csv", index=False, sep=',')
c.drop_duplicates(subset='o_url', inplace=True)
print(c.groupby('verdict').count())
print(c.columns)
print(len(c))
