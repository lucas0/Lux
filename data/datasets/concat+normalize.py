import pandas as pd
import os, sys
import numpy as np

cwd = os.path.abspath(__file__+"/..")

MIN_BODY_LEN = 300
MAX_SENT_LEN = 3000

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
d6 = pd.read_csv(cwd+"/trusted1.csv", sep="\t")
d6.o_body = d6.o_body.astype('str')
d6 = d6[d6['o_body'].map(len) > MIN_BODY_LEN]
d6.verdict = "true"
d6 = d6.sample(n=228)
print("size of trusted1.csv:", len(d6))

dataframes = [d1,d2,d3,d4,d5,d6]

cols = set(d1.columns)
for i in dataframes:
    cols = cols.intersection(i)
cols = list(cols)

for i in dataframes:
    i = i.loc[:,cols]

c = pd.concat(dataframes, sort=True).sample(frac=1)

c['verdict'] = c['verdict'].str.lower()

c.loc[c['verdict'] == "legend", 'verdict'] = 'false'
#c.loc[c['verdict'] == "false.", 'verdict'] = 'false'
#c.loc[c['verdict'] == "mfalse", 'verdict'] = 'false'
#c.loc[c['verdict'] == "mostly false", 'verdict'] = 'false'
#c.loc[c['verdict'] == "true.", 'verdict'] = 'true'
#c.loc[c['verdict'] == "mtrue", 'verdict'] = 'true'
#c.loc[c['verdict'] == "mostly true", 'verdict'] = 'true'
c.loc[c['verdict'].str.contains("true"), 'verdict'] = 'true'
c.loc[c['verdict'].str.contains("false"), 'verdict'] = 'false'
c.loc[c['verdict'].str.startswith("un"), 'verdict'] = 'unverified'

c = c[c['verdict'].isin(['true', 'false'])]

print("\nsize of concatenated \"dataset\":",len(c))
c.o_body = c.o_body.astype('str')
c = c[c['o_body'].map(len) > MIN_BODY_LEN]
lens = np.asarray([len(e.split(" ")) for e in c['o_body'].values])
c = c[lens < MAX_SENT_LEN]
print("after dropping origins with less than "+str(MIN_BODY_LEN)+" chars or more than "+str(MAX_SENT_LEN)+" sentences:",len(c))
c.drop_duplicates(subset=['o_url'], inplace=True, keep='last')
print("size of final version of  \"dataset\" (after dropping dupplicated o_url):",len(c))

print(c['verdict'].value_counts())
dif = (c['verdict'].value_counts()[0] - c['verdict'].value_counts()[1])
print("Difference False-True:", dif)

c.to_csv(cwd+"/dataset.csv", index=False, sep=',')
