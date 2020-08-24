import pandas as pd
import os, sys
import numpy as np

cwd = os.path.abspath(__file__+"/..")

#use this min len if dealing with twitter/title/small body texts
MIN_BODY_LEN = 300
#use this otherwise
MIN_BODY_LEN = 500

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

#d6 = pd.read_csv(cwd+"/trusted.csv", sep="\t")
d6 = pd.read_csv(cwd+"/trusted_body.csv", sep="\t")
d6.o_body = d6.o_body.astype('str')
d6 = d6[d6['o_body'].map(len) > MIN_BODY_LEN]
d6.verdict = "true"
d6 = d6.sample(n=212)
print("size of trusted1.csv:", len(d6))

#d7 = pd.read_csv(cwd+"/fever.csv", sep="\t")
#d7.verdict = d7.verdict.astype('str')
#d7['o_body'] = d7.verdict
#d7['o_url'] = range(len(d7))
#print("size of fever.csv:", len(d7))
#print("columns of fever.csv:", d7.columns)

#d8 = pd.read_csv(cwd+"/thoracle.csv", sep="\t")
#d8.verdict = d8.verdict.astype('str')
#d8['o_body'] = d8.verdict
#d8['o_url'] = range(len(d8))
#print("size of fever.csv:", len(d8))

dataframes = [d1,d2,d3,d4,d5,d6]
#dataframes = [d4,d5,d6]
#dataframes = [d5]
#dataframes = [d7]
#dataframes = [d1,d2,d3,d4]

cols = set(d1.columns)
for i in dataframes:
    cols = cols.intersection(i)
cols = list(cols)
print(cols)

for i in dataframes:
    i = i.loc[:,cols]

c = pd.concat(dataframes).sample(frac=1)

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
print(c['verdict'].value_counts())
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
