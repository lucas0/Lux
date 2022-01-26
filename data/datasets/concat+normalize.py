import pandas as pd
import os, sys
import numpy as np

cwd = os.path.abspath(__file__+"/..")

#use this min len if dealing with twitter/title/small body texts
MIN_BODY_LEN = 300
MIN_BODY_LEN = 1

#use this otherwise
#MIN_BODY_LEN = 500

MAX_SENT_LEN = 3000

#data from evaluation of the automatic methods
d2 = pd.read_csv(cwd+"/pos_samples.csv", sep='\t')
print("size of pos_samples.csv:", len(d2))
d3 = pd.read_csv(cwd+"/good_a3.csv")
print("size of good_a3.csv:", len(d3))

#data from the FIRST annotation session (lucas annotation day)
d1 = pd.read_csv(cwd+"/veritas3.csv")
print("size of veritas3.csv:", len(d1))

#from MAIN annotation session
d4 = pd.read_csv(cwd+"/veritas4.csv", sep="\t")
print("size of veritas4.csv:", len(d4))
d5 = pd.read_csv(cwd+"/emergent_gold.csv", sep="\t")
print("size of emergent_gold.csv:", len(d5))

#this one is with claims instead of bodies (for datasets that have small texts)
#d6 = pd.read_csv(cwd+"/trusted.csv", sep="\t")

#data extracted from trusted news sources
d6 = pd.read_csv(cwd+"/trusted_body.csv", sep="\t")
d6.o_body = d6.o_body.astype('str')
d6 = d6[d6['o_body'].map(len) > MIN_BODY_LEN]
d6.verdict = "true"
#roughly the size of trusted to make T2 (the balancing for V4+EM)
#d6 = d6.sample(n=142)
#roughly the size of trusted to make T1 (the balancing for V4)
d6 = d6.sample(n=376)
print("size of trusted1.csv:", len(d6))

#d7 = pd.read_csv(cwd+"/fever.csv", sep="\t")
#d7.verdict = d7.verdict.astype('str')
#d7['o_body'] = d7.claim
#d7['o_url'] = range(len(d7))
#print("size of fever.csv:", len(d7))
#print("columns of fever.csv:", d7.columns)

#d8 = pd.read_csv(cwd+"/thoracle.csv", sep="\t")
#d8.verdict = d8['Is Tweet Content Valid'].astype('str')
#d8['o_body'] = d8['Tweet Content']
#d8['o_url'] = range(len(d8))
#print("size of thoracle.csv:", len(d8))

#d9 = pd.read_csv(cwd+"/snopes2019.csv", sep="\t")
#d9.verdict = d9.verdict.astype('str')
#d9['o_url'] = range(len(d9))
#print("size of snopes2019.csv:", len(d9))
#print("columns of snopes2019.csv:", d9.columns)

#list that defines which datasets will be concatenated in order to generate dataset.csv
dataframes = [d1,d2,d3,d4,d5]
#dataframes = [d9]

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

print("=======")
print(len(c))
print(c['verdict'].value_counts())
print("=======")
c = c[c['verdict'].isin(['true', 'false'])]

print("\nsize of concatenated \"dataset\":",len(c))
print(c['verdict'].value_counts())
c.o_body = c.o_body.astype('str')
c = c[c['o_body'].map(len) > MIN_BODY_LEN]
lens = np.asarray([len(e.split(" ")) for e in c['o_body'].values])
c = c[lens < MAX_SENT_LEN]
print("after dropping origins with less than "+str(MIN_BODY_LEN)+" chars or more than "+str(MAX_SENT_LEN)+" sentences:",len(c))
c.drop_duplicates(subset=['o_url'], inplace=True, keep='last')
print("size of \"dataset\" (after dropping dupplicated o_url):",len(c))

#optional line to manage the size of the dataset
c = c.sample(frac=1)
print("final size of \"dataset\":",len(c))

print(c['verdict'].value_counts())
dif = (c['verdict'].value_counts()[0] - c['verdict'].value_counts()[1])
print("Difference False-True:", dif)

c.to_csv(cwd+"/bck_dataset_v4+em.csv", index=False, sep=',')
sys.exit(1)
c.to_csv(cwd+"/dataset.csv", index=False, sep=',')
