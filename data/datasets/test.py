import pandas as pd
import os, sys

cwd = os.path.abspath(__file__+"/..")

v3 = pd.read_csv(cwd+"/veritas3.csv")
pos_samples = pd.read_csv(cwd+"/pos_samples.csv", sep='\t')

print(type(v3.iloc[0].o_body))
print(type(pos_samples.iloc[0].o_body))
