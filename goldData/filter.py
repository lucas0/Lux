import pandas as pd
import os

cwd = os.getcwd()

results = pd.read_csv(cwd+"/results.csv", sep=',')

pos = results.loc[results['value'] == 1]
neg = results.loc[results['value'] == 0]

pos.to_csv(cwd+"/good_a3.csv", index=False)
neg.to_csv(cwd+"/bad_a3.csv", index=False)
