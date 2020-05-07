import pandas as pd
import os

cwd = os.path.abspath(__file__)
datasets_path = os.path.abspath(cwd+"/../../")

v4 = pd.read_csv(datasets_path+"/veritas4.csv", sep="\t")

print(v4.groupby('verdict').count())
