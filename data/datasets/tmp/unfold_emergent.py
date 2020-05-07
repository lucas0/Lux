import pandas as pd
import ast

df = pd.read_csv("emergent.csv", sep='\t')

L = []
for i,e in df.iterrows():
    #page    claim   claim_label     tags    origin_list     date
    origin_list = ast.literal_eval(e['origin_list'])
    for origin_url in origin_list:
        e['o_url'] = origin_url
        L.append(list(e))

u_df = pd.DataFrame(L, columns=["a_url","claim","verdict","a_tags","source_list","a_date","o_url"])
u_df.to_csv("unfolded_emergent.csv", index=False, sep="\t")
