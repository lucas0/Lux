import numpy as np
import pickle

with open("features", "rb") as f:
    features = pickle.load(f)

drop_feat_idx=[50]

print(features.shape)
features = np.delete(features,drop_feat_idx,1)
print(features.shape)

input()

for idx,f in enumerate(features):
    if len(f) != 100:
        print(idx, len(f))
