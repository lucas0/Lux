import pickle
import os,sys
import numpy as np

drop = [17,23,81]

filename = sys.argv[0]
cwd = os.path.abspath(filename+"/..")

data_dir = cwd+"/data"
feat_file = data_dir+"/features"

features = pickle.load(open(feat_file,"rb" ))

print(type(features))
print(features.shape)

features = np.delete(features,drop,1)
print(features.shape)
