import itertools
from itertools import groupby,chain,combinations
import sys, os

class LineSeekableFile:
    def __init__(self, seekable):
        self.c = 0
        self.fin = seekable
        self.line_map = list() # Map from line index -> file position.
        self.line_map.append(0)
        while seekable.readline():
            print(self.c)
            self.c += 1
            self.line_map.append(seekable.tell())

    def __getitem__(self, index):
         # NOTE: This assumes that you're not reading the file sequentially.
        # For that, just use 'for line in file'.
        self.fin.seek(self.line_map[index])
        return self.fin.readline()

#filename = sys.argv[0]
#cwd = os.path.abspath(filename+"/..")
#bert_dir = cwd+"/res/bert"
#bert_filename = bert_dir+"/output4layers.json"

#with open(bert_filename, "rt") as fin:
#    seek = LineSeekableFile(fin)
#    b_line = seek[idx]


input_type = ['bert']
learning_rate = [0.0001, 0.001]
num_dims = [64,128]
epochs = [20,50,100,200]


initial_feat = set(range(97))
removed_feat = set([])
check_feat = initial_feat - removed_feat
drop_features_idx = [list(x) for x in (itertools.combinations(check_feat, 1))]

setup = itertools.product(drop_features_idx,epochs,input_type,learning_rate,num_dims)
la = zip(setup, drop_features_idx)

print(sum([1 for e in setup]))

for s in setup:
    print(s)

