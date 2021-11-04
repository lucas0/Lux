import os,sys

scriptname = sys.argv[0]
cwd = os.path.abspath(scriptname+"/..")

filename = cwd+"/seed4419.txt"
filename = cwd+"/new_seed4419.txt"

with open(filename, "r+", encoding="utf-8") as f:
    lines = f.readlines()

accs, diffs, seeds = [], [], []
pairs = []
for line in lines:
    acc = float("{:.4f}".format(float(line.split(" ")[10])))
    f1 = float("{:.4f}".format(float(line.split(" ")[14])))
    setup = line.split(" ")[4:9]

    pairs.append((acc,f1,setup))

sorted_pairs = sorted(pairs, key=lambda x: x[0], reverse=True)

print(sorted_pairs[:5])

