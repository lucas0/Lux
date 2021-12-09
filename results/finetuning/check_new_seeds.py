import os,sys

scriptname = sys.argv[0]
cwd = os.path.abspath(scriptname+"/..")

filename = cwd+"/results08-12-21.txt"

with open(filename, "r+", encoding="utf-8") as f:
    b_lines = f.readlines()

TrainShape:(1182, 837) #EPOCH: ([], 200, 'bert', 0.0001, 128, 0.3, 32) AVG: 0.7452020913559375 VAR: 0.0005738122241386669 F1: 0.743987339765227 SEED: 30507

f1s, accs, diffs, seeds = [], [], [], []
for b in b_lines:
    acc = float("{:.4f}".format(float(b.split(" ")[11])))
    accs.append(acc)

    seed = b.split(" ")[-1]
    seeds.append(seed)

    f1 = float("{:.4f}".format(float(b.split(" ")[15])))
    f1s.append(f1)

pairs = zip(accs,seeds,f1s)
sorted_h_acc = sorted(pairs, key=lambda x: x[0], reverse=True)
#sorted_h_dif = sorted(pairs, key=lambda x: x[2], reverse=True)

print(sorted_h_acc[:5])
#print(sorted_h_dif[:5])
