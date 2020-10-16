import itertools
import sys, os

cwd = os.path.abspath(__file__+"/..")
abwd = cwd+"/results_ablation"

AB_DEPTH = 1
HEAD_N = 3

ab_file = abwd+"/ablation"+str(AB_DEPTH)+"res.txt"
map_file = abwd+"/map_of_features.txt"
out_file = abwd+"/output.txt"

#TrainShape:(1182, 864) #EPOCH: ([0], 100, 'bert', 0.001, 64) AVG: 0.7731481481481481 F1: 0.7715793816939528

#TrainShape:(1182, 864) #EPOCH: ([0, 17], 100, 'bert', 0.001, 64) AVG: 0.7731481481481481 F1: 0.7715793816939528


with open(ab_file, "r+") as f:
    acc = [float(line.split()[8+AB_DEPTH]) for line in f]
with open(ab_file, "r+") as f:
    drop = [" ".join(line.split()[3:(3+AB_DEPTH)]) for line in f]

idx = range(96)

base = float(0.8050793650793651)

avg = sum(acc)/len(acc)
print("AVG of runs:", avg)

dev1 = [e-avg for e in acc]

dev = [base-e for e in dev1]

dev1 = [float("{:.5f}".format(x)) for x in dev1]
dev = [float("{:.5f}".format(x)) for x in dev]

with open(map_file, "r+") as f:
    names = []
    for line in f:
        names.append(" ".join(line.split()[1:]))

i_acc = sorted(zip(idx, names, dev, dev1), key=lambda x: x[2], reverse=True)

range_var = i_acc[0][2] - i_acc[-1][2]
with open(out_file, "w+") as of:
    for i in i_acc:
        line = str(i[0])+"\t&\t"+i[1]+"\t&\t"+str(i[2])+"\t&\t"+str(i[3])+"\\\\\n"
        of.write(line)
        of.write("\midrule\n")

sys.exit(1)
print(len(i_acc))
remove = [i for i in i_acc if i[2] > 0]
#remove = [i for i in i_acc if i[2] > 0]

rmv_dev = [e[2] for e in remove]
avg_rmv_dev = sum(rmv_dev)/len(rmv_dev)
print(avg_rmv_dev)

#remove = [i for i in i_acc if i[2] < avg_rmv_dev]
#remove = [i for i in i_acc if i[2] > avg_rmv_dev]
print(len(remove))

for e in remove:
    print(e)

feats = [str(e[0])[2:-2] for e in remove]

print(feats)
