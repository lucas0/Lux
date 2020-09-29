import itertools
import sys, os

cwd = os.path.abspath(__file__+"/..")
abwd = cwd+"/results_ablation"

AB_DEPTH = 2
HEAD_N = 3

ab_file = abwd+"/ablation"+str(AB_DEPTH)+"res.txt"

#TrainShape:(1182, 864) #EPOCH: ([0], 100, 'bert', 0.001, 64) AVG: 0.7731481481481481 F1: 0.7715793816939528

#TrainShape:(1182, 864) #EPOCH: ([0, 17], 100, 'bert', 0.001, 64) AVG: 0.7731481481481481 F1: 0.7715793816939528


with open(ab_file, "r+") as f:
    acc = [float(line.split()[8+AB_DEPTH]) for line in f]
with open(ab_file, "r+") as f:
    drop = [" ".join(line.split()[3:(3+AB_DEPTH)]) for line in f]

avg = sum(acc)/len(acc)
print("AVG of runs:", avg)

dev = [e-avg for e in acc]

i_acc = sorted(zip(drop, acc, dev), key=lambda x: x[1], reverse=False)

range_var = i_acc[0][1] - i_acc[-1][1]
print("range:", range_var)
for i in i_acc[:HEAD_N]:
    print(i)

print(len(i_acc))
remove = [i for i in i_acc if i[2] < -0.01]
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
