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
#TrainShape:(1182, 869) #EPOCH: ([98], 100, 'bert', 0.001, 64) AVG: 0.8121693121693121 VAR: 0.00017409227065311717 F1: 0.8119331358845181
#TrainShape:(1182, 864) #EPOCH: ([0, 17], 100, 'bert', 0.001, 64) AVG: 0.7731481481481481 F1: 0.7715793816939528
with open(ab_file, "r+") as f:
    acc = [float("{:.5f}".format(float(line.split()[8+AB_DEPTH]))) for line in f]
    f.seek(0)
    var = [float("{:.5f}".format(float(line.split()[10+AB_DEPTH]))) for line in f]
    f.seek(0)
    drop = [" ".join(line.split()[3:(3+AB_DEPTH)]) for line in f]

base = acc.pop(0)
base_var = var.pop(0)
idx = range(len(acc))

print("BASELINE acc:", base)

diff = [float("{:.5f}".format(e-base)) for e in acc]
percent = ["{:.2%}".format(abs(x)/base) for x in diff]

with open(map_file, "r+") as f:
    names = []
    for line in f:
        names.append(" ".join(line.split()[1:]))

i_acc = sorted(zip(idx, names, diff, percent, var), key=lambda x: x[2], reverse=False)

range_var = i_acc[0][2] - i_acc[-1][2]
with open(out_file, "w+") as of:
    line = str("baseline")+"\t&\tbaseline\t&\t"+str(base)+"\t&\t"+str("100%")+"\t&\t"+str(base_var)+"\\\\\n"
    of.write(line)
    of.write("\midrule\n")
    for i in i_acc:
        line = str(i[0])+"\t&\t"+i[1]+"\t&\t"+str(i[2])+"\t&\t"+str(i[3])+"\t&\t"+str(i[4])+"\\\\\n"
        of.write(line)
        of.write("\midrule\n")

