import os,sys

scriptname = sys.argv[0]
cwd = os.path.abspath(scriptname+"/..")

filename = cwd+"/results11-11-21.txt"
filename = cwd+"/big_seed_mix.txt"

with open(filename, "r+", encoding="utf-8") as f:
    lines = f.readlines()

ob_lines = [line for line in lines if "_bert" in line]
b_lines = [line for line in lines if line not in ob_lines]

f1s, accs, diffs, seeds = [], [], [], []
ob_accs = []
for b,ob in zip(b_lines, ob_lines):
    acc = float("{:.4f}".format(float(b.split(" ")[10])))
    ob_acc = float("{:.4f}".format(float(ob.split(" ")[10])))
    accs.append(acc)
    ob_accs.append(ob_acc)

    dif = acc- ob_acc
    diffs.append(dif)

    seed = b.split(" ")[-1]
    seeds.append(seed)

    f1 = float("{:.4f}".format(float(b.split(" ")[14])))
    f1s.append(f1)

pairs = zip(accs, ob_accs,diffs,seeds,f1s)
sorted_h_acc = sorted(pairs, key=lambda x: x[0], reverse=True)
sorted_h_dif = sorted(pairs, key=lambda x: x[2], reverse=True)

print(sorted_h_acc[:5])
print(sorted_h_dif[:5])
