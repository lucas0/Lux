import os,sys

scriptname = sys.argv[0]
cwd = os.path.abspath(scriptname+"/..")

filename = cwd+"/baseline.txt"

with open(filename, "r+", encoding="utf-8") as f:
    lines = f.readlines()

ob_lines = [line for line in lines if "only_bert" in line]
b_lines = [line for line in lines if line not in ob_lines]

accs, diffs, seeds = [], [], []
for b,ob in zip(b_lines,ob_lines):
    acc = float("{:.4f}".format(float(b.split(" ")[9])))
    ob_acc = float("{:.4f}".format(float(ob.split(" ")[9])))
    accs.append(acc)

    dif = acc - ob_acc
    diffs.append(dif)

    seed = b.split(" ")[-1]
    seeds.append(seed)

pairs = zip(accs, diffs, seeds)
sorted_h_acc = sorted(pairs, key=lambda x: x[0], reverse=True)
sorted_h_dif = sorted(pairs, key=lambda x: x[1], reverse=True)

print(sorted_h_acc[:5])
print(sorted_h_dif)

