import os,sys

scriptname = sys.argv[0]
cwd = os.path.abspath(scriptname+"/..")

filename = cwd+"/results08-12-21.txt"
filename = cwd+"/results10-12-21.txt"
filename = cwd+"/results08-01-22.txt"

with open(filename, "r+", encoding="utf-8") as f:
    lines = f.readlines()

ob_lines = [l for l in lines if "only_bert" in l]
b_lines = [l for l in lines if l not in ob_lines]

print("b_lines:", len(b_lines))
print("ob_lines:", len(ob_lines))

#TrainShape:(1182, 837) #EPOCH: ([], 200, 'bert', 0.0001, 128, 0.3, 32) AVG: 0.7452020913559375 VAR: 0.0005738122241386669 F1: 0.743987339765227 SEED: 30507

f1s, accs, accs2, diffs, seeds =  [],[],[],[],[]
for i,e in enumerate(zip(b_lines,ob_lines)):
    b,ob = e
    s_b = ''.join(b.split(" ")[6:9])[:-1]
    s_ob = ''.join(ob.split(" ")[6:9])[:-1]
    seed_b = b.split(" ")[-1]
    seed_ob = ob.split(" ")[-1]

    if s_b != s_ob:
        print(i*2)
        print(s_b+" "+seed_b)
        print(s_ob+" "+seed_ob)
        exit(1)

    acc = float("{:.4f}".format(float(b.split(" ")[11])))
    accs.append(acc)

    acc2 = float("{:.4f}".format(float(ob.split(" ")[11])))
    acc2 = float(ob.split(" ")[11])
    accs2.append(acc2)

    diff = acc - acc2
    diffs.append(diff)

    seed = b.split(" ")[-1]
    seeds.append(seed)

    f1 = float("{:.4f}".format(float(b.split(" ")[15])))
    f1s.append(f1)

pairs = list(zip(accs,accs2,diffs,seeds,f1s))
pairs2 = pairs.copy()
sorted_h_acc = sorted(pairs, key=lambda x: x[0], reverse=True)
sorted_h_dif = sorted(pairs2, key=lambda x: x[2], reverse=True)

print("sorted by acc_B")
print("acc_b", "acc_ob", "diff", "seed", "f1")
print(sorted_h_acc[:5])
print("sorted by DIFF")
print("acc_b", "acc_ob", "diff", "seed", "f1")
print(sorted_h_dif[:5])
