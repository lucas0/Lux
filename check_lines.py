import os,sys

scriptname = sys.argv[0]
cwd = os.path.abspath(scriptname+"/..")

f0 = cwd+"/feat_log.txt"
f1 = cwd+"/feat_log1.txt"
f0 = cwd+"/log_test_folds.txt"
f1 = cwd+"/log_test_folds1.txt"

with open(f0,"r+") as f:
    lines0 = f.readlines()
with open(f1,"r+") as f:
    lines1 = f.readlines()
for i,a in enumerate(zip(lines0, lines1)):
    if a[0] != a[1]:
        print("i:", i)
        print("a:", a)
print(cwd)
