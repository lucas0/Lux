
with open("results.txt", "r+") as f:
    lines = f.readlines()

avgs = []
f1s = []
models = []
for i,l in enumerate(lines):
    model, results = l.split('AVG: ')
    models.append((i,model))
    avg,f1 = results[:-1].split('F1: ')
    f1s.append(f1)
    avgs.append(avg)

a, m = zip(*sorted(zip(avgs, models), reverse=True))
f, m2 = zip(*sorted(zip(f1s, models), reverse=True))

for i in range(5):
    print(a[i], m[i])
    print(f[i], m2[i])
