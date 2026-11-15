import json, glob, statistics as s
for W in (30, 100, 300):
    f1 = []
    for f in sorted(glob.glob(f"/pscratch/sd/l/leana_ab/perlmutter-class/project/logs/pw{W}_r*/eval.json")):
        f1.append(json.load(open(f))["by_threshold"]["0.05"]["f1"])
    if f1:
        print(f"pos_weight {W}: F1@0.05 runs {[round(x,3) for x in f1]} mean {s.mean(f1):.3f} sd {s.pstdev(f1):.3f}")
