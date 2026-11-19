import json, glob, statistics as s
L = "/pscratch/sd/l/leana_ab/perlmutter-class/project/logs"
TP = FP = FN = 0
for S in ("industry_safety", "lobby", "office", "retail"):
    f1 = []
    for f in sorted(glob.glob(f"{L}/sc_{S}_r*/eval.json")):
        d = json.load(open(f))["by_threshold"]["0.05"]
        f1.append(d["f1"])
        TP += d["tp"]; FP += d["fp"]; FN += d["fn"]
    if f1:
        print(f"{S}: F1@0.05 runs {[round(x,3) for x in f1]} mean {s.mean(f1):.3f} sd {s.pstdev(f1):.3f}")
p = TP / max(TP + FP, 1); r = TP / max(TP + FN, 1)
print(f"pooled over 4 scenes and all runs: P {p:.3f} R {r:.3f} F1 {2*p*r/max(p+r,1e-9):.3f}")
