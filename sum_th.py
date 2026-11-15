import json, glob, statistics as s
L = "/pscratch/sd/l/leana_ab/perlmutter-class/project/logs"
SC = ["cafe_shop", "industry_safety", "lobby", "office", "retail"]
TH = ["0.05", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.9"]
res = {}
for S in SC:
    for T in TH:
        v = [json.load(open(f))["by_threshold"][T]
             for f in sorted(glob.glob(f"{L}/th_{S}_r*/eval.json"))]
        if v:
            res[S, T] = v
print("mean F1 per scene (3 runs) by threshold")
print("scene".ljust(17) + "".join(t.rjust(7) for t in TH))
for S in SC:
    print(S.ljust(17) + "".join(f"{s.mean(x['f1'] for x in res[S, T]):7.3f}" if (S, T) in res else "    n/a" for T in TH))
print("\nmean across scenes".ljust(17) + "".join(
    f"{s.mean(s.mean(x['f1'] for x in res[S, T]) for S in SC if (S, T) in res):7.3f}" for T in TH))
print("worst scene".ljust(17) + "".join(
    f"{min(s.mean(x['f1'] for x in res[S, T]) for S in SC if (S, T) in res):7.3f}" for T in TH))
best = max(TH, key=lambda T: s.mean(s.mean(x['f1'] for x in res[S, T]) for S in SC if (S, T) in res))
print("\nbest single threshold across scenes:", best)
for S in SC:
    sd = s.pstdev([x['f1'] for x in res[S, best]]) if len(res.get((S, best), [])) > 1 else float('nan')
    m = s.mean(x['f1'] for x in res[S, best])
    p = s.mean(x['precision'] for x in res[S, best]); r = s.mean(x['recall'] for x in res[S, best])
    print(f"  {S}: F1 {m:.3f} (sd {sd:.3f}) P {p:.3f} R {r:.3f}")
