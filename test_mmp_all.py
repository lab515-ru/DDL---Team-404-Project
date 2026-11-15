import glob, itertools, json, os
import numpy as np

ROOT = "calib_test"
SCENES = ["cafe_shop", "industry_safety", "lobby", "office", "retail"]


def project(c, X, conv):
    R = np.array(c["ExtrinsicParameters"]["Rotation"]).reshape(3, 3)
    t = np.array(c["ExtrinsicParameters"]["Translation"])
    K = c["IntrinsicParameters"]
    Xc = {"A": R @ X + t, "B": R.T @ X + t, "C": R.T @ (X - t), "D": R @ (X - t)}[conv]
    if Xc[2] <= 0:
        return None
    return K["Fx"] * Xc[0] / Xc[2] + K["Cx"], K["Fy"] * Xc[1] / Xc[2] + K["Cy"]


for sc in SCENES:
    d = f"{ROOT}/{sc}"
    cal = json.load(open(f"{d}/calibrations.json"))
    sp = cal["Space"]
    nU = (sp["MaxU"] - sp["MinU"]) // sp["VoxelSizeInMM"]
    nV = (sp["MaxV"] - sp["MinV"]) // sp["VoxelSizeInMM"]
    cams = {c["CameraId"]: c for c in cal["Cameras"]}
    frames = sorted(os.path.basename(f)[8:13] for f in glob.glob(f"{d}/topdown_*.csv"))
    data = []
    mx = my = 0
    for fr in frames:
        rows = [l.split(",") for l in open(f"{d}/topdown_{fr}.csv") if l.strip()]
        ppl = {r[0].strip(): (float(r[1]), float(r[2])) for r in rows}
        for x, y in ppl.values():
            mx, my = max(mx, x), max(my, y)
        bx = {cid: json.load(open(f"{d}/rgb_{fr}_{cid}.json")) for cid in cams
              if os.path.exists(f"{d}/rgb_{fr}_{cid}.json")}
        data.append((ppl, bx))
    res = []
    for conv, swap, half, W in itertools.product("ABCD", [False, True], [0, 0.5], [0, 100]):
        hit = n = 0
        for ppl, bx in data:
            for pid, (x, y) in ppl.items():
                a, b = (y, x) if swap else (x, y)
                X = np.array([sp["MinU"] + 20 * (a + half), sp["MinV"] + 20 * (b + half), W], float)
                for cid, c in cams.items():
                    if cid not in bx or pid not in bx[cid]:
                        continue
                    l, t, r, bt = bx[cid][pid]
                    n += 1
                    p = project(c, X, conv)
                    if p and abs(p[0] - (l + r) / 2) <= max(15, 0.5 * (r - l)) and abs(p[1] - bt) <= 25:
                        hit += 1
        res.append((hit, n, conv, "swap" if swap else "xU,yV", half, W))
    res.sort(reverse=True)
    print(f"== {sc}: cams {sorted(cams)}, grid {nU}(U) x {nV}(V), max label x={mx:.0f} y={my:.0f}")
    for r in res[:3]:
        print("  ", r)
