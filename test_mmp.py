import itertools, json, sys
import numpy as np

CAL = "../sample/train/calibrations/cafe_shop/calibrations.json"
D = "cafe_shop_0"
FRAME = "03147"

cal = json.load(open(CAL))
sp = cal["Space"]
cams = {c["CameraId"]: c for c in cal["Cameras"]}
rows = [l.strip().split(",") for l in open(f"{D}/topdown_{FRAME}.csv") if l.strip()]
people = {r[0].strip(): (float(r[1]), float(r[2])) for r in rows}
boxes = {c: json.load(open(f"{D}/rgb_{FRAME}_{c}.json")) for c in cams}


def project(c, X, conv):
    R = np.array(c["ExtrinsicParameters"]["Rotation"]).reshape(3, 3)
    t = np.array(c["ExtrinsicParameters"]["Translation"])
    K = c["IntrinsicParameters"]
    Xc = {"A": R @ X + t, "B": R.T @ X + t, "C": R.T @ (X - t), "D": R @ (X - t)}[conv]
    if Xc[2] <= 0:
        return None
    return K["Fx"] * Xc[0] / Xc[2] + K["Cx"], K["Fy"] * Xc[1] / Xc[2] + K["Cy"]


res = []
for conv, swap, half, W in itertools.product("ABCD", [False, True], [0, 0.5], [0, 100, 200]):
    hit = n = 0
    for pid, (x, y) in people.items():
        a, b = (y, x) if swap else (x, y)
        U = sp["MinU"] + 20 * (a + half)
        V = sp["MinV"] + 20 * (b + half)
        for cid, c in cams.items():
            if pid not in boxes[cid]:
                continue
            l, t, r, bt = boxes[cid][pid]
            n += 1
            p = project(c, np.array([U, V, W], float), conv)
            if p and abs(p[0] - (l + r) / 2) <= max(15, 0.5 * (r - l)) and abs(p[1] - bt) <= 25:
                hit += 1
    res.append((hit, n, conv, "swap" if swap else "xU,yV", half, W))
for r in sorted(res, reverse=True)[:8]:
    print(r)
