import json
import xml.etree.ElementTree as ET
import numpy as np
from multiviewx import load_calib_mx

root = "/pscratch/sd/l/leana_ab/perlmutter-class/project/data/multiviewx/MultiviewX"
calib = load_calib_mx(root)
dist = []
for c in range(1, 7):
    e = ET.parse(f"{root}/calibrations/intrinsic/intr_Camera{c}.xml").getroot()
    dist.append(np.array(e.find("distortion_coefficients/data").text.split(), float))
    print("cam", c, "dist", np.round(dist[-1], 5))


def proj(cam, X, use_dist):
    K, R, t = calib[cam]
    Xc = R @ X + t
    x, y = Xc[0] / Xc[2], Xc[1] / Xc[2]
    if use_dist:
        k1, k2, p1, p2, k3 = dist[cam]
        r2 = x * x + y * y
        rad = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        x, y = (x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)
    return K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]


for use in (False, True):
    hit = n = 0
    for fr in (0, 100, 200, 300, 399):
        for p in json.load(open(f"{root}/annotations_positions/{fr:05d}.json")):
            X = np.array([(p["positionID"] % 1000) * 0.025, (p["positionID"] // 1000) * 0.025, 0.0])
            for cam in range(6):
                b = p["views"][cam]
                if b["xmin"] < 0:
                    continue
                u, w = proj(cam, X, use)
                n += 1
                hit += (b["xmin"] - 20 <= u <= b["xmax"] + 20) and (b["ymin"] <= w <= b["ymax"] + 40)
    print("with distortion" if use else "without distortion", f"{hit}/{n}")
