import json
import itertools
import numpy as np
from multiviewx import *

root = "/pscratch/sd/l/leana_ab/perlmutter-class/project/data/multiviewx/MultiviewX"
calib = load_calib_mx(root)
people = json.load(open(f"{root}/annotations_positions/00000.json"))

def cand(pid, order, spacing, ox, oy):
    a, b = (pid % 640, pid // 640) if order == "x640" else (pid % 1000, pid // 1000)
    return ox + spacing * a, oy + spacing * b

for order, ox, oy in itertools.product(["x640", "x1000"], [0.0, -8.0, -12.5, -16.0, -25.0], [0.0, -8.0, -12.5, -16.0, -25.0]):
    hits = n = 0
    for p in people:
        x, y = cand(p["positionID"], order, 0.025, ox, oy)
        for cam in range(6):
            b = p["views"][cam]
            if b["xmin"] < 0:
                continue
            u, w = project(calib[cam], np.array([[x, y, 0.0]]))[0]
            n += 1
            hits += (b["xmin"] - 20 <= u <= b["xmax"] + 20) and (b["ymin"] <= w <= b["ymax"] + 40)
    if n and hits / n > 0.5:
        print(order, ox, oy, f"{hits}/{n}")
print("done")