import json
import numpy as np
from wildtrack import *

root = "/pscratch/sd/l/leana_ab/perlmutter-class/project/data/wildtrack/Wildtrack_dataset"
ds = WildtrackDataset(root)
v, t = ds[0]
print("views", tuple(v.shape), "target", tuple(t.shape), "people", int(t.sum()))

calib = load_calib(root)
people = json.load(open(f"{root}/annotations_positions/00000000.json"))
for cam in range(7):
    hits = n = 0
    for p in people:
        b = p["views"][cam]
        if b["xmin"] < 0:
            continue
        x, y = id_to_xy_cm(p["positionID"])
        u, w = project(calib[cam], np.array([[x, y, 0.0]]))[0]
        n += 1
        hits += (b["xmin"] - 20 <= u <= b["xmax"] + 20) and (b["ymin"] <= w <= b["ymax"] + 40)
    print(f"cam {cam + 1}: {hits}/{n} foot points land inside their box")