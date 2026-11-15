import json
import numpy as np
from multiviewx import *

root = "/pscratch/sd/l/leana_ab/perlmutter-class/project/data/multiviewx/MultiviewX"
calib = load_calib_mx(root)
people = json.load(open(f"{root}/annotations_positions/00000.json"))
for p in people:
    pid = p["positionID"]
    x, y = (pid % 1000) * 0.025, (pid // 1000) * 0.025
    for cam in range(6):
        b = p["views"][cam]
        if b["xmin"] < 0:
            continue
        u, w = project(calib[cam], np.array([[x, y, 0.0]]))[0]
        ok = (b["xmin"] - 20 <= u <= b["xmax"] + 20) and (b["ymin"] <= w <= b["ymax"] + 40)
        if not ok:
            print(f"person {p['personID']} cam {cam+1}: proj ({u:.0f},{w:.0f}) box x[{b['xmin']},{b['xmax']}] y[{b['ymin']},{b['ymax']}] (image 1920x1080)")
