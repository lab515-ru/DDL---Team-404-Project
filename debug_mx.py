import json
import numpy as np
from geometry import cell_points
from multiviewx import load_calib_mx

R = "/pscratch/sd/l/leana_ab/perlmutter-class/project/data/multiviewx/MultiviewX"
calib = load_calib_mx(R)
pts = cell_points("multiviewx", (160, 250))
print("cell x range", pts[:, 0].min(), pts[:, 0].max(), "y range", pts[:, 1].min(), pts[:, 1].max())
p = json.load(open(f"{R}/annotations_positions/00000.json"))[0]
pid = p["positionID"]
print("person 0 world:", (pid % 1000) * 0.025, (pid // 1000) * 0.025)
for c in range(6):
    K, Rm, t = calib[c]
    cam = pts @ Rm.T + t
    z = cam[:, 2]
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    w = K[1, 1] * cam[:, 1] / z + K[1, 2]
    print(f"cam {c+1}: z {z.min():.1f}..{z.max():.1f}, frac z>0 {np.mean(z > 0):.2f}, "
          f"u {np.percentile(u, 5):.0f}..{np.percentile(u, 95):.0f}, w {np.percentile(w, 5):.0f}..{np.percentile(w, 95):.0f}")
