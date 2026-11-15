import glob, json, sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mmptrack import load_scene

C = "/pscratch/sd/l/leana_ab/perlmutter-class/project/data/mmptrack/cache"
scene = sys.argv[1] if len(sys.argv) > 1 else "cafe_shop"
sp, cams, (H, W) = load_scene(C, scene)
cell = 100 // sp["VoxelSizeInMM"]


def maps(prefix):
    out = []
    for f in sorted(glob.glob(f"{C}/{prefix}__*{scene}_*.json")):
        for lab in json.load(open(f))["labels"]:
            t = np.zeros((H, W), np.float32)
            for x, y in lab:
                t[min(max(int(x) // cell, 0), H - 1), min(max(int(y) // cell, 0), W - 1)] = 1
            out.append(t)
    return out


train, val = maps("train"), maps("validation")
mean = torch.tensor(np.mean(train, 0))[None, None]
peak = (mean == F.max_pool2d(mean, 5, 1, 2)) & (mean > 0)
order = torch.argsort((peak[0, 0] * mean[0, 0]).flatten(), descending=True)

tp = fp = fn = 0
for t in val:
    gt = np.argwhere(t > 0).astype(float)
    k = len(gt)
    idx = order[:k]
    pr = np.stack([(idx // W).numpy(), (idx % W).numpy()], 1).astype(float)
    d = np.linalg.norm(pr[:, None] - gt[None], axis=2)
    r, c = linear_sum_assignment(d)
    m = int((d[r, c] <= 5).sum())
    tp += m
    fp += k - m
    fn += len(gt) - m
p = tp / max(tp + fp, 1)
r_ = tp / max(tp + fn, 1)
print(f"{scene} mean-map baseline: precision {p:.3f} recall {r_:.3f} F1 {2*p*r_/max(p+r_,1e-9):.3f} "
      f"(tp {tp} fp {fp} fn {fn}, train frames {len(train)}, val frames {len(val)})")
