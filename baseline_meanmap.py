import json, glob, numpy as np, torch, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

R = "/pscratch/sd/l/leana_ab/perlmutter-class/project/data/wildtrack/Wildtrack_dataset"
HOLD, GAP, H, W = 40, 5, 360, 120
files = sorted(glob.glob(R + "/annotations_positions/*.json"))
n = len(files)


def target(f):
    t = np.zeros((H, W), np.float32)
    for p in json.load(open(f)):
        pid = p["positionID"]
        t[(pid // 480) * H // 1440, (pid % 480) * W // 480] = 1
    return t


train = [target(f) for f in files[: n - HOLD - GAP]]
val = [target(f) for f in files[n - HOLD:]]
mean = torch.tensor(np.mean(train, 0))[None, None]
peak = (mean == F.max_pool2d(mean, 5, 1, 2)) & (mean > 0)
cand = (peak[0, 0] * mean[0, 0]).flatten()
order = torch.argsort(cand, descending=True)

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
print(f"mean-map baseline: precision {p:.3f} recall {r_:.3f} F1 {2*p*r_/max(p+r_,1e-9):.3f}  (tp {tp} fp {fp} fn {fn}, val frames {len(val)})")
