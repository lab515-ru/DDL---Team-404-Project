import json
import re
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

W0, H0 = 480, 270


def load_boxes(root):
    pos = json.load(open(f"{root}/mv_examples/positions_55x45.json"))
    box = np.full((len(pos), 3, 4), -1.0)
    for p in pos:
        for v in p["views"]:
            box[p["numPosition"], v["viewNum"]] = [v["xmin"], v["ymin"], v["xmax"], v["ymax"]]
    vis = (box[..., 3] >= 0) & (box[..., 2] > 0)
    foot = np.stack([(box[..., 0] + box[..., 2]) / 2, box[..., 3]], -1)
    return box, vis, foot


def grid_shape(root):
    box, vis, foot = load_boxes(root)
    n = len(box)
    med = {}
    for L in (45, 55):
        d = [np.linalg.norm(foot[i + L, v] - foot[i, v])
             for i in range(n - L) for v in range(3) if vis[i, v] and vis[i + L, v]]
        med[L] = float(np.median(d))
    L = min(med, key=med.get)                 # true row length -> adjacent cells
    return (n // L, L), med


def make_geometry_epfl(root, feat_hw, patch=16, img_hw=(H0, W0)):
    box, vis, foot = load_boxes(root)
    (H, W), _ = grid_shape(root)
    sx, sy = img_hw[1] / W0, img_hw[0] / H0
    grids, masks = [], []
    for v in range(3):
        gx = 2 * (foot[:, v, 0] * sx / patch) / feat_hw[1] - 1
        gy = 2 * (foot[:, v, 1] * sy / patch) / feat_hw[0] - 1
        ok = vis[:, v] & (gx > -1) & (gx < 1) & (gy > -1) & (gy < 1)
        gx, gy = np.where(ok, gx, 2.0), np.where(ok, gy, 2.0)
        grids.append(np.stack([gx, gy], -1).reshape(H, W, 2))
        masks.append(ok.reshape(H, W))
    return (torch.tensor(np.stack(grids), dtype=torch.float32),
            torch.tensor(np.stack(masks), dtype=torch.float32))


def load_examples(root):
    """frame -> {'pos': [cell ids], 'neg': [cell ids]} from positive.json / negative.json."""
    out = {}
    for name, key in (("positive", "pos"), ("negative", "neg")):
        for e in json.load(open(f"{root}/mv_examples/{name}.json")):
            fr = re.search(r"_(\d+)\.jpeg", e["views"][0]["imageFile"]).group(1)
            out.setdefault(fr, {"pos": [], "neg": []})[key].append(e["numPosition"])
    return out


def load_views(root, fr):
    ims = [np.asarray(Image.open(f"{root}/frames/cam{c}/RLCAFTCONF-C{c}_{fr}.jpeg").convert("RGB"))
           for c in range(3)]
    return torch.from_numpy(np.stack(ims)).permute(0, 3, 1, 2).float() / 255


@torch.no_grad()
def evaluate_epfl(model, root, rank, bs=4):
    """Per-cell occupied/free scoring. Needs --fusion calibrated. All ranks run forward."""
    from scipy.stats import rankdata
    inner = model.module if hasattr(model, "module") else model
    assert inner.geo is not None, "run with --fusion calibrated"
    old = inner.geo
    inner.geo = make_geometry_epfl(root, (inner.gh, inner.gw))
    ex = load_examples(root)
    keys = sorted(ex)
    model.eval()
    pos, neg = [], []
    for i in range(0, len(keys), bs):
        b = keys[i:i + bs]
        v = torch.stack([load_views(root, f) for f in b]).cuda()
        prob = torch.sigmoid(model(v).float())
        for j, f in enumerate(b):
            flat = prob[j, 0].flatten().cpu().numpy()
            pos += flat[ex[f]["pos"]].tolist()
            neg += flat[ex[f]["neg"]].tolist()
    inner.geo = old
    model.train()
    if rank != 0:
        return None
    pos, neg = np.array(pos), np.array(neg)
    r = rankdata(np.concatenate([pos, neg]))
    auc = (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    thr = float(np.median(np.concatenate([pos, neg])))      # balanced classes, so median threshold
    return {"frames": len(keys), "n_pos": len(pos), "n_neg": len(neg), "auc": float(auc),
            "acc_at_median": float(((pos > thr).sum() + (neg <= thr).sum()) / (len(pos) + len(neg)))}


def demo(root, out, n=40):
    box, vis, foot = load_boxes(root)
    ex = load_examples(root)
    frames = []
    for fr in sorted(ex)[:n]:
        row = []
        for c in range(3):
            im = Image.open(f"{root}/frames/cam{c}/RLCAFTCONF-C{c}_{fr}.jpeg").convert("RGB")
            d = ImageDraw.Draw(im)
            for key, col in (("pos", (0, 255, 0)), ("neg", (255, 0, 0))):
                for cid in ex[fr][key]:
                    if vis[cid, c]:
                        d.rectangle(list(box[cid, c]), outline=col, width=2)
            d.text((5, 5), f"cam{c} frame {fr}", fill=(255, 255, 0))
            row.append(im)
        canvas = Image.new("RGB", (W0 * 3, H0))
        for c, im in enumerate(row):
            canvas.paste(im, (c * W0, 0))
        frames.append(canvas)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=500, loop=0)
    print("wrote", out, len(frames), "frames (green = occupied, red = free)")


if __name__ == "__main__":
    root = sys.argv[2]
    if sys.argv[1] == "check":
        shape, med = grid_shape(root)
        box, vis, foot = load_boxes(root)
        ex = load_examples(root)
        print("grid (rows, cols)", shape, "median neighbor step px", med)
        print("visible fraction per camera", vis.mean(0).round(3).tolist())
        print("labeled frames", len(ex), "pos", sum(len(v["pos"]) for v in ex.values()),
              "neg", sum(len(v["neg"]) for v in ex.values()))
    elif sys.argv[1] == "demo":
        demo(root, sys.argv[3])