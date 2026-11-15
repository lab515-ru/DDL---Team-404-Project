import json
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

ORIG_W, ORIG_H = 640.0, 360.0


def load_scene(root, scene):
    d = json.load(open(f"{root}/calib/{scene}.json"))
    sp = d["Space"]
    vox = sp["VoxelSizeInMM"]
    cell = 100 // vox
    nU = (sp["MaxU"] - sp["MinU"]) // vox
    nV = (sp["MaxV"] - sp["MinV"]) // vox
    grid = (-(-nV // cell), -(-nU // cell))          # (rows = V, cols = U)
    cams = sorted(d["Cameras"], key=lambda c: c["CameraId"])
    return sp, cams, grid


def make_geometry_mmp(root, scene, img_hw, grid_hw, feat_hw, patch=16):
    sp, cams, _ = load_scene(root, scene)
    H, W = grid_hw
    vox = sp["VoxelSizeInMM"]
    cell = 100 // vox
    V = sp["MinV"] + vox * cell * (np.arange(H) + 0.5)
    U = sp["MinU"] + vox * cell * (np.arange(W) + 0.5)
    UU, VV = np.meshgrid(U, V)                        # (H, W)
    pts = np.stack([UU.ravel(), VV.ravel(), np.zeros(UU.size)], 1)
    sx, sy = img_hw[1] / ORIG_W, img_hw[0] / ORIG_H
    grids, masks = [], []
    for c in cams:
        R = np.array(c["ExtrinsicParameters"]["Rotation"]).reshape(3, 3)
        t = np.array(c["ExtrinsicParameters"]["Translation"])
        K = c["IntrinsicParameters"]
        Xc = (pts - t) @ R                            # = R^T (X - t)
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K["Fx"] * Xc[:, 0] / z + K["Cx"]
            w = K["Fy"] * Xc[:, 1] / z + K["Cy"]
        gx = 2 * (u * sx / patch) / feat_hw[1] - 1
        gy = 2 * (w * sy / patch) / feat_hw[0] - 1
        ok = (z > 0) & (gx > -1) & (gx < 1) & (gy > -1) & (gy < 1)
        gx, gy = np.where(ok, gx, 2.0), np.where(ok, gy, 2.0)
        grids.append(np.stack([gx, gy], -1).reshape(H, W, 2))
        masks.append(ok.reshape(H, W))
    return (torch.tensor(np.stack(grids), dtype=torch.float32),
            torch.tensor(np.stack(masks), dtype=torch.float32))


class MMPDataset(Dataset):
    """One scene, one or more cached recordings (stems from cache_mmp.py).
    Returns views (V,3,H,W) float and target (1, gridH, gridW)."""

    def __init__(self, root, scene, names, img_hw=(270, 480)):
        self.sp, self.cams, self.grid = load_scene(root, scene)
        self.cell = 100 // self.sp["VoxelSizeInMM"]
        self.arrs, self.labels, self.index = [], [], []
        for k, n in enumerate(names):
            a = np.load(f"{root}/{n}.npy", mmap_mode="r")
            meta = json.load(open(f"{root}/{n}.json"))
            assert meta["scene"] == scene, (n, meta["scene"])
            assert a.shape[1:] == (len(self.cams), 3, *img_hw), a.shape
            self.arrs.append(a)
            self.labels.append(meta["labels"])
            self.index += [(k, i) for i in range(len(a))]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        k, j = self.index[i]
        views = torch.from_numpy(np.array(self.arrs[k][j])).float() / 255
        H, W = self.grid
        t = torch.zeros(1, H, W)
        for x, y in self.labels[k][j]:
            t[0, min(max(int(x) // self.cell, 0), H - 1),
              min(max(int(y) // self.cell, 0), W - 1)] = 1.0
        return views, t


if __name__ == "__main__":      # python mmptrack.py CACHE_ROOT scene
    sp, cams, grid = load_scene(sys.argv[1], sys.argv[2])
    print("cameras", [c["CameraId"] for c in cams], "views", len(cams),
          "--grid_h", grid[0], "--grid_w", grid[1])