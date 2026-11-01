import numpy as np
import torch

ORIG_W, ORIG_H = 1920.0, 1080.0


def cell_points(set_name, grid_hw):
    """World coordinates (Hg*Wg, 3), in the calibration's units, of each target-cell center (z=0)."""
    H, W = grid_hw
    if set_name == "wildtrack":      # 480 cols (X) x 1440 rows (Y), 2.5 cm, origin (-300,-900) cm
        fx, fy = 480 / W, 1440 / H
        x0, y0, step = -300.0, -900.0, 2.5
    else:                            # multiviewx: 1000 cols x 640 rows, 0.025 m, origin 0
        fx, fy = 1000 / W, 640 / H
        x0, y0, step = 0.0, 0.0, 0.025
    xs = x0 + step * (np.arange(W) * fx + (fx - 1) / 2)
    ys = y0 + step * (np.arange(H) * fy + (fy - 1) / 2)
    X, Y = np.meshgrid(xs, ys)       # shape (H, W), rows = Y, cols = X
    return np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)


def make_geometry(set_name, root, img_hw, grid_hw, feat_hw, patch=16):
    if set_name == "wildtrack":
        from wildtrack import load_calib
        calib = load_calib(root)
    else:
        from multiviewx import load_calib_mx
        calib = load_calib_mx(root)
    pts = cell_points(set_name, grid_hw)
    sx, sy = img_hw[1] / ORIG_W, img_hw[0] / ORIG_H
    grids, masks = [], []
    for K, R, t in calib:
        cam = pts @ R.T + t
        z = cam[:, 2]
        uv = cam @ K.T
        with np.errstate(divide="ignore", invalid="ignore"):
            u, w = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
        gx = 2 * (u * sx / patch) / feat_hw[1] - 1
        gy = 2 * (w * sy / patch) / feat_hw[0] - 1
        ok = (z > 0) & (gx > -1) & (gx < 1) & (gy > -1) & (gy < 1)
        gx, gy = np.where(ok, gx, 2.0), np.where(ok, gy, 2.0)
        grids.append(np.stack([gx, gy], -1).reshape(*grid_hw, 2))
        masks.append(ok.reshape(grid_hw))
    return (torch.tensor(np.stack(grids), dtype=torch.float32),
            torch.tensor(np.stack(masks), dtype=torch.float32))