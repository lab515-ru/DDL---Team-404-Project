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


def load_dist_mx(root):
    """OpenCV distortion coefficients (k1, k2, p1, p2, k3) for MultiviewX cameras 1-6."""
    import xml.etree.ElementTree as ET
    out = []
    for c in range(1, 7):
        e = ET.parse(f"{root}/calibrations/intrinsic/intr_Camera{c}.xml").getroot()
        out.append(np.array(e.find("distortion_coefficients/data").text.split(), float))
    return out


def make_geometry(set_name, root, img_hw, grid_hw, feat_hw, patch=16):
    if set_name == "wildtrack":
        from wildtrack import load_calib
        calib = load_calib(root)
        dists = [None] * len(calib)      # WILDTRACK uses intrinsic_zero (undistorted images)
    else:
        from multiviewx import load_calib_mx
        calib = load_calib_mx(root)
        dists = load_dist_mx(root)       # only camera 4 is non-negligible
    pts = cell_points(set_name, grid_hw)
    sx, sy = img_hw[1] / ORIG_W, img_hw[0] / ORIG_H
    grids, masks = [], []
    for (K, R, t), dist in zip(calib, dists):
        cam = pts @ R.T + t
        z = cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x, y = cam[:, 0] / z, cam[:, 1] / z
        if dist is not None:
            k1, k2, p1, p2, k3 = dist
            r2 = x * x + y * y
            rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
            x, y = (x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                    y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)
        u, w = K[0, 0] * x + K[0, 2], K[1, 1] * y + K[1, 2]
        gx = 2 * (u * sx / patch) / feat_hw[1] - 1
        gy = 2 * (w * sy / patch) / feat_hw[0] - 1
        zs = z if set_name == "wildtrack" else -z   # MultiviewX camera coordinates are negated
        ok = (zs > 0) & (gx > -1) & (gx < 1) & (gy > -1) & (gy < 1)
        gx, gy = np.where(ok, gx, 2.0), np.where(ok, gy, 2.0)
        grids.append(np.stack([gx, gy], -1).reshape(*grid_hw, 2))
        masks.append(ok.reshape(grid_hw))
    return (torch.tensor(np.stack(grids), dtype=torch.float32),
            torch.tensor(np.stack(masks), dtype=torch.float32))