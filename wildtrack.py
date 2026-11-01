import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CAMS = ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"]
ORIG_HW = (1080, 1920)


def _floats(txt):
    return np.array(txt.split(), dtype=np.float64)


def load_calib(root):
    """Return per-camera K (3x3), R (3x3), t (3,) for the undistorted images."""
    calib = []
    for c in CAMS:
        e = ET.parse(Path(root) / "calibrations/extrinsic" / f"extr_{c}.xml").getroot()
        rvec = _floats(e.find("rvec").text)
        tvec = _floats(e.find("tvec").text)
        i = ET.parse(Path(root) / "calibrations/intrinsic_zero" / f"intr_{c}.xml").getroot()
        K = _floats(i.find("camera_matrix/data").text).reshape(3, 3)
        calib.append((K, rodrigues(rvec), tvec))
    return calib


def rodrigues(r):
    th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def id_to_xy_cm(pid):
    """positionID -> world x, y in cm (README: 2.5 cm cells, origin (-300, -900) cm)."""
    return -300.0 + 2.5 * (pid % 480), -900.0 + 2.5 * (pid // 480)


def project(calib_cam, xyz_cm):
    """World points (N,3), in cm, -> pixel coords (N,2) in the 1920x1080 image."""
    K, R, t = calib_cam
    cam = xyz_cm @ R.T + t
    uv = cam @ K.T
    return uv[:, :2] / uv[:, 2:3]


class WildtrackDataset(Dataset):
    """Returns views (7,3,H,W) float in [0,1] and target (1, grid_h, grid_w) occupancy."""

    def __init__(self, root, img_hw=(270, 480), grid_hw=(360, 120), cache=None):
        self.root, self.hw, self.grid = Path(root), img_hw, grid_hw
        self.frames = sorted(p.stem for p in (self.root / "annotations_positions").glob("*.json"))
        self.calib = load_calib(self.root)
        self.cache = None
        if cache:
            self.cache = np.load(cache, mmap_mode="r")
            assert self.cache.shape == (len(self.frames), 7, 3, *self.hw), self.cache.shape

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        f = self.frames[i]
        if self.cache is not None:
            views = torch.from_numpy(np.array(self.cache[i])).float() / 255
        else:
            views = []
            for c in range(7):
                im = Image.open(self.root / "Image_subsets" / f"C{c + 1}" / f"{f}.png").convert("RGB")
                im = im.resize((self.hw[1], self.hw[0]), Image.BILINEAR)
                views.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255)
            views = torch.stack(views)
        target = torch.zeros(1, *self.grid)
        for p in json.load(open(self.root / "annotations_positions" / f"{f}.json")):
            pid = p["positionID"]
            gy = (pid // 480) * self.grid[0] // 1440
            gx = (pid % 480) * self.grid[1] // 480
            target[0, gy, gx] = 1.0
        return views, target