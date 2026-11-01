import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from wildtrack import rodrigues, _floats, project

NCAM = 6
GRID_W, GRID_H = 640, 1000  # labeling grid (guess: X-first); verify with the test


def load_calib_mx(root):
    calib = []
    for c in range(1, NCAM + 1):
        e = ET.parse(Path(root) / "calibrations/extrinsic" / f"extr_Camera{c}.xml").getroot()
        rvec = _floats(e.find("rvec/data").text)
        tvec = _floats(e.find("tvec/data").text)
        i = ET.parse(Path(root) / "calibrations/intrinsic" / f"intr_Camera{c}.xml").getroot()
        K = _floats(i.find("camera_matrix/data").text).reshape(3, 3)
        calib.append((K, rodrigues(rvec), tvec))
    return calib

class MultiviewXDataset(Dataset):
    """views (6,3,H,W) in [0,1]; target (1,grid_h,grid_w). Rows = Y, cols = X."""

    def __init__(self, root, img_hw=(270, 480), grid_hw=(160, 250), cache=None):
        self.root, self.hw, self.grid = Path(root), img_hw, grid_hw
        self.frames = sorted(p.stem for p in (self.root / "annotations_positions").glob("*.json"))
        self.calib = load_calib_mx(self.root)
        self.cache = None
        if cache:
            self.cache = np.load(cache, mmap_mode="r")
            assert self.cache.shape == (len(self.frames), NCAM, 3, *self.hw), self.cache.shape

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        f = self.frames[i]
        if self.cache is not None:
            views = torch.from_numpy(np.array(self.cache[i])).float() / 255
        else:
            vs = []
            for c in range(NCAM):
                im = Image.open(self.root / "Image_subsets" / f"C{c + 1}" / f"{int(f):04d}.png").convert("RGB")
                im = im.resize((self.hw[1], self.hw[0]), Image.BILINEAR)
                vs.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255)
            views = torch.stack(vs)
        target = torch.zeros(1, *self.grid)
        for p in json.load(open(self.root / "annotations_positions" / f"{f}.json")):
            pid = p["positionID"]
            gy = (pid // GRID_W) * self.grid[0] // GRID_H
            gx = (pid % GRID_W) * self.grid[1] // GRID_W
            target[0, gy, gx] = 1.0
        return views, target