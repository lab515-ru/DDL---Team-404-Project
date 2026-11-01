import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image

root, out = Path(sys.argv[1]), Path(sys.argv[2])
H, W, V = 270, 480, 6
frames = sorted(p.stem for p in (root / "annotations_positions").glob("*.json"))


def load(args):
    i, f = args
    arr = np.empty((V, 3, H, W), np.uint8)
    for c in range(V):
        im = Image.open(root / "Image_subsets" / f"C{c + 1}" / f"{int(f):04d}.png").convert("RGB")
        arr[c] = np.asarray(im.resize((W, H), Image.BILINEAR)).transpose(2, 0, 1)
    return i, arr


if __name__ == "__main__":
    mm = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=(len(frames), V, 3, H, W))
    with Pool(4) as pool:
        for k, (i, arr) in enumerate(pool.imap_unordered(load, enumerate(frames))):
            mm[i] = arr
            if k % 50 == 0:
                print(f"{k}/{len(frames)}", flush=True)
    mm.flush()
    print("saved", out, mm.shape)