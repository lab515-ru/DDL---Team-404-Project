import argparse, io, json, os, re, shutil, zipfile
import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--zip", required=True)
p.add_argument("--split", required=True, help="train (training.zip) or validation")
p.add_argument("--scene", required=True)
p.add_argument("--n", type=int, default=2, help="recordings to cache, spread evenly")
p.add_argument("--every", type=int, default=10)
p.add_argument("--out", required=True)
p.add_argument("--list", action="store_true")
a = p.parse_args()
H, W = 270, 480
zf = zipfile.ZipFile(a.zip)
pat = re.compile(rf"^{a.split}/images/([^/]+)/({a.scene}_\d+)\.zip$")
recs = sorted((m.group(1), m.group(2)) for n in zf.namelist() if (m := pat.match(n)))
if a.list:
    print(len(recs), recs)
    raise SystemExit
os.makedirs(f"{a.out}/calib", exist_ok=True)
cal = zf.read(f"{a.split}/calibrations/{a.scene}/calibrations.json")
open(f"{a.out}/calib/{a.scene}.json", "wb").write(cal)
cams = sorted(c["CameraId"] for c in json.loads(cal)["Cameras"])
tmp = f"{a.out}/_tmp.zip"


def inner(name):
    with zf.open(name) as s, open(tmp, "wb") as d:
        shutil.copyfileobj(s, d, 1 << 24)
    return zipfile.ZipFile(tmp)


for i in np.linspace(0, len(recs) - 1, min(a.n, len(recs))).round().astype(int):
    tod, rec = recs[i]
    stem = f"{a.split}__{tod}__{rec}"
    base = f"{a.split}/%s/{tod}/{rec}.zip"
    z = inner(base % "topdown_labels")
    labels = {}
    for n in z.namelist():
        m = re.search(r"topdown_(\d+)\.csv$", n)
        if m:
            rows = [l.split(",") for l in z.read(n).decode().splitlines() if l.strip()]
            labels[m.group(1)] = [[float(r[1]), float(r[2])] for r in rows]
    z.close()
    z = inner(base % "images")
    bn = {os.path.basename(n): n for n in z.namelist()}
    keep = [f for f in sorted(labels)[::a.every] if all(f"rgb_{f}_{c}.jpg" in bn for c in cams)]
    mm = np.lib.format.open_memmap(f"{a.out}/{stem}.npy", mode="w+", dtype=np.uint8,
                                   shape=(len(keep), len(cams), 3, H, W))
    for k, f in enumerate(keep):
        for j, c in enumerate(cams):
            im = Image.open(io.BytesIO(z.read(bn[f"rgb_{f}_{c}.jpg"]))).convert("RGB")
            mm[k, j] = np.asarray(im.resize((W, H), Image.BILINEAR)).transpose(2, 0, 1)
    mm.flush()
    z.close()
    json.dump({"scene": a.scene, "cams": cams, "frames": keep,
               "labels": [labels[f] for f in keep]}, open(f"{a.out}/{stem}.json", "w"))
    print(stem, len(keep), "frames", flush=True)
os.remove(tmp)