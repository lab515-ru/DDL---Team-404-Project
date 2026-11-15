import glob, json, os
from mmptrack import load_scene, make_geometry_mmp

R = "/pscratch/sd/l/leana_ab/perlmutter-class/project/data/mmptrack/calib_test"
for sc in ["cafe_shop", "industry_safety", "lobby", "office", "retail"]:
    sp, cams, grid = load_scene(R, sc)
    geo, mask = make_geometry_mmp(R, sc, (270, 480), grid, (16, 30))
    cell = 100 // sp["VoxelSizeInMM"]
    hit = n = masked = 0
    for f in sorted(glob.glob(f"{R}/{sc}/topdown_*.csv")):
        fr = os.path.basename(f)[8:13]
        bxs = {c["CameraId"]: json.load(open(f"{R}/{sc}/rgb_{fr}_{c['CameraId']}.json")) for c in cams}
        for line in open(f):
            if not line.strip():
                continue
            s = line.split(",")
            pid, x, y = s[0].strip(), float(s[1]), float(s[2])
            ri, ci = min(int(x) // cell, grid[0] - 1), min(int(y) // cell, grid[1] - 1)
            for k, c in enumerate(cams):
                b = bxs[c["CameraId"]]
                if pid not in b:
                    continue
                l, t, rr, bt = b[pid]
                if mask[k, ri, ci] == 0:
                    masked += 1
                    continue
                gx, gy = geo[k, ri, ci].tolist()
                u = (gx + 1) / 2 * 30 * 16 / 0.75
                v = (gy + 1) / 2 * 16 * 16 / 0.75
                n += 1
                hit += abs(u - (l + rr) / 2) <= max(25, 0.5 * (rr - l)) and abs(v - bt) <= 30
    print(f"{sc}: cams {len(cams)} grid {grid} inside {hit}/{n}, masked-but-annotated {masked}")