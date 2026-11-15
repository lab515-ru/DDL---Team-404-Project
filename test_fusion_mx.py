import json
from geometry import make_geometry

R = "/pscratch/sd/l/leana_ab/perlmutter-class/project/data/multiviewx/MultiviewX"
grid, mask = make_geometry("multiviewx", R, (270, 480), (160, 250), (16, 30))
print("grid", tuple(grid.shape), "visible:", [round(m.mean().item(), 3) for m in mask])
hits = n = invisible = 0
for fr in (0, 100, 200, 300, 399):
    for p in json.load(open(f"{R}/annotations_positions/{fr:05d}.json")):
        pid = p["positionID"]
        gy, gx = (pid // 1000) * 160 // 640, (pid % 1000) * 250 // 1000
        for c in range(6):
            b = p["views"][c]
            if b["xmin"] < 0:
                continue
            if mask[c, gy, gx] == 0:
                invisible += 1
                continue
            gxn, gyn = grid[c, gy, gx].tolist()
            u = (gxn + 1) / 2 * 30 * 16 / 0.25
            v = (gyn + 1) / 2 * 16 * 16 / 0.25
            n += 1
            hits += (b["xmin"] - 40 <= u <= b["xmax"] + 40) and (b["ymin"] <= v <= b["ymax"] + 60)
print(f"inside box: {hits}/{n}; masked out although annotated: {invisible}")