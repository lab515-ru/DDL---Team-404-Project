import json
from geometry import make_geometry

R = "/pscratch/sd/l/leana_ab/perlmutter-class/project/data/wildtrack/Wildtrack_dataset"
grid, mask = make_geometry("wildtrack", R, (270, 480), (360, 120), (16, 30))
print("grid", tuple(grid.shape), "visible fraction per camera:", [round(m.mean().item(), 3) for m in mask])
people = json.load(open(R + "/annotations_positions/00000000.json"))
hits = n = invisible = 0
for p in people:
    pid = p["positionID"]
    gy, gx = (pid // 480) * 360 // 1440, (pid % 480) * 120 // 480
    for c in range(7):
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
print(f"sample points inside annotated box: {hits}/{n}; masked-out although annotated: {invisible}")