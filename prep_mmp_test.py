import json, os, shutil, zipfile

Z = "training.zip"
OUT = "calib_test"
TMP = OUT + "/_tmp.zip"
SCENES = ["cafe_shop", "industry_safety", "lobby", "office", "retail"]
zf = zipfile.ZipFile(Z)
names = zf.namelist()


def inner(name):
    with zf.open(name) as s, open(TMP, "wb") as d:
        shutil.copyfileobj(s, d)
    return zipfile.ZipFile(TMP)


for sc in SCENES:
    img = sorted(n for n in names if n.startswith("train/images/") and n.endswith(".zip")
                 and "/" + sc + "_" in n)[0]
    rec = os.path.basename(img)[:-4]
    d = f"{OUT}/{sc}"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/calibrations.json", "wb") as f:
        f.write(zf.read(f"train/calibrations/{sc}/calibrations.json"))
    z = inner(img.replace("/images/", "/topdown_labels/"))
    frames = sorted(n.split("topdown_")[1][:-4] for n in z.namelist() if n.endswith(".csv"))
    pick = [frames[len(frames) // 4], frames[len(frames) // 2], frames[3 * len(frames) // 4]]
    for fr in pick:
        open(f"{d}/topdown_{fr}.csv", "wb").write(z.read(f"{rec}/topdown_{fr}.csv"))
    z.close()
    z = inner(img.replace("/images/", "/labels/"))
    for fr in pick:
        for n in z.namelist():
            if f"rgb_{fr}_" in n and n.endswith(".json"):
                open(f"{d}/{os.path.basename(n)}", "wb").write(z.read(n))
    z.close()
    z = inner(img)
    for fr in pick:
        for n in z.namelist():
            if f"rgb_{fr}_" in n and n.endswith(".jpg"):
                open(f"{d}/{os.path.basename(n)}", "wb").write(z.read(n))
    z.close()
    print(sc, rec, "frames", pick, flush=True)
os.remove(TMP)
