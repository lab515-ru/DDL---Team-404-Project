import json, sys
from PIL import Image, ImageDraw
R = sys.argv[1]; out = sys.argv[2]
for i in (0, 200, 399):
    boxes = json.load(open(f"{R}/annotations_positions/{i:05d}.json"))
    for c in (0, 3):
        im = Image.open(f"{R}/Image_subsets/C{c+1}/{i:04d}.png").convert("RGB")
        d = ImageDraw.Draw(im)
        for p in boxes:
            b = p["views"][c]
            if b["xmin"] >= 0:
                d.rectangle([b["xmin"], b["ymin"], b["xmax"], b["ymax"]], outline=(255, 0, 0), width=4)
        im.resize((960, 540)).save(f"{out}/mx_f{i}_C{c+1}.png")
print("done")
