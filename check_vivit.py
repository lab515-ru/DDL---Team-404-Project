from pathlib import Path
import torch
from huggingface_hub import snapshot_download

REPO = "google/vivit-b-16x2-kinetics400"      # from memory, unverified
out = Path("/pscratch/sd/l/leana_ab/perlmutter-class/project/weights/vivit")
p = snapshot_download(repo_id=REPO, local_dir=out)
print("downloaded to", p)
for f in sorted(out.rglob("*")):
    if f.is_file():
        print(f.name, round(f.stat().st_size / 1e6), "MB")
from transformers import VivitModel
m = VivitModel.from_pretrained(out).eval()
c = m.config
print("frames", c.num_frames, "image", c.image_size, "tubelet", c.tubelet_size,
      "hidden", c.hidden_size, "params M", round(sum(x.numel() for x in m.parameters()) / 1e6, 1))
with torch.no_grad():
    print("32 frames ->", tuple(m(pixel_values=torch.randn(1, 32, 3, 224, 224)).last_hidden_state.shape))
    try:
        print("4 frames ->", tuple(m(pixel_values=torch.randn(1, 4, 3, 224, 224)).last_hidden_state.shape))
    except Exception as e:
        print("4 frames FAILED:", str(e)[:200])
