import timm, torch
from pathlib import Path

w = Path(__file__).resolve().parent.parent / "weights" / "vit_base_patch16_224.pth"
m = timm.create_model(
    "vit_base_patch16_224", pretrained=True, num_classes=0,
    img_size=(224, 384),
    pretrained_cfg_overlay=dict(file=str(w)))
x = torch.randn(1, 3, 224, 384)
y = m.forward_features(x)
print("loaded ok; output shape:", tuple(y.shape))
print("patch grid:", m.patch_embed.grid_size)