#!/usr/bin/env python3
"""Download timm pretrained weights on a login node and save as a local file.
Usage: python download_weights.py [model_name]
"""
import sys
from pathlib import Path

import timm
import torch

name = sys.argv[1] if len(sys.argv) > 1 else "vit_base_patch16_224"
out_dir = Path(__file__).resolve().parent.parent / "weights"
out_dir.mkdir(exist_ok=True)

m = timm.create_model(name, pretrained=True)
path = out_dir / f"{name}.pth"
torch.save(m.state_dict(), path)

n_params = sum(p.numel() for p in m.parameters()) / 1e6
print(f"saved {path}")
print(f"{n_params:.1f}M params, {path.stat().st_size / 1e6:.0f} MB")