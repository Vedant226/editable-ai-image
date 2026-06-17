"""
Phase X1 runner — generate the universal proposal pool and report coverage.

  python -m pipeline.run_x1      (from the repo root)
"""

import os
from collections import Counter

import numpy as np
from PIL import Image

from . import config as C
from .proposals import generate, save_candidates

os.makedirs(C.WORK_DIR, exist_ok=True)

print(f"device: {C.DEVICE}")
print("running SAM-AMG + GroundingDINO ... (this is the heavy step)")
image_rgb, sam, dino = generate()
H, W = image_rgb.shape[:2]
total = float(H * W)
candidates = sam + dino
save_candidates(candidates)


def buckets(cs):
    b = {"tiny <0.1%": 0, "small <2%": 0, "medium <25%": 0, "large >25%": 0}
    for c in cs:
        f = c["area"] / total
        if f < 0.001:
            b["tiny <0.1%"] += 1
        elif f < 0.02:
            b["small <2%"] += 1
        elif f < 0.25:
            b["medium <25%"] += 1
        else:
            b["large >25%"] += 1
    return b


print("\n" + "=" * 64)
print(f"PHASE X1 — UNIVERSAL PROPOSALS   image {W}x{H}")
print("=" * 64)
print(f"  SAM-AMG (segment everything): {len(sam)}")
print(f"  GroundingDINO -> SAM:          {len(dino)}")
print(f"  TOTAL candidates:              {len(candidates)}   (today's metadata: 108 objects)")
print(f"  size buckets (all):           {buckets(candidates)}")
labels = Counter(c["label"] for c in dino if c["label"])
print(f"  DINO labels (top 20):         {dict(labels.most_common(20))}")

# overlay: random colour per mask over the original (coverage visualization)
rng = np.random.default_rng(7)
overlay = image_rgb.astype(np.float32).copy()
for c in candidates:
    col = rng.integers(50, 255, 3).astype(np.float32)
    m = c["mask"]
    overlay[m] = overlay[m] * 0.45 + col * 0.55
Image.fromarray(np.clip(overlay, 0, 255).astype("uint8")).save(os.path.join(C.WORK_DIR, "x1_overlay.png"))
print(f"\n  overlay  -> {os.path.relpath(os.path.join(C.WORK_DIR, 'x1_overlay.png'), C.ROOT)}")
print(f"  masks    -> pipeline/_work/cand_masks/  ({len(candidates)} PNGs)")
print(f"  manifest -> pipeline/_work/candidates.json")
print("=" * 64 + "\n")
