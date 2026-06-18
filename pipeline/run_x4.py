"""
Phase X4 — mask refinement -> RGBA cutout PNGs.

For each canonical object, refine the binary mask into a soft alpha matte
(guided filter, reusing backend/matting.py) and emit an RGBA PNG cropped to the
object's footprint — a drop-in for the editor's per-object layers, but with soft
/ thin / semi-transparent edges. (Foreground decontamination is applied later by
the editor's /lift at edit time, where a clean background estimate exists.)

Output: pipeline/_out/layers/<id>_<category>.png  + background.png
The chosen filename is written back into semantic_hier.json (`file`).

  python -m pipeline.run_x4
"""

import json
import os
import sys

import numpy as np
from PIL import Image

from . import config as C

sys.path.insert(0, os.path.join(C.ROOT, "backend"))
from matting import refine_matte  # noqa: E402  (reuse the editor's matting)

WORK = C.WORK_DIR
MASKS_DIR = os.path.join(WORK, "cand_masks")
OUT = os.path.join(C.ROOT, "pipeline", "_out", "layers")
PAD = 4


def main():
    os.makedirs(OUT, exist_ok=True)
    image = np.array(Image.open(C.IMAGE_PATH).convert("RGB"))
    H, W = image.shape[:2]
    objs = json.load(open(os.path.join(WORK, "semantic_hier.json")))
    canon = [o for o in objs if o.get("isCanonical")]

    Image.fromarray(image).save(os.path.join(OUT, "background.png"))

    soft_total = written = 0
    samples = {}
    for o in canon:
        x, y, w, h = o["bbox"]
        mx0, my0 = max(0, x - PAD), max(0, y - PAD)
        mx1, my1 = min(W, x + w + PAD), min(H, y + h + PAD)
        crop = image[my0:my1, mx0:mx1]

        cm = np.array(Image.open(os.path.join(MASKS_DIR, f"cand_{o['id']}.png")).convert("L")) > 0
        binb = np.zeros((my1 - my0, mx1 - mx0), bool)
        binb[(y - my0) : (y - my0) + h, (x - mx0) : (x - mx0) + w] = cm[:h, :w]

        alpha = refine_matte(crop, binb.astype("uint8"), radius=max(4, round(min(w, h) * 0.04)))

        fy, fx = y - my0, x - mx0
        rgb_fp = crop[fy : fy + h, fx : fx + w]
        a_fp = np.clip(alpha[fy : fy + h, fx : fx + w] * 255, 0, 255).astype("uint8")
        rgba = np.dstack([rgb_fp, a_fp]).astype("uint8")

        fname = f"{o['id']}_{o['category']}.png"
        Image.fromarray(rgba, "RGBA").save(os.path.join(OUT, fname))
        o["file"] = fname
        written += 1
        soft_total += int(((a_fp > 16) & (a_fp < 239)).sum())
        if o["category"] not in samples and o["editable"]:
            samples[o["category"]] = fname

    json.dump(objs, open(os.path.join(WORK, "semantic_hier.json"), "w"), indent=1)

    print("\n" + "=" * 64)
    print("PHASE X4 — MASK REFINEMENT -> RGBA CUTOUTS")
    print("=" * 64)
    print(f"  RGBA cutouts written: {written}  -> pipeline/_out/layers/")
    print(f"  background.png written")
    print(f"  avg soft-edge px/cutout: {soft_total / max(1, written):.0f}  (>0 confirms soft mattes)")
    print(f"  sample files per category:")
    for cat, fn in sorted(samples.items()):
        print(f"      {cat:14} {fn}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
