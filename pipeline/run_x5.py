"""
Phase X5 — text -> BitmapTextObject.

OCR (EasyOCR) finds text regions. For each we store a BitmapTextObject that
PRESERVES THE ORIGINAL PIXELS (an RGBA crop), so text is visually identical to
the source until the user edits it. We also estimate enough to synthesize
typography on edit:
  text (OCR string), polygon, baseline, fontSize, fontWeight, fontColor.

Output: pipeline/_out/layers/text_<i>_text.png  +  pipeline/_work/bitmap_text.json
(font family detection is best-effort 'serif'; the bitmap carries the true look.)

  python -m pipeline.run_x5
"""

import json
import os

import numpy as np
from PIL import Image
import easyocr

from . import config as C

OUT = os.path.join(C.ROOT, "pipeline", "_out", "layers")
TEXT_ID_BASE = 5000
MIN_CONF = 0.30


def estimate_style(crop_rgb):
    arr = crop_rgb.reshape(-1, 3).astype(np.float32)
    med = np.median(arr, axis=0)  # background ≈ most common colour
    dist = np.linalg.norm(arr - med, axis=1)
    thr = dist.mean() + dist.std()
    text_pix = arr[dist >= thr]
    frac = float((dist >= thr).mean())
    color = text_pix.mean(axis=0) if len(text_pix) else (255 - med)
    hexcol = "#%02x%02x%02x" % tuple(int(max(0, min(255, c))) for c in color)
    return hexcol, ("bold" if frac > 0.28 else "normal")


def main():
    os.makedirs(OUT, exist_ok=True)
    image = np.array(Image.open(C.IMAGE_PATH).convert("RGB"))
    H, W = image.shape[:2]

    reader = easyocr.Reader(["en"], gpu=(C.DEVICE == "cuda"))
    results = reader.readtext(C.IMAGE_PATH)

    objs = []
    for i, (poly, text, conf) in enumerate(results):
        if conf < MIN_CONF or not text.strip():
            continue
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x0, y0 = max(0, int(min(xs))), max(0, int(min(ys)))
        x1, y1 = min(W, int(max(xs))), min(H, int(max(ys)))
        w, h = x1 - x0, y1 - y0
        if w < 4 or h < 4:
            continue

        crop = image[y0:y1, x0:x1]
        hexcol, weight = estimate_style(crop)

        oid = TEXT_ID_BASE + i
        fname = f"text_{i}_text.png"
        rgba = np.dstack([crop, np.full((h, w), 255, "uint8")]).astype("uint8")
        Image.fromarray(rgba, "RGBA").save(os.path.join(OUT, fname))

        objs.append(
            {
                "id": oid,
                "file": fname,
                "type": "text",
                "category": "text",
                "kind": "bitmap_text",
                "text": text.strip(),
                "bbox": [x0, y0, w, h],
                "x": x0, "y": y0, "width": w, "height": h, "rotation": 0,
                "confidence": round(float(conf), 3),
                "importance": round(min(1.0, 0.85 * (0.5 + 0.5 * float(conf))), 3),
                "editable": True,
                "isCanonical": True,
                "aliasOf": None,
                "parent": None,
                "children": [],
                "source": "ocr",
                # typography estimation (the bitmap preserves the true look until edited)
                "fontFamily": "serif",
                "fontSize": int(round(h * 0.95)),
                "fontWeight": weight,
                "fontColor": hexcol,
                "baseline": y0 + int(round(h * 0.82)),
                "polygon": [[int(p[0]), int(p[1])] for p in poly],
            }
        )

    json.dump(objs, open(os.path.join(C.WORK_DIR, "bitmap_text.json"), "w"), indent=1)

    print("\n" + "=" * 64)
    print("PHASE X5 — TEXT -> BitmapTextObject")
    print("=" * 64)
    print(f"  OCR regions: {len(results)}   kept as BitmapTextObjects: {len(objs)}")
    print(f"  bitmaps -> pipeline/_out/layers/text_*_text.png   meta -> _work/bitmap_text.json")
    print("  samples (text · size · weight · color):")
    for o in objs[:14]:
        print(f"      #{o['id']}  \"{o['text'][:28]:28}\"  {o['fontSize']}px {o['fontWeight']} {o['fontColor']}")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
