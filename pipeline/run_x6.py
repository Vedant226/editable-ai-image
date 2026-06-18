"""
Phase X6 — backward-compatible metadata writer + coverage report.

Assembles the editor-ready metadata.json from the canonical objects (X3/X4) +
BitmapTextObjects (X5). Schema is a SUPERSET of the current one: the editor
reads `type` (set to an OM-known editable token), geometry and zIndex exactly as
before; the precise `category` and the rest (confidence/importance/editable/
parent/children/source/kind + text fields) are ADDITIVE and ignored by today's
OM. The frozen OM thus keeps working and simply gets cleaner tokens + far more
(and decorative) objects.

Writes pipeline/_out/layers/metadata.json (does NOT touch the live one) and
prints the headline old-vs-new coverage comparison.

  python -m pipeline.run_x6
"""

import json
import os
from collections import Counter

import numpy as np
from PIL import Image

from . import config as C

WORK = C.WORK_DIR
OUT = os.path.join(C.ROOT, "pipeline", "_out", "layers")
OLD_META = os.path.join(C.ROOT, "editable-editor", "public", "layers", "metadata.json")

# precise category -> token the current OM vocabulary recognises as editable
OM_TYPE = {
    "person": "person", "face": "face", "head": "head", "hair": "head", "hand": "face", "eye": "face",
    "crown": "crown", "hat": "crown", "helmet": "crown",
    "sword": "sword", "weapon": "sword", "staff": "sword", "shield": "ornament", "armor": "clothing",
    "clothing": "clothing", "robe": "clothing", "cape": "clothing", "collar": "clothing",
    "book": "ornament", "scroll": "ornament", "banner": "ornament",
    "flower": "flower", "plant": "flower", "tree": "flower", "animal": "animal",
    "logo": "emblem", "emblem": "emblem", "crest": "emblem", "badge": "emblem",
    "ornament": "ornament", "border": "ornament", "frame": "ornament", "embroidery": "ornament",
    "pattern": "ornament", "symbol": "symbol", "icon": "symbol", "jewelry": "ornament",
    "gem": "ornament", "decoration": "decoration", "throne": "ornament", "column": "ornament",
    "text": "text",
    "building": "background", "furniture": "ornament",
    "sky": "sky", "cloud": "sky", "background": "background", "wall": "background",
    "floor": "background", "texture": "texture",
}


def om_type(cat):
    return OM_TYPE.get(cat, "ornament")  # unknown editable cat -> editable decor


def zindex(o):
    cat = o["category"]
    if o.get("kind") == "bitmap_text" or cat == "text":
        return 10
    if cat in ("background", "sky", "cloud", "wall", "floor", "texture"):
        return 0
    if o.get("parent") is not None:
        return 7
    if cat == "person":
        return 4
    return 6


def main():
    image = Image.open(C.IMAGE_PATH).convert("RGB")
    W, H = image.size
    canon = [
        o for o in json.load(open(os.path.join(WORK, "semantic_hier.json")))
        if o.get("isCanonical") and o["editable"] and o["category"] != "text"
    ]
    texts = json.load(open(os.path.join(WORK, "bitmap_text.json")))

    out = [{
        "id": 99999, "file": "background.png", "type": "background", "category": "background",
        "x": 0, "y": 0, "width": W, "height": H, "rotation": 0, "zIndex": 0,
        "confidence": 1.0, "importance": 0.0, "editable": False, "kind": "object",
        "parent": None, "children": [], "source": "base",
    }]

    for o in canon:
        x, y, w, h = o["bbox"]
        out.append({
            "id": o["id"], "file": o["file"], "type": om_type(o["category"]), "category": o["category"],
            "x": x, "y": y, "width": w, "height": h, "rotation": 0, "zIndex": zindex(o),
            "confidence": o["confidence"], "importance": o["importance"], "editable": True,
            "kind": "object", "parent": o.get("parent"), "children": o.get("children", []),
            "source": o.get("source"), "evidence": o.get("evidence"),
        })

    for t in texts:
        out.append({
            "id": t["id"], "file": t["file"], "type": "text", "category": "text",
            "x": t["x"], "y": t["y"], "width": t["width"], "height": t["height"], "rotation": 0,
            "zIndex": 10, "confidence": t["confidence"], "importance": t["importance"], "editable": True,
            "kind": "bitmap_text", "parent": None, "children": [], "source": "ocr",
            "text": t["text"], "fontFamily": t["fontFamily"], "fontSize": t["fontSize"],
            "fontWeight": t["fontWeight"], "fontColor": t["fontColor"], "baseline": t["baseline"],
            "polygon": t["polygon"],
        })

    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "metadata.json"), "w"), indent=1)

    # ---- coverage report (old vs new) ----
    old = json.load(open(OLD_META)) if os.path.exists(OLD_META) else []
    old_obj = [o for o in old if (o.get("type") or "") != "background"]
    new_editable = [o for o in out if o["editable"]]
    decor_cats = {"emblem", "crest", "ornament", "border", "embroidery", "symbol", "logo", "jewelry", "badge"}
    new_decor = [o for o in new_editable if o["category"] in decor_cats]

    print("\n" + "=" * 64)
    print("PHASE X6 — METADATA + COVERAGE")
    print("=" * 64)
    print(f"  OLD metadata objects:      {len(old_obj)}")
    print(f"  NEW metadata objects:      {len(out) - 1}  (+background)")
    print(f"  NEW editable objects:      {len(new_editable)}")
    print(f"  NEW decorative editable:   {len(new_decor)}   (was ~0)")
    print(f"  NEW BitmapTextObjects:     {len(texts)}")
    print("\n  new editable by category:")
    for cat, n in Counter(o["category"] for o in new_editable).most_common():
        print(f"      {cat:14} {n}")
    print(f"\n  metadata -> pipeline/_out/layers/metadata.json")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
