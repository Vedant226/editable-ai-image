"""
Phase X3 runner — dedup + hierarchy over the labeled proposals.

Reads pipeline/_work/semantic.json (from X2), collapses duplicate masks, builds
the person -> parts hierarchy, and writes pipeline/_work/semantic_hier.json with
parent / children / aliasOf / isCanonical added. Reports stats + a colour
overlay (parents/parts/decor) and a grouped HTML.

  python -m pipeline.run_x3
"""

import base64
import json
import os
from collections import Counter

import numpy as np
from PIL import Image

from . import config as C
from . import hierarchy as Hy

WORK = C.WORK_DIR
MASKS_DIR = os.path.join(WORK, "cand_masks")
CROPS_DIR = os.path.join(WORK, "crops")


def load_full_masks(objs, H, W):
    masks = {}
    for o in objs:
        x, y, w, h = o["bbox"]
        crop = np.array(Image.open(os.path.join(MASKS_DIR, f"cand_{o['id']}.png")).convert("L")) > 0
        m = np.zeros((H, W), bool)
        m[y : y + h, x : x + w] = crop[: h, : w]
        masks[o["id"]] = m
    return masks


def main():
    image = np.array(Image.open(C.IMAGE_PATH).convert("RGB"))
    H, W = image.shape[:2]
    objs = json.load(open(os.path.join(WORK, "semantic.json")))
    by_id = {o["id"]: o for o in objs}
    masks = load_full_masks(objs, H, W)

    canonical_ids, alias = Hy.dedup(objs, masks)
    for o in objs:
        o["isCanonical"] = o["id"] in canonical_ids
        o["aliasOf"] = alias.get(o["id"])
        o.setdefault("parent", None)
        o.setdefault("children", [])
        if o["aliasOf"] is not None:
            o["editable"] = False  # retained but not separately editable

    parents = Hy.build_hierarchy(by_id, canonical_ids, masks)
    json.dump(objs, open(os.path.join(WORK, "semantic_hier.json"), "w"), indent=1)

    # ---- stats ----
    canon = [o for o in objs if o["isCanonical"]]
    canon_editable = [o for o in canon if o["editable"]]
    parts_assigned = [o for o in canon if o.get("parent") is not None]
    standalone = [o for o in canon if o["editable"] and o.get("parent") is None and o["category"] not in Hy.PARENT_CATS]

    print("\n" + "=" * 64)
    print("PHASE X3 — DEDUP + HIERARCHY")
    print("=" * 64)
    print(f"  proposals in:        {len(objs)}")
    print(f"  duplicates collapsed: {len(alias)}  (retained as aliases)")
    print(f"  canonical objects:    {len(canon)}   (editable {len(canon_editable)})")
    print(f"  persons (parents):    {len(parents)}")
    print(f"  parts with a parent:  {len(parts_assigned)}")
    print(f"  standalone editable:  {len(standalone)}")
    print(f"  old metadata.json had 108 objects; X1 proposed 251.")
    print("\n  canonical editable by category:")
    for cat, n in Counter(o["category"] for o in canon_editable).most_common():
        print(f"      {cat:14} {n}")
    print("\n  hierarchy (person -> parts):")
    for p in parents:
        if p["children"]:
            kids = Counter(by_id[c]["category"] for c in p["children"])
            print(f"      person#{p['id']}: " + ", ".join(f"{k}x{v}" for k, v in kids.items()))

    _overlay(image, objs, by_id, masks)
    _html(objs, by_id, parents)
    print(f"\n  semantic+hierarchy -> pipeline/_work/semantic_hier.json")
    print(f"  overlay            -> pipeline/_work/x3_overlay.png")
    print(f"  HTML               -> pipeline/_work/x3_report.html")
    print("=" * 64 + "\n")


def _overlay(image, objs, by_id, masks):
    out = image.astype(np.float32).copy()
    for o in objs:
        if not o["isCanonical"] or not o["editable"]:
            continue
        if o["category"] in Hy.PARENT_CATS:
            col = np.array([235, 70, 70], np.float32)      # person = red
        elif o.get("parent") is not None:
            col = np.array([80, 220, 90], np.float32)       # child part = green
        else:
            col = np.array([80, 170, 235], np.float32)      # standalone decor = blue
        m = masks[o["id"]]
        out[m] = out[m] * 0.45 + col * 0.55
    Image.fromarray(np.clip(out, 0, 255).astype("uint8")).save(os.path.join(WORK, "x3_overlay.png"))


def _thumb(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _card(o):
    crop = _thumb(os.path.join(CROPS_DIR, f"crop_{o['id']}.png"))
    return (
        f"<div class='c'><img src='{crop}'>"
        f"<div>{o['category']} <span class='m'>#{o['id']} {o['confidence']}</span></div></div>"
    )


def _html(objs, by_id, parents):
    canon = [o for o in objs if o["isCanonical"] and o["editable"]]
    standalone = [o for o in canon if o.get("parent") is None and o["category"] not in Hy.PARENT_CATS]
    parts = [
        "<html><head><meta charset='utf-8'><style>",
        "body{background:#15110d;color:#e8d9bf;font:13px system-ui,sans-serif;margin:18px}",
        "h2{color:#d8b36a}.group{border:1px solid #3a2f1c;border-radius:8px;padding:8px;margin:8px 0}",
        ".row{display:flex;flex-wrap:wrap;gap:6px;align-items:flex-start}",
        ".c{width:90px;text-align:center}.c img{width:84px;height:84px;object-fit:contain;background:#0c0a07;border-radius:4px}",
        ".m{color:#9a8a70}.par{outline:2px solid #eb4646}", "</style></head><body>",
        f"<h1 style='color:#d8b36a'>Phase X3 — Dedup + Hierarchy</h1>",
    ]
    for p in parents:
        pcrop = _thumb(os.path.join(CROPS_DIR, f"crop_{p['id']}.png"))
        parts.append(f"<div class='group'><h2>person #{p['id']} &middot; {len(p['children'])} parts</h2><div class='row'>")
        parts.append(f"<div class='c par'><img src='{pcrop}'><div>person <span class='m'>#{p['id']}</span></div></div>")
        for cid in p["children"]:
            parts.append(_card(by_id[cid]))
        parts.append("</div></div>")
    parts.append(f"<div class='group'><h2>standalone decor &middot; {len(standalone)}</h2><div class='row'>")
    for o in sorted(standalone, key=lambda o: -o["importance"]):
        parts.append(_card(o))
    parts.append("</div></div></body></html>")
    with open(os.path.join(WORK, "x3_report.html"), "w") as f:
        f.write("".join(parts))


if __name__ == "__main__":
    main()
