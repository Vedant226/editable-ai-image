"""
Universal Extraction Engine — the production, generalized, single-process entry
point. Loads every model ONCE and runs the full X1->X6 pipeline in memory for an
arbitrary image, with no image-specific logic. The per-phase run_xN scripts
remain as diagnostic tools; this is what production + benchmarking call.

  from pipeline.engine import load_all, process_image
  load_all()
  summary = process_image("some.png", "out_dir/")

process_image is defensive: any single proposal/stage failure is contained so a
bad mask never aborts the whole image (graceful degradation).
"""

import json
import os
import sys
import time

import numpy as np
from PIL import Image

from . import config as C
from . import proposals as P
from . import clip_labeler as CL
from . import hierarchy as Hy
from .geometry import features as gfeat, category_scores as gscore
from .fusion import fuse, map_phrase_to_category
from .run_x5 import estimate_style
from .run_x6 import om_type, zindex

sys.path.insert(0, os.path.join(C.ROOT, "backend"))
from matting import refine_matte  # noqa: E402

_OCR = {}
PAD = 4


def _reader():
    if "r" not in _OCR:
        import easyocr

        # CPU OCR: leaves the GPU to SAM; EasyOCR is fast enough on CPU for offline use
        _OCR["r"] = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _OCR["r"]


def load_all():
    """Warm every model once (SAM+DINO, CLIP, OCR)."""
    P.load_models()
    CL.load_clip()
    _reader()


def _iou_bbox(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _obj(c, fname):
    x, y, w, h = c["bbox"]
    return {
        "id": c["id"], "file": fname, "type": om_type(c["category"]), "category": c["category"],
        "x": x, "y": y, "width": w, "height": h, "rotation": 0, "zIndex": zindex(c),
        "confidence": c["confidence"], "importance": c["importance"], "editable": True,
        "kind": "object", "parent": c.get("parent"), "children": c.get("children", []),
        "source": c.get("source"), "evidence": c.get("evidence"),
    }


def process_image(image_path, out_dir):
    timings = {}
    t0 = time.time()
    image = np.array(Image.open(image_path).convert("RGB"))
    H, W = image.shape[:2]
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(image).save(os.path.join(out_dir, "background.png"))

    # X1 — proposals
    t = time.time()
    _, sam, dino = P.generate(image_path)
    cands = sam + dino
    for i, c in enumerate(cands):
        c["id"] = i
    masks = {c["id"]: c["mask"] for c in cands}
    dino_cands = [c for c in cands if c["source"] == "dino"]
    timings["proposals"] = round(time.time() - t, 1)

    # X2 — semantic fusion labeling (defensive per candidate)
    t = time.time()
    for c in cands:
        try:
            x, y, w, h = c["bbox"]
            cm = c["mask"][y : y + h, x : x + w]
            masked = np.where(cm[:, :, None], image[y : y + h, x : x + w], 128).astype("uint8")
            clip_probs = CL.classify(Image.fromarray(masked))
            gf = gfeat(cm, c["bbox"], W, H)
            gs = gscore(gf)
            if c["source"] == "dino":
                dcat, dstr = map_phrase_to_category(c["label"]), float(c["score"])
            else:
                best, bi = None, 0.0
                for d in dino_cands:
                    o = _iou_bbox(c["bbox"], d["bbox"])
                    if o > bi:
                        best, bi = d, o
                dcat, dstr = (map_phrase_to_category(best["label"]), float(best["score"]) * bi) if (best and bi > 0.4) else (None, 0.0)
            c.update(fuse(clip_probs, dcat, dstr, gf, gs))
            c["geometry"] = gf
        except Exception as exc:  # noqa: BLE001 — never let one mask abort the image
            c.update({"category": "unknown", "confidence": 0.0, "importance": 0.0,
                      "editable": False, "kind": "object", "evidence": {}, "geometry": {"solidity": 0}})
            print(f"  [warn] label failed for candidate {c['id']}: {exc}")
    timings["label"] = round(time.time() - t, 1)

    # X3 — dedup + hierarchy
    t = time.time()
    canonical_ids, alias = Hy.dedup(cands, masks)
    by_id = {c["id"]: c for c in cands}
    for c in cands:
        c["isCanonical"] = c["id"] in canonical_ids
        c["aliasOf"] = alias.get(c["id"])
        c.setdefault("parent", None)
        c.setdefault("children", [])
        if c["aliasOf"] is not None:
            c["editable"] = False
    Hy.build_hierarchy(by_id, canonical_ids, masks)
    timings["hierarchy"] = round(time.time() - t, 1)

    # X4 — RGBA cutouts for canonical editable non-text
    t = time.time()
    final = [{
        "id": 99999, "file": "background.png", "type": "background", "category": "background",
        "x": 0, "y": 0, "width": W, "height": H, "rotation": 0, "zIndex": 0,
        "confidence": 1.0, "importance": 0.0, "editable": False, "kind": "object",
        "parent": None, "children": [], "source": "base",
    }]
    cov = np.zeros((H, W), bool)
    for c in cands:
        if not c["isCanonical"] or not c["editable"] or c["category"] == "text":
            continue
        try:
            x, y, w, h = c["bbox"]
            mx0, my0 = max(0, x - PAD), max(0, y - PAD)
            mx1, my1 = min(W, x + w + PAD), min(H, y + h + PAD)
            crop = image[my0:my1, mx0:mx1]
            binb = np.zeros((my1 - my0, mx1 - mx0), bool)
            binb[(y - my0) : (y - my0) + h, (x - mx0) : (x - mx0) + w] = masks[c["id"]][y : y + h, x : x + w]
            alpha = refine_matte(crop, binb.astype("uint8"), radius=max(4, round(min(w, h) * 0.04)))
            fy, fx = y - my0, x - mx0
            rgba = np.dstack([crop[fy : fy + h, fx : fx + w], np.clip(alpha[fy : fy + h, fx : fx + w] * 255, 0, 255).astype("uint8")]).astype("uint8")
            fname = f"{c['id']}_{c['category']}.png"
            Image.fromarray(rgba, "RGBA").save(os.path.join(out_dir, fname))
            final.append(_obj(c, fname))
            cov |= masks[c["id"]]
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] cutout failed for {c['id']}: {exc}")
    timings["matte"] = round(time.time() - t, 1)

    # X5 — text BitmapTextObjects (graceful on OCR failure)
    t = time.time()
    n_text = 0
    try:
        results = _reader().readtext(image_path)
    except Exception as exc:  # noqa: BLE001
        results = []
        print(f"  [warn] OCR failed: {exc}")
    for i, (poly, text, conf) in enumerate(results):
        if conf < 0.30 or not str(text).strip():
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
        fname = f"text_{i}_text.png"
        Image.fromarray(np.dstack([crop, np.full((h, w), 255, "uint8")]).astype("uint8"), "RGBA").save(os.path.join(out_dir, fname))
        final.append({
            "id": 5000 + i, "file": fname, "type": "text", "category": "text",
            "x": x0, "y": y0, "width": w, "height": h, "rotation": 0, "zIndex": 10,
            "confidence": round(float(conf), 3), "importance": round(min(1.0, 0.85 * (0.5 + 0.5 * float(conf))), 3),
            "editable": True, "kind": "bitmap_text", "parent": None, "children": [], "source": "ocr",
            "text": str(text).strip(), "fontFamily": "serif", "fontSize": int(round(h * 0.95)),
            "fontWeight": weight, "fontColor": hexcol, "baseline": y0 + int(round(h * 0.82)),
            "polygon": [[int(p[0]), int(p[1])] for p in poly],
        })
        cov[y0:y1, x0:x1] = True
        n_text += 1
    timings["ocr"] = round(time.time() - t, 1)

    json.dump(final, open(os.path.join(out_dir, "metadata.json"), "w"), indent=1)
    timings["total"] = round(time.time() - t0, 1)

    editable = [o for o in final if o.get("editable")]
    confs = [c["confidence"] for c in cands if c.get("isCanonical") and c.get("editable")]
    sol = [c["geometry"].get("solidity", 0) for c in cands if c.get("isCanonical") and c.get("editable") and c.get("geometry")]
    return {
        "image": os.path.basename(image_path), "size": [W, H], "timings": timings,
        "proposals": len(cands), "canonical": len(canonical_ids), "aliases": len(alias),
        "editable": len(editable), "bitmap_text": n_text,
        "dedup_ratio": round(len(alias) / max(1, len(cands)), 3),
        "editable_coverage": round(float(cov.sum()) / float(W * H), 3),
        "avg_confidence": round(float(np.mean(confs)), 3) if confs else 0.0,
        "pct_uncertain": round(float(np.mean([c < 0.5 for c in confs])), 3) if confs else 0.0,
        "avg_solidity": round(float(np.mean(sol)), 3) if sol else 0.0,
    }
