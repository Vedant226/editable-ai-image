"""
Geometric + positional evidence for the fusion engine.

`features` computes interpretable shape descriptors from a mask crop.
`category_scores` turns them into a soft per-category compatibility (0..1),
the "geometry" evidence — e.g. text is wide/thin, a border is hollow and near
the edge, a face is compact/round/high-solidity, an emblem is compact/central.
"""

import cv2
import numpy as np


def _clamp(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else float(x)


def features(crop_mask, bbox, W, H):
    """crop_mask: HxW bool/uint8 cropped to bbox; bbox: [x,y,w,h]."""
    x, y, w, h = bbox
    m = (np.asarray(crop_mask) > 0).astype("uint8")
    area = int(m.sum())
    bbox_area = max(1, w * h)
    extent = area / bbox_area
    aspect = w / h if h else 0.0

    cnts, _ = cv2.findContours(m * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perim = sum(cv2.arcLength(c, True) for c in cnts) or 1.0
    hull_area = sum(cv2.contourArea(cv2.convexHull(c)) for c in cnts) or float(area)
    solidity = area / hull_area if hull_area else 0.0
    circularity = 4 * np.pi * area / (perim * perim) if perim else 0.0

    elongation = 1.0
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        if len(c) >= 5:
            (_, _), (rw, rh), _ = cv2.minAreaRect(c)
            lo, hi = sorted([rw, rh])
            elongation = hi / lo if lo > 0 else 1.0

    return {
        "area": area,
        "areaFraction": area / float(W * H),
        "extent": round(extent, 3),
        "aspect": round(aspect, 3),
        "solidity": round(min(solidity, 1.0), 3),
        "circularity": round(min(circularity, 1.0), 3),
        "elongation": round(elongation, 3),
        "cx": round((x + w / 2) / W, 3),
        "cy": round((y + h / 2) / H, 3),
    }


def category_scores(f):
    a, sol, circ = f["aspect"], f["solidity"], f["circularity"]
    ext, el, af = f["extent"], f["elongation"], f["areaFraction"]
    cx, cy = f["cx"], f["cy"]
    edge = min(cx, 1 - cx, cy, 1 - cy)  # 0 = at edge, 0.5 = center
    central = edge > 0.2

    s = {}
    s["text"] = _clamp((a > 2.0) * 0.5 + (el > 2.5) * 0.3 + (cy < 0.25 or cy > 0.7) * 0.2)
    s["border"] = _clamp((ext < 0.5) * 0.4 + (el > 2.0) * 0.3 + (edge < 0.12) * 0.3)
    s["frame"] = _clamp((ext < 0.55) * 0.5 + (edge < 0.12) * 0.3 + (af > 0.05) * 0.2)
    s["face"] = _clamp((0.6 < a < 1.5) * 0.4 + (sol > 0.85) * 0.3 + (circ > 0.5) * 0.3)
    s["head"] = s["face"]
    s["crown"] = _clamp((1.0 < a < 2.5) * 0.4 + (cy < 0.5) * 0.3 + (sol > 0.6) * 0.3)
    s["emblem"] = _clamp((circ > 0.45) * 0.4 + central * 0.3 + (sol > 0.7) * 0.3)
    s["crest"] = s["emblem"]
    s["symbol"] = _clamp((circ > 0.4) * 0.4 + (af < 0.02) * 0.3 + central * 0.3)
    s["icon"] = s["symbol"]
    s["ornament"] = _clamp((af < 0.02) * 0.4 + (edge < 0.2) * 0.3 + (sol < 0.9) * 0.3)
    s["embroidery"] = _clamp((af < 0.03) * 0.4 + (el > 1.5) * 0.3 + (edge < 0.2) * 0.3)
    s["person"] = _clamp((a < 1.0) * 0.4 + (af > 0.03) * 0.3 + (sol > 0.7) * 0.3)
    s["flower"] = _clamp((circ > 0.5) * 0.4 + (0.6 < a < 1.6) * 0.3 + (af < 0.03) * 0.3)
    s["jewelry"] = _clamp((af < 0.01) * 0.5 + (sol > 0.6) * 0.5)
    return s
