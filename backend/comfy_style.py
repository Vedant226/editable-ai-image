"""
comfy_style.py — automatic artistic-style detection (Phase 12, CPU heuristic).

The spec requires edits to match the SURROUNDING image's style and never
introduce a different one. Without a CLIP/aesthetic model installed (and to stay
download-free), this classifies the local crop from cheap image statistics into
one of: photo / oil_painting / watercolor / sketch / anime / digital_art /
illustration, and returns the prompt suffix that keeps an edit in that style.

It is deliberately CONSERVATIVE: "photo" is only chosen when the crop is clearly
photographic (real high-frequency detail + sensor-like noise), otherwise it falls
back through the painterly buckets to a generic "illustration". So on painted /
illustrated artwork an edit never gets pushed toward photoreal.

Used by comfy_engine.run_object_edit when style="auto" (the default); the fixed
styles "preserve"/"photoreal"/"none" bypass detection.
"""

import cv2
import numpy as np

# Prompt suffix per detected style (what keeps the edit in the same style).
STYLE_SUFFIXES = {
    "photo": "photorealistic, realistic photo, natural lighting, sharp focus",
    "oil_painting": "in the same oil-painting style, visible brush strokes, painterly texture",
    "watercolor": "in the same watercolor style, soft washes, blended pigments, paper texture",
    "sketch": "in the same hand-drawn pencil sketch style, line work, graphite shading, monochrome",
    "anime": "in the same anime / cel-shaded style, clean line art, flat vibrant colours",
    "digital_art": "in the same digital illustration style, clean rendering, smooth shading",
    "illustration": ("in the same painted illustration art style as the original artwork, "
                     "consistent brushwork, texture and colour palette"),
}


def _colorfulness(rgb):
    """Hasler–Süsstrunk colourfulness metric."""
    r, g, b = rgb[:, :, 0].astype("float"), rgb[:, :, 1].astype("float"), rgb[:, :, 2].astype("float")
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def features(rgb):
    """Cheap style features of an RGB crop (downscaled for speed)."""
    h, w = rgb.shape[:2]
    s = 256.0 / max(1, max(h, w))
    small = cv2.resize(rgb, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype("float") / 255.0
    edges = cv2.Canny(gray, 80, 180)
    blur = cv2.GaussianBlur(gray.astype("float"), (0, 0), 2)
    localvar = cv2.GaussianBlur((gray.astype("float") - blur) ** 2, (0, 0), 2)
    med = cv2.medianBlur(gray, 3)
    q = (small // 32).reshape(-1, 3)
    return {
        "sat": float(sat.mean()),
        "edge": float((edges > 0).mean()),
        "lap": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "noise": float(np.abs(gray.astype("float") - med.astype("float")).mean()),
        "flat": float((localvar < 8).mean()),
        "colorful": _colorfulness(small),
        "ncolors": int(len(np.unique(q, axis=0))),
    }


def classify(f):
    """Map style features -> a style key (conservative; defaults to illustration)."""
    # near-monochrome + line-dominated -> pencil sketch
    if f["sat"] < 0.12 and f["edge"] > 0.06:
        return "sketch"
    # vivid + large flat regions + few distinct colours + crisp -> anime / cel
    if f["sat"] > 0.45 and f["flat"] > 0.55 and f["ncolors"] < 1200:
        return "anime"
    # clearly photographic: real detail + sensor-like noise + a RICH colour palette
    # (the key art-vs-photo signal — illustrations use few quantized colours, photos
    # hundreds). Without the palette gate, detailed digital paintings read as photos.
    if f["noise"] > 3.0 and f["lap"] > 130 and f["flat"] < 0.45 and f["ncolors"] > 180:
        return "photo"
    # soft, low-edge, low-detail, still coloured -> watercolor
    if f["edge"] < 0.035 and f["lap"] < 80 and f["sat"] > 0.15:
        return "watercolor"
    # textured + colourful, limited palette -> oil painting
    if f["lap"] > 60 and f["colorful"] > 25 and f["ncolors"] < 180:
        return "oil_painting"
    # vivid + smooth digital render
    if f["sat"] > 0.35 and f["flat"] > 0.4:
        return "digital_art"
    return "illustration"


def detect_style(rgb):
    """Return (style_key, feature_report) for an RGB crop. Never raises."""
    try:
        f = features(rgb)
        return classify(f), f
    except Exception:  # noqa: BLE001
        return "illustration", {}
