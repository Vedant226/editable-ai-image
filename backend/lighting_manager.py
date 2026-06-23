"""
Lighting Manager — config + scene-light analysis for the Lighting & Shadow
Harmonization Engine (Phase 10).

Isolated and additive: nothing here imports or changes any existing feature
module. It owns (1) the harmonization config (loaded from
workflows/lighting_harmonization.json — never hardcoded) and (2) the analysis
that measures the lighting of an image region: brightness, exposure, white
balance, scene colour temperature, ambient light, dominant light direction,
local contrast, saturation, gamma and the luminance histogram.

Pure CPU/OpenCV — no SDXL, no network. Designed to run in well under 2 seconds on
a small crop.
"""

import json
import os

import cv2
import numpy as np

CONFIG_PATH = os.environ.get(
    "LIGHTING_CONFIG",
    os.path.join(os.path.dirname(__file__), "workflows", "lighting_harmonization.json"),
)

_cfg_cache = {"data": None}


def config():
    """Load (and cache) the harmonization config. Robust to a missing file."""
    if _cfg_cache["data"] is None:
        try:
            with open(CONFIG_PATH) as fh:
                data = json.load(fh)
            data.pop("_comment", None)
        except Exception:  # noqa: BLE001 - fall back to safe defaults
            data = {}
        _cfg_cache["data"] = data
    return _cfg_cache["data"]


def stage(name, default=None):
    """A stage's config block, merged with sensible defaults."""
    return (config().get(name) or {}) if default is None else {**default, **(config().get(name) or {})}


# -- colour-temperature estimate (McCamy's CCT from average chromaticity) ----

def _cct(mean_rgb):
    r, g, b = [float(c) / 255.0 for c in mean_rgb]
    # linearise sRGB
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = lin(r), lin(g), lin(b)
    X = r * 0.4124 + g * 0.3576 + b * 0.1805
    Y = r * 0.2126 + g * 0.7152 + b * 0.0722
    Z = r * 0.0193 + g * 0.1192 + b * 0.9505
    s = X + Y + Z
    if s <= 1e-6:
        return None
    x, y = X / s, Y / s
    denom = (0.1858 - y)
    if abs(denom) < 1e-6:
        return None
    n = (x - 0.3320) / denom
    cct = 449 * n ** 3 + 3525 * n ** 2 + 6823.3 * n + 5520.33
    return float(max(1000.0, min(40000.0, cct)))


def analyze(rgb, mask=None):
    """Measure the lighting of an RGB region (optionally restricted to `mask`).

    Returns a JSON-friendly dict with every metric the engine reports.
    """
    rgb = rgb[:, :, :3]
    if mask is None:
        sel = np.ones(rgb.shape[:2], bool)
    else:
        sel = mask.astype(bool)
        if int(sel.sum()) < 16:
            sel = np.ones(rgb.shape[:2], bool)
    px = rgb[sel].astype(np.float32)

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0][sel]
    A = lab[:, :, 1][sel] - 128.0
    B = lab[:, :, 2][sel] - 128.0
    chroma = np.sqrt(A * A + B * B)

    mean_rgb = [float(px[:, i].mean()) for i in range(3)]
    bins = int((config().get("analysis") or {}).get("hist_bins", 32))
    hist, _ = np.histogram(L, bins=bins, range=(0, 255))
    hist = (hist / max(1, hist.sum())).astype(float)

    # dominant light direction: luminance gradient (light comes from the bright side)
    Lfull = lab[:, :, 0]
    gx = cv2.Sobel(Lfull, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(Lfull, cv2.CV_32F, 0, 1, ksize=5)
    m2 = sel
    vx, vy = float(gx[m2].mean()), float(gy[m2].mean())
    light_dir = float(np.degrees(np.arctan2(-vy, vx)))  # screen y is down; flip for "up"

    med = float(np.median(L))
    gamma_proxy = float(np.log(max(1e-3, med / 255.0)) / np.log(0.5)) if med > 0 else 1.0

    return {
        "brightness": float(L.mean()),
        "exposure": float(L.mean() / 255.0),
        "ambient_light": float(np.percentile(L, 60)),
        "white_balance": {"r": mean_rgb[0], "g": mean_rgb[1], "b": mean_rgb[2]},
        "color_temperature": _cct(mean_rgb),
        "contrast": float(L.std()),
        "saturation": float(chroma.mean()),
        "gamma": round(gamma_proxy, 3),
        "lab_mean": {"L": float(L.mean()), "a": float(A.mean()), "b": float(B.mean())},
        "lab_std": {"L": float(L.std()), "a": float(A.std()), "b": float(B.std())},
        "chroma_mean": float(chroma.mean()),
        "median_L": med,
        "light_direction_deg": light_dir,
        "histogram": hist.tolist(),
        "pixels": int(sel.sum()),
    }


def histogram_distance(h1, h2):
    """L1 distance between two normalised luminance histograms (0..2)."""
    a, b = np.asarray(h1, float), np.asarray(h2, float)
    n = min(len(a), len(b))
    if n == 0:
        return None
    return float(np.abs(a[:n] - b[:n]).sum())
