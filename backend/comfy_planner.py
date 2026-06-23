"""
comfy_planner.py — the Universal AI Edit Planner (Phase 14).

A single, additive intelligence layer that every AI edit passes through before
generation. It does NOT replace the engine or any feature — the shared engine
(comfy_engine.run_object_edit) consults it when plan=True, so every feature that
goes through the engine gets it for free, with zero feature-module changes.

Pipeline contributed here (the rest — crop, mask, generation, blend, validate,
retry — already lives in the engine):

  analyze()  : style (comfy_style) + MATERIAL detection + lighting/scene stats
  augment()  : material- and lighting-aware additions to the positive/negative
               prompt (texture / specular / fold / shadow / perspective
               preservation) — purely additive prompt optimisation
  refine()   : a safe, hue-neutral high-frequency detail-recovery pass on the
               generated patch (luminance unsharp; never undoes a recolour)

Everything is CPU and model-free, so it adds negligible latency and needs no
download. Material is inferred from pixel statistics (works for ANY object
category — no hardcoded per-feature assumptions); a feature may also pass an
explicit `material_hint`.
"""

import cv2
import numpy as np

import comfy_style

# material -> (positive preservation terms, negative terms)
_MATERIAL_TERMS = {
    "metallic": ("preserve metallic sheen, specular highlights and reflections", "dull, matte, flat metal"),
    "fabric":   ("natural fabric folds, weave and stitching", "stiff plastic fabric"),
    "skin":     ("natural skin texture and even tone", "plastic skin, waxy, blurry skin"),
    "hair":     ("natural hair strands and flow", "blurry clumped hair"),
    "foliage":  ("natural leaf and plant texture", "plastic-looking plant"),
    "glass":    ("transparent glass with realistic refraction and highlights", "opaque muddy glass"),
    "stone":    ("natural stone and rock texture", "smooth plastic surface"),
    "matte":    ("consistent surface texture and material appearance", "glossy plastic, artificial"),
}

_HINT_ALIASES = {  # feature/category hint -> canonical material
    "skin": "skin", "face": "skin", "person": "skin",
    "hair": "hair", "beard": "hair",
    "cloth": "fabric", "clothes": "fabric", "fabric": "fabric", "garment": "fabric",
    "metal": "metallic", "gold": "metallic", "silver": "metallic", "jewel": "metallic",
    "crown": "metallic", "crest": "metallic", "emblem": "metallic", "logo": "metallic", "coin": "metallic",
    "glass": "glass", "plant": "foliage", "tree": "foliage", "leaf": "foliage", "foliage": "foliage",
    "stone": "stone", "rock": "stone", "wall": "stone", "brick": "stone",
}


def _material_from_stats(crop_rgb, mask_crop):
    """Infer a coarse material from pixel statistics of the masked region."""
    m = mask_crop > 0
    if int(m.sum()) < 20:
        return "matte"
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    sat = (hsv[:, :, 1].astype("float") / 255.0)[m]
    val = (hsv[:, :, 2].astype("float") / 255.0)[m]
    hue = (hsv[:, :, 0].astype("float"))[m]
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    tex = float(np.abs(cv2.Laplacian(gray, cv2.CV_64F))[m].mean())
    spec = float((val > 0.92).mean())          # bright specular fraction
    sat_m = float(sat.mean())
    green = float(((hue > 35) & (hue < 85)).mean()) if hue.size else 0.0

    # Conservative: only commit to a specific material on a STRONG signal — fine
    # texture alone is unreliable on detailed artwork, so we never guess "fabric"
    # / "skin" from stats (those come from a feature hint). Unknown -> "matte",
    # whose term is generic and always safe ("preserve surface texture/material").
    if spec > 0.18 and tex < 14:
        return "glass"
    if green > 0.5 and tex > 10:
        return "foliage"
    if spec > 0.10 and sat_m < 0.35:
        return "metallic"
    return "matte"


def _material(crop_rgb, mask_crop, material_hint=None):
    if material_hint:
        h = str(material_hint).lower()
        for k, v in _HINT_ALIASES.items():
            if k in h:
                return v
    return _material_from_stats(crop_rgb, mask_crop)


def _lighting(crop_rgb, mask_crop):
    """Brightness / warmth / contrast of the SURROUNDINGS (so an edit can match)."""
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2].astype("float") / 255.0
    ring = mask_crop == 0
    r = crop_rgb[:, :, 0].astype("float")
    b = crop_rgb[:, :, 2].astype("float")
    use = ring if ring.any() else np.ones_like(ring, bool)
    return {
        "brightness": round(float(v[use].mean()), 3),
        "warmth": round(float((r - b)[use].mean()), 1),
        "contrast": round(float(v.std()), 3),
    }


def analyze(crop_rgb, mask_crop, material_hint=None):
    """Full scene/object analysis for one edit. Never raises."""
    try:
        style, sfeat = comfy_style.detect_style(crop_rgb)
        material = _material(crop_rgb, mask_crop, material_hint)
        lighting = _lighting(crop_rgb, mask_crop)
        return {"style": style, "material": material, "lighting": lighting, "styleFeatures": sfeat}
    except Exception:  # noqa: BLE001
        return {"style": "illustration", "material": "matte", "lighting": {}}


def augment(positive, negative, analysis):
    """Append material/lighting/perspective preservation terms (additive)."""
    pos, neg = _MATERIAL_TERMS.get(analysis.get("material", "matte"), ("", ""))
    lit = analysis.get("lighting", {})
    b = lit.get("brightness")
    light_term = ("matching the scene lighting" if b is None
                  else "bright matching lighting" if b > 0.6
                  else "soft dim matching lighting" if b < 0.32
                  else "matching the scene lighting")
    extra_pos = ", ".join(t for t in [pos, light_term, "preserve original shadows and perspective"] if t)
    positive2 = f"{positive.rstrip(' ,')}, {extra_pos}" if extra_pos else positive
    negative2 = f"{negative.rstrip(' ,')}, {neg}" if neg else negative
    return positive2, negative2


def refine(patch_rgb, analysis=None):
    """Safe, hue-neutral high-frequency detail recovery on the generated patch.

    Unsharp masks the LUMINANCE channel only (LAB-L), so it recovers detail lost
    to the VAE round-trip + downscale WITHOUT shifting hue/saturation — i.e. it
    never undoes a recolour. Skips tiny patches (would amplify artefacts).
    """
    h, w = patch_rgb.shape[:2]
    if min(h, w) < 24:
        return patch_rgb
    try:
        lab = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2LAB)
        L = lab[:, :, 0].astype("float32")
        blur = cv2.GaussianBlur(L, (0, 0), 1.0)
        L2 = np.clip(L + 0.35 * (L - blur), 0, 255)  # gentle unsharp
        lab[:, :, 0] = L2.astype("uint8")
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    except Exception:  # noqa: BLE001
        return patch_rgb
