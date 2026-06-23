"""
Logo effects — built on the shared AI Editing Engine (comfy_engine.py).

Material finishes for a selected logo/emblem object, footprint-only (the rest of
the canvas is never touched):

  * metallic / glass — an SDXL material restyle that PRESERVES the logo's shape
    and text: a low-denoise pass with a strong tile ControlNet (locks structure)
    plus the planner's material terms. Reuses run_object_edit, so it inherits the
    Phase-14 quality controls (harmonize / best-of-N / evaluator) for free.
  * emboss / transparent — pure CPU (no SDXL): emboss applies a relief filter to
    the footprint; transparent returns the logo cut to its own alpha (clean
    transparent background). Both are instant and never call ComfyUI.

Replace / Recolor logos already work via comfy_replace / comfy_recolor; this adds
the material-finish features without touching either.

Config (env vars):
  COMFY_LOGO_CKPT     — SDXL checkpoint (falls back to the engine default)
  COMFY_LOGO_TIMEOUT  — seconds to wait for a render (default 180)
"""

import os

import cv2
import numpy as np

from comfy_engine import (
    _load_metadata, _background_rgb, _footprint_mask, _bbox_of, _data_url,
    EditError, run_object_edit, LAYERS_DIR, CKPT_NAME as ENGINE_CKPT,
)

CKPT_NAME = os.environ.get("COMFY_LOGO_CKPT", ENGINE_CKPT)
RENDER_TIMEOUT = float(os.environ.get("COMFY_LOGO_TIMEOUT", "180"))

CONTEXT_MARGIN_FRAC = 0.35
MASK_DILATE = 5
MASK_FEATHER = 9
DEFAULTS = {"steps": 26, "cfg": 6.5, "denoise": 0.45,
            "sampler": "dpmpp_2m", "scheduler": "karras", "seed": 0}
DEFAULT_NEGATIVE = ("blurry, lowres, distorted, deformed shape, changed text, extra letters, "
                    "artifacts, seam, watermark")

# SDXL material finishes (shape/text preserved by a strong tile ControlNet).
LOGO_FEATURES = {
    "metallic": {"material": "metallic", "denoise": 0.45,
                 "prompt": ("polished metallic chrome finish, reflective brushed metal, "
                            "specular highlights, same exact logo shape and text")},
    "glass": {"material": "glass", "denoise": 0.45,
              "prompt": ("glossy translucent glass finish, subtle reflections and refraction, "
                         "clean highlights, same exact logo shape and text")},
}


class LogoError(Exception):
    """A logo edit that cannot be fulfilled (bad object / unknown feature)."""


def _resolve_logo(object_id):
    meta = _load_metadata()
    obj = meta.get(int(object_id))
    if obj is None:
        raise LogoError(f"object {object_id} not found")
    if not os.path.exists(os.path.join(LAYERS_DIR, obj.get("file", ""))):
        raise LogoError(f"layer file missing: {obj.get('file')}")
    bg = _background_rgb()
    H, W = bg.shape[:2]
    try:
        foot, _ = _footprint_mask(obj, H, W)
    except EditError as exc:
        raise LogoError(str(exc))
    return obj, bg, foot


def _cpu_logo(feature, object_id, bg, foot):
    """Instant CPU effects (emboss / transparent) — footprint RGBA patch."""
    box = _bbox_of(foot > 0)
    if box is None:
        raise LogoError("logo footprint is empty")
    x0, y0, x1, y1 = box
    crop = bg[y0:y1, x0:x1]
    fk = MASK_FEATHER | 1
    alpha = cv2.GaussianBlur(foot[y0:y1, x0:x1].astype(np.float32), (fk, fk), 0).clip(0, 255).astype("uint8")
    if feature == "emboss":
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
        emb = cv2.filter2D(gray, -1, kernel) + 128.0
        emb = np.clip(emb, 0, 255).astype("uint8")
        rgb = cv2.cvtColor(emb, cv2.COLOR_GRAY2RGB)
    else:  # transparent -> the logo on a clean transparent background
        rgb = crop
    rgba = np.dstack([rgb, alpha]).astype("uint8")
    return {"objectId": int(object_id), "x": int(x0), "y": int(y0),
            "w": int(x1 - x0), "h": int(y1 - y0), "png": _data_url(rgba),
            "engine": "cpu", "promptId": None, "feature": feature, "harmonized": False}


def logo_edit(object_id, feature, options=None, client=None):
    """Apply a logo finish (metallic / glass / emboss / transparent).

    Returns { objectId, feature, x, y, w, h, png } — an RGBA footprint patch.
    """
    options = options or {}
    feature = str(feature or "").lower()
    obj, bg, foot = _resolve_logo(object_id)

    if feature in ("emboss", "transparent"):
        return _cpu_logo(feature, object_id, bg, foot)

    cfg = LOGO_FEATURES.get(feature)
    if cfg is None:
        raise LogoError(f"unknown logo feature: {feature!r}")

    params = dict(DEFAULTS)
    params["denoise"] = cfg["denoise"]
    for k in ("steps", "cfg", "denoise", "sampler", "scheduler", "seed", "ckpt"):
        if options.get(k) is not None:
            params[k] = options[k]
    if options.get("intensity") is not None and options.get("denoise") is None:
        try:
            iv = max(0.0, min(1.0, float(options["intensity"])))
            params["denoise"] = round(max(0.2, min(0.8, cfg["denoise"] * (0.6 + 0.8 * iv))), 3)
        except (TypeError, ValueError):
            pass

    positive = options.get("prompt") or cfg["prompt"]
    negative = options.get("negative") or DEFAULT_NEGATIVE
    try:
        return run_object_edit(
            object_id, bg, foot, params=params, positive=positive, negative=negative,
            slug=f"logo_{feature}", ckpt=params.get("ckpt", CKPT_NAME),
            context_margin_frac=CONTEXT_MARGIN_FRAC, dilate=MASK_DILATE, feather=MASK_FEATHER,
            timeout=RENDER_TIMEOUT,
            controlnet={"type": "tile", "strength": 0.7, "end": 0.9},
            style=options.get("style") or "auto", material_hint=cfg["material"], client=client,
            harmonize=bool(options.get("harmonize")), harmonize_strength=options.get("harmonizeStrength"),
            n=options.get("n") or 1, evaluator=bool(options.get("evaluator")),
            extra={"feature": feature},
        )
    except EditError as exc:
        raise LogoError(str(exc))
