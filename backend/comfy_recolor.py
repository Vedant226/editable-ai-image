"""
Recolor Object — built on the shared AI Editing Engine (comfy_engine.py).

Recolours a selected object to a target colour while keeping it physically
realistic: texture, lighting, shadows, reflections, gradients, highlights,
fabric folds and metallic/glossy speculars are all preserved — only the hue
changes.

How (the important part): pure SDXL recolour flattens objects, and (until Phase
3 installs a ControlNet to lock structure) there is none here. So the colour
change is done first in LAB space — the L (luminance) channel carries ALL
texture/lighting/shadow/specular detail and is kept untouched; only the chroma
(a,b) is rotated toward the target hue, rolled off in the highlights so glossy/
metallic speculars stay white. A LOW-denoise SDXL pass over the footprint then
refines the result without destroying structure.

Phase-2 refactor: the LAB recolour is now the engine's `init_builder`; the
context crop / scaling / SDXL run / feathered patch are shared.

Config (env vars):
  COMFY_RECOLOR_CKPT     — SDXL checkpoint (falls back to the engine default)
  COMFY_RECOLOR_TIMEOUT  — seconds to wait for a render (default 180)
"""

import os
import re

import cv2
import numpy as np

from comfy_engine import (
    _load_metadata, _background_rgb, _footprint_mask, _bbox_of,
    EditError, run_object_edit, LAYERS_DIR, CKPT_NAME as ENGINE_CKPT,
)

CKPT_NAME = os.environ.get("COMFY_RECOLOR_CKPT", ENGINE_CKPT)
RENDER_TIMEOUT = float(os.environ.get("COMFY_RECOLOR_TIMEOUT", "180"))

CONTEXT_MARGIN_FRAC = 0.35
MASK_DILATE = 5             # px; cover the anti-aliased object edge only
MASK_FEATHER = 9            # px; soft collar so the patch melts into the original
HILIGHT_KEEP = 0.72         # above this normalised L, chroma rolls off (speculars stay white)
# Low denoise on purpose: the LAB recolour already changed the colour correctly;
# SDXL only refines. Higher would start to flatten texture.
DEFAULTS = {"steps": 24, "cfg": 6.0, "denoise": 0.28,
            "sampler": "dpmpp_2m", "scheduler": "karras", "seed": 0}

DEFAULT_NEGATIVE = ("flat, plastic, posterized, banding, blurry, lowres, distorted, "
                    "artifacts, seam, different object, watermark, text")


class RecolorError(Exception):
    """A recolor request that cannot be fulfilled (bad object/colour…)."""


def _parse_color(c):
    """Accept '#ff0000', 'ff0000', '#f00', 'rgb(255,0,0)' or [r,g,b] -> (r,g,b)."""
    if isinstance(c, (list, tuple)) and len(c) >= 3:
        return tuple(int(max(0, min(255, round(float(v))))) for v in c[:3])
    s = str(c).strip()
    if s.lower().startswith("rgb"):
        nums = re.findall(r"\d+", s)
        if len(nums) >= 3:
            return tuple(int(max(0, min(255, int(n)))) for n in nums[:3])
    s = s.lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) == 6:
        try:
            return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            pass
    raise RecolorError(f"could not parse targetColor: {c!r}")


def _recolor_lab(crop_rgb, mask01, target_rgb):
    """Luminance-preserving recolour of the masked region toward target_rgb.

    Keeps L (all texture/lighting/shadow/specular detail), rotates chroma to the
    target hue, and rolls chroma off in the highlights so speculars stay white.
    """
    m = mask01[..., 0] if mask01.ndim == 3 else mask01  # ensure 2D (H,W)
    lab = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0]
    a0 = lab[..., 1] - 128.0
    b0 = lab[..., 2] - 128.0

    tlab = cv2.cvtColor(np.uint8([[list(target_rgb)]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    tA, tB = tlab[1] - 128.0, tlab[2] - 128.0
    tC = float(np.hypot(tA, tB))
    if tC < 1e-3:  # target is greyscale -> desaturate (no hue to apply)
        uA = uB = 0.0
    else:
        uA, uB = tA / tC, tB / tC

    Ln = L / 255.0
    # 1.0 below HILIGHT_KEEP, ramps to 0 at L=1 -> specular highlights stay white
    rolloff = np.clip(1.0 - (Ln - HILIGHT_KEEP) / (1.0 - HILIGHT_KEEP), 0.0, 1.0)
    Cbase = np.maximum(tC, np.hypot(a0, b0) * 0.6)
    chroma = Cbase * rolloff

    newA = uA * chroma
    newB = uB * chroma
    A2 = 128.0 + (newA * m + a0 * (1.0 - m))
    B2 = 128.0 + (newB * m + b0 * (1.0 - m))
    out = np.clip(np.stack([L, A2, B2], -1), 0, 255).astype("uint8")
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


def recolor_object(object_id, target_color, options=None, client=None):
    """Recolour `object_id` to `target_color`, preserving texture/lighting.

    Returns { objectId, targetColor, x, y, w, h, png } — an RGBA patch for the
    object's footprint only. The rest of the canvas is never touched.
    """
    options = options or {}
    params = dict(DEFAULTS)
    for k in ("steps", "cfg", "denoise", "sampler", "scheduler", "seed", "ckpt"):
        if options.get(k) is not None:
            params[k] = options[k]
    rgb = _parse_color(target_color)

    meta = _load_metadata()
    obj = meta.get(int(object_id))
    if obj is None:
        raise RecolorError(f"object {object_id} not found")
    if not os.path.exists(os.path.join(LAYERS_DIR, obj.get("file", ""))):
        raise RecolorError(f"layer file missing: {obj.get('file')}")

    bg = _background_rgb()
    H, W = bg.shape[:2]
    try:
        foot, _ = _footprint_mask(obj, H, W)
    except EditError as exc:
        raise RecolorError(str(exc))

    # init-builder: luminance-preserving LAB recolour of the masked footprint
    def _init(ctx):
        mask01 = (ctx.dil.astype(np.float32) / 255.0)[:, :, None]
        return _recolor_lab(ctx.crop, mask01, rgb)

    label = str(obj.get("category") or obj.get("type") or "object").replace("_", " ")
    positive = options.get("prompt") or (
        f"a {label} recoloured rgb({rgb[0]},{rgb[1]},{rgb[2]}), same texture, fabric folds, "
        "metallic reflections, glossy highlights, lighting and shadows")
    negative = options.get("negative") or DEFAULT_NEGATIVE
    try:
        return run_object_edit(
            object_id, bg, foot, params=params, positive=positive, negative=negative,
            slug="recolor", ckpt=params.get("ckpt", CKPT_NAME),
            context_margin_frac=CONTEXT_MARGIN_FRAC, dilate=MASK_DILATE, feather=MASK_FEATHER,
            timeout=RENDER_TIMEOUT, init_builder=_init,
            controlnet={"type": "tile", "strength": 0.6, "end": 0.9},
            style=options.get("style") or "auto", material_hint=obj.get("category"), client=client,
            harmonize=bool(options.get("harmonize")), harmonize_strength=options.get("harmonizeStrength"),
            n=options.get("n") or 1, evaluator=bool(options.get("evaluator")),
            extra={"targetColor": "#%02x%02x%02x" % rgb},
        )
    except EditError as exc:
        raise RecolorError(str(exc))
