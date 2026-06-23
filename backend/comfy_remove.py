"""
Remove Object — built on the shared AI Editing Engine (comfy_engine.py).

The inverse of Replace Object: instead of compositing an uploaded image into the
footprint, it ERASES the object and generates a realistic background fill that
continues the surrounding scene, matching its lighting and leaving every
neighbouring pixel (and shadow) untouched.

Phase-2 refactor: the only remove-specific step — a coarse OpenCV inpaint that
gives SDXL an object-FREE starting point (so it refines a background, never
re-imagines the object) — is now the engine's `init_builder`. Everything else is
the shared pipeline.

Config (env vars):
  COMFY_REMOVE_CKPT     — SDXL checkpoint (falls back to the engine default)
  COMFY_REMOVE_TIMEOUT  — seconds to wait for a render (default 180)
"""

import base64
import io
import json
import os
import urllib.request

import cv2
import numpy as np
from PIL import Image

from comfy_engine import (
    _load_metadata, _background_rgb, _footprint_mask, _bbox_of, _data_url,
    EditError, run_object_edit, LAYERS_DIR, CKPT_NAME as ENGINE_CKPT,
)

CKPT_NAME = os.environ.get("COMFY_REMOVE_CKPT", ENGINE_CKPT)
RENDER_TIMEOUT = float(os.environ.get("COMFY_REMOVE_TIMEOUT", "180"))
LIFT_URL = os.environ.get("LIFT_URL", "http://127.0.0.1:8000")  # the LaMa inpaint service
LAMA_REFINE_DENOISE = 0.5   # LaMa already fills cleanly, so SDXL only refines

CONTEXT_MARGIN_FRAC = 0.4   # more context helps the fill match the surroundings
MASK_DILATE = 13            # px; "expand mask slightly" — also hides the object edge
MASK_FEATHER = 11           # px; soft collar so the fill melts into the original
DEFAULTS = {"steps": 28, "cfg": 7.0, "denoise": 0.82,
            "sampler": "dpmpp_2m", "scheduler": "karras", "seed": 0}

DEFAULT_POSITIVE = ("seamless empty background, natural continuation of the surrounding "
                    "scene, consistent lighting, colour and shadows, highly detailed, "
                    "no object")
DEFAULT_NEGATIVE = ("object, person, figure, text, emblem, logo, blurry, lowres, "
                    "distorted, artifacts, seam, duplicate, watermark")


class RemoveError(Exception):
    """A remove request that cannot be fulfilled (bad object, empty footprint…)."""


def _lama_fill(object_id):
    """Clean background fill from the LaMa lift service (:8000), or None.

    LaMa is purpose-built for object removal, so it gives SDXL a far better
    object-free starting point than OpenCV TELEA. Best-effort: any failure
    (service down, OOM, timeout) returns None and the caller falls back.
    Returns (rgb_array, x, y, w, h) at full-canvas coordinates.
    """
    try:
        urllib.request.urlopen(LIFT_URL + "/health", timeout=1.5).read()
        req = urllib.request.Request(
            LIFT_URL + "/inpaint", data=json.dumps({"objectId": int(object_id)}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as r:
            j = json.load(r)
        png = j.get("png", "")
        s = png.split(",", 1)[1] if "," in png else png
        if not s:
            return None
        arr = np.array(Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB"))
        return arr, int(j["x"]), int(j["y"]), int(j["w"]), int(j["h"])
    except Exception:  # noqa: BLE001
        return None


def _direct_lama_patch(object_id, bg, foot, lama, dilate, feather):
    """Build the removal patch DIRECTLY from the LaMa background fill (no SDXL).

    LaMa is purpose-built for object removal and produces a clean, object-free
    continuation of the surrounding scene. Running base SDXL over it — even at a
    low denoise — only re-hallucinates structure (faces, text) back into the
    footprint (the model here has no inpaint/ControlNet conditioning). So when a
    LaMa fill is available we return IT as the patch, with a feathered footprint
    alpha so it melts into the untouched original pixels. Deterministic, clean and
    fast — a true erase, not a re-imagine.
    """
    H, W = bg.shape[:2]
    gbox = _bbox_of(foot > 0)
    if gbox is None:
        raise RemoveError("object footprint is empty")
    gx0, gy0, gx1, gy1 = gbox
    gw, gh = gx1 - gx0, gy1 - gy0

    # Dilate the footprint a touch (hides the object's own anti-aliased edge),
    # TELEA-fill it as a safe base, then lay the higher-quality LaMa fill on top.
    dil = cv2.dilate(foot, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate)))
    filled = cv2.inpaint(bg, (dil > 0).astype("uint8"), 6, cv2.INPAINT_TELEA)
    arr, lx, ly, lw, lh = lama
    dx0, dy0 = max(0, lx), max(0, ly)
    dx1, dy1 = min(W, lx + lw), min(H, ly + lh)
    if dx1 > dx0 and dy1 > dy0:
        filled[dy0:dy1, dx0:dx1] = arr[(dy0 - ly):(dy1 - ly), (dx0 - lx):(dx1 - lx)]

    # feathered footprint alpha — soft collar so the fill blends seamlessly
    fk = int(feather) | 1
    alpha = cv2.GaussianBlur(dil.astype(np.float32), (fk, fk), 0) / 255.0

    patch_rgb = filled[gy0:gy1, gx0:gx1]
    patch_a = np.clip(alpha[gy0:gy1, gx0:gx1] * 255.0, 0, 255).astype("uint8")
    patch_rgba = np.dstack([patch_rgb, patch_a]).astype("uint8")
    return {
        "objectId": int(object_id),
        "x": int(gx0), "y": int(gy0), "w": int(gw), "h": int(gh),
        "png": _data_url(patch_rgba),
        "engine": "lama",
        "coarseFill": "lama",
        "upgrades": {"style": None, "controlnet": None, "soft_inpaint": False, "harmonized": False},
        "harmonized": False,
    }


def remove_object(object_id, options=None, client=None):
    """Erase `object_id` and fill its footprint with realistic background.

    Returns { objectId, x, y, w, h, png } — an RGBA patch to drop onto the
    object's footprint. The rest of the canvas is never touched.

    Default path is a DIRECT LaMa fill (a clean erase). Pass options.forceSdxl to
    fall back to the SDXL background-generation path (legacy behaviour).
    """
    options = options or {}
    params = dict(DEFAULTS)
    for k in ("steps", "cfg", "denoise", "sampler", "scheduler", "seed", "ckpt"):
        if options.get(k) is not None:
            params[k] = options[k]

    meta = _load_metadata()
    obj = meta.get(int(object_id))
    if obj is None:
        raise RemoveError(f"object {object_id} not found")
    if not os.path.exists(os.path.join(LAYERS_DIR, obj.get("file", ""))):
        raise RemoveError(f"layer file missing: {obj.get('file')}")

    bg = _background_rgb()
    H, W = bg.shape[:2]
    try:
        foot, _ = _footprint_mask(obj, H, W)
    except EditError as exc:
        raise RemoveError(str(exc))

    # Prefer a LaMa fill (purpose-built for removal); fall back to OpenCV TELEA.
    lama = _lama_fill(object_id)

    # DEFAULT: a clean, deterministic erase straight from the LaMa fill (no SDXL
    # re-imagining of the footprint). Only the SDXL path is taken when LaMa is
    # unavailable or the caller explicitly asks for it (forceSdxl).
    if lama is not None and not options.get("forceSdxl"):
        try:
            return _direct_lama_patch(object_id, bg, foot, lama, MASK_DILATE, MASK_FEATHER)
        except RemoveError:
            raise
        except Exception:  # noqa: BLE001 - on any failure, fall through to SDXL
            pass

    if lama is not None and options.get("denoise") is None:
        params["denoise"] = LAMA_REFINE_DENOISE  # clean base -> SDXL only refines

    def _init(ctx):
        # TELEA covers the whole dilated region; LaMa (if any) overwrites the
        # footprint area with its higher-quality fill.
        base = cv2.inpaint(ctx.crop, (ctx.dil > 0).astype("uint8"), 6, cv2.INPAINT_TELEA)
        if lama is not None:
            arr, lx, ly, lw, lh = lama
            rx, ry = lx - ctx.cx0, ly - ctx.cy0
            # intersection of the LaMa patch with the crop
            dx0, dy0 = max(0, rx), max(0, ry)
            dx1, dy1 = min(ctx.cw, rx + lw), min(ctx.ch, ry + lh)
            if dx1 > dx0 and dy1 > dy0:
                base[dy0:dy1, dx0:dx1] = arr[(dy0 - ry):(dy1 - ry), (dx0 - rx):(dx1 - rx)]
        return base

    positive = options.get("prompt") or DEFAULT_POSITIVE
    negative = options.get("negative") or DEFAULT_NEGATIVE
    try:
        return run_object_edit(
            object_id, bg, foot, params=params, positive=positive, negative=negative,
            slug="remove", ckpt=params.get("ckpt", CKPT_NAME),
            context_margin_frac=CONTEXT_MARGIN_FRAC, dilate=MASK_DILATE, feather=MASK_FEATHER,
            timeout=RENDER_TIMEOUT, init_builder=_init,
            style=options.get("style") or "auto", client=client,
            harmonize=bool(options.get("harmonize")), harmonize_strength=options.get("harmonizeStrength"),
            n=options.get("n") or 1, evaluator=bool(options.get("evaluator")),
            extra={"coarseFill": "lama" if lama is not None else "telea"},
        )
    except EditError as exc:
        raise RemoveError(str(exc))
