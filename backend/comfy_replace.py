"""
Replace Object — built on the shared AI Editing Engine (comfy_engine.py).

Replaces a single selected object with a user-supplied image, while preserving
the object's footprint, the scene's lighting/perspective/shadows, and every
surrounding pixel.

Phase-2 refactor: the orchestration (context crop, mask handling, SDXL working-
size scaling, upload, graph injection, queue/await/fetch, scale-back, feathered
RGBA patch) now lives once in comfy_engine.run_object_edit. This module supplies
only the replace-specific bits: the footprint mask and an init-builder that
composites the uploaded image into that footprint before SDXL harmonises it.

For backward compatibility this module still re-exports the shared helpers
(`_round8`, `_footprint_mask`, …) and `CKPT_NAME`/`LAYERS_DIR` that sibling
modules historically imported from here.

Config (env vars):
  COMFY_REPLACE_CKPT    — checkpoint filename (default sd_xl_base_1.0.safetensors)
  COMFY_REPLACE_TIMEOUT — seconds to wait for a render (default 180)
"""

import os

import cv2
import numpy as np

# Canonical helpers live in the engine; re-exported here for backward
# compatibility (comfy_identity and others still import them from comfy_replace).
from comfy_engine import (  # noqa: F401  (re-exported)
    _round8, _data_url, _decode_rgba, _decode_data_url, _load_metadata,
    _background_rgb, _footprint_mask, _bbox_of, EditError, run_object_edit,
    LAYERS_DIR, CKPT_NAME as ENGINE_CKPT,
)

CKPT_NAME = os.environ.get("COMFY_REPLACE_CKPT", ENGINE_CKPT)
RENDER_TIMEOUT = float(os.environ.get("COMFY_REPLACE_TIMEOUT", "180"))

# Geometry / sampling defaults (all per-request overridable).
CONTEXT_MARGIN_FRAC = 0.35   # context crop padding as a fraction of bbox size
MASK_DILATE = 9              # px; hides anti-aliased object edges
MASK_FEATHER = 9             # px; soft collar so the patch melts into the original
DEFAULTS = {"steps": 28, "cfg": 7.0, "denoise": 0.5,
            "sampler": "dpmpp_2m", "scheduler": "karras", "seed": 0}

DEFAULT_NEGATIVE = ("blurry, lowres, distorted, deformed, artifacts, seam, "
                    "extra objects, watermark, text")


class ReplaceError(Exception):
    """A replace request that cannot be fulfilled (bad object, decode error...)."""


def _default_prompt():
    # Content-neutral on purpose: the replacement is whatever the user uploaded,
    # so the prompt must NOT describe the original object (that would pull the
    # render back toward the thing being removed). It only asks for a clean,
    # well-integrated result that adopts the scene's light/perspective/shadows.
    return ("naturally integrated into the scene, matching the surrounding "
            "lighting, perspective and shadows, seamless, high detail")


def replace_object(object_id, replacement_data_url, options=None, client=None):
    """Replace `object_id` with the uploaded image. Returns the patch descriptor.

    Raises ReplaceError for bad input, ComfyUIError for transport/validation
    failures, TimeoutError if the render stalls.
    """
    options = options or {}
    params = dict(DEFAULTS)
    for k in ("steps", "cfg", "denoise", "sampler", "scheduler", "seed", "ckpt"):
        if options.get(k) is not None:
            params[k] = options[k]

    meta = _load_metadata()
    obj = meta.get(int(object_id))
    if obj is None:
        raise ReplaceError(f"object {object_id} not found")
    if not os.path.exists(os.path.join(LAYERS_DIR, obj.get("file", ""))):
        raise ReplaceError(f"layer file missing: {obj.get('file')}")

    bg = _background_rgb()
    H, W = bg.shape[:2]
    try:
        foot, _ = _footprint_mask(obj, H, W)
    except EditError as exc:
        raise ReplaceError(str(exc))

    # init-builder: composite the uploaded image into the dilated footprint, then
    # let SDXL harmonise it into the scene (the rest of the crop stays original).
    def _init(ctx):
        repl_rgb = _decode_rgba(replacement_data_url)[:, :, :3]
        px0, py0, px1, py1 = _bbox_of(ctx.dil > 0)
        repl_resized = cv2.resize(repl_rgb, (px1 - px0, py1 - py0), interpolation=cv2.INTER_AREA)
        repl_layer = ctx.crop.copy()
        repl_layer[py0:py1, px0:px1] = repl_resized
        m = (ctx.dil.astype(np.float32) / 255.0)[:, :, None]
        return (ctx.crop * (1 - m) + repl_layer * m).astype("uint8")

    positive = options.get("prompt") or _default_prompt()
    negative = options.get("negative") or DEFAULT_NEGATIVE
    try:
        return run_object_edit(
            object_id, bg, foot, params=params, positive=positive, negative=negative,
            slug="replace", ckpt=params.get("ckpt", CKPT_NAME),
            context_margin_frac=CONTEXT_MARGIN_FRAC, dilate=MASK_DILATE, feather=MASK_FEATHER,
            timeout=RENDER_TIMEOUT, init_builder=_init,
            controlnet={"type": "canny", "strength": 0.5, "end": 0.6},
            style=options.get("style") or "auto", material_hint=obj.get("category"), client=client,
            harmonize=bool(options.get("harmonize")), harmonize_strength=options.get("harmonizeStrength"),
            n=options.get("n") or 1, evaluator=bool(options.get("evaluator")),
        )
    except EditError as exc:
        raise ReplaceError(str(exc))
