"""
Person Replace — built on the shared AI Editing Engine (comfy_engine.py).

Replaces the entire selected person — either with an UPLOADED person image or
with one GENERATED from a text prompt — while preserving the surrounding
background, neighbouring objects, lighting, perspective and shadows. The result
matches the original's scale, camera angle and body orientation.

How it matches geometry without a pose model:
  • Prompt mode denoises FROM the original person (its latent anchors scale,
    framing and camera angle), so the new person inherits the same geometry.
  • Upload mode fits the upload into the person's footprint and CLIPS it to that
    silhouette (which also removes the upload's own background), then SDXL
    harmonises it into the scene's lighting at low denoise.

Phase-2 refactor: the mode-specific init (coarse inpaint + clipped paste for
upload; plain crop for prompt) is the engine's `init_builder`; the rest is the
shared pipeline.

Config (env vars):
  COMFY_PERSON_CKPT     — SDXL checkpoint (falls back to the engine default)
  COMFY_PERSON_TIMEOUT  — seconds to wait for a render (default 180)
"""

import os

import cv2
import numpy as np

from comfy_engine import (
    _decode_rgba, _load_metadata, _background_rgb, _footprint_mask,
    EditError, run_object_edit, LAYERS_DIR, CKPT_NAME as ENGINE_CKPT,
)

CKPT_NAME = os.environ.get("COMFY_PERSON_CKPT", ENGINE_CKPT)
RENDER_TIMEOUT = float(os.environ.get("COMFY_PERSON_TIMEOUT", "180"))

CONTEXT_MARGIN_FRAC = 0.3
MASK_DILATE = 9
MASK_FEATHER = 11
UPLOAD_DENOISE = 0.55      # keep the uploaded person, harmonise lighting/edges
PROMPT_DENOISE = 0.82      # generate a new person, anchored to the original geometry
DEFAULTS = {"steps": 30, "cfg": 7.0, "sampler": "dpmpp_2m", "scheduler": "karras", "seed": 0}

DEFAULT_NEGATIVE = ("extra people, extra limbs, deformed, distorted, blurry, lowres, "
                    "artifacts, seam, floating, cut off, watermark, text")


class PersonError(Exception):
    """A person-replace request that cannot be fulfilled."""


def replace_person(object_id, image=None, prompt=None, options=None, client=None):
    """Replace `object_id` with an uploaded image OR a prompt-generated person.

    Returns { objectId, mode, x, y, w, h, png } — an RGBA patch for the person's
    footprint only. Background and neighbouring objects are never touched.
    """
    options = options or {}
    mode = "upload" if image else ("prompt" if prompt and str(prompt).strip() else None)
    if mode is None:
        raise PersonError("provide an `image` to upload or a `prompt` to generate")

    params = dict(DEFAULTS)
    params["denoise"] = UPLOAD_DENOISE if mode == "upload" else PROMPT_DENOISE
    for k in ("steps", "cfg", "denoise", "sampler", "scheduler", "seed", "ckpt"):
        if options.get(k) is not None:
            params[k] = options[k]
    # intensity (0..1) scales denoise around the mode default (more = more change)
    if options.get("intensity") is not None and options.get("denoise") is None:
        try:
            iv = max(0.0, min(1.0, float(options["intensity"])))
            base = UPLOAD_DENOISE if mode == "upload" else PROMPT_DENOISE
            params["denoise"] = round(max(0.3, min(0.95, base * (0.7 + 0.6 * iv))), 3)
        except (TypeError, ValueError):
            pass

    meta = _load_metadata()
    obj = meta.get(int(object_id))
    if obj is None:
        raise PersonError(f"object {object_id} not found")
    if not os.path.exists(os.path.join(LAYERS_DIR, obj.get("file", ""))):
        raise PersonError(f"layer file missing: {obj.get('file')}")

    bg = _background_rgb()
    H, W = bg.shape[:2]
    try:
        foot, _ = _footprint_mask(obj, H, W)
    except EditError as exc:
        raise PersonError(str(exc))

    if mode == "upload":
        def _init(ctx):
            # erase the old person (coarse), then paste the uploaded person clipped
            # to the footprint silhouette (which discards the upload's background)
            coarse = cv2.inpaint(ctx.crop, (ctx.dil > 0).astype("uint8"), 6, cv2.INPAINT_TELEA)
            up = _decode_rgba(image)
            up_rgb = up[:, :, :3]
            has_alpha = bool((up[:, :, 3] < 250).any())
            up_rgb_r = cv2.resize(up_rgb, (ctx.gw, ctx.gh), interpolation=cv2.INTER_AREA)
            if has_alpha:
                up_a_r = cv2.resize(up[:, :, 3], (ctx.gw, ctx.gh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            else:
                up_a_r = np.ones((ctx.gh, ctx.gw), np.float32)
            rel_x, rel_y = ctx.gx0 - ctx.cx0, ctx.gy0 - ctx.cy0
            person_rgb = coarse.copy()
            person_a = np.zeros((ctx.ch, ctx.cw), np.float32)
            person_rgb[rel_y:rel_y + ctx.gh, rel_x:rel_x + ctx.gw] = up_rgb_r
            person_a[rel_y:rel_y + ctx.gh, rel_x:rel_x + ctx.gw] = up_a_r
            person_a *= (ctx.dil.astype(np.float32) / 255.0)  # clip to footprint silhouette
            a3 = person_a[:, :, None]
            return (coarse * (1 - a3) + person_rgb * a3).astype("uint8")

        init_builder = _init
        positive = (options.get("prompt") or prompt or "a person") + \
            ", matching the scene lighting, perspective and image quality, seamless"
    else:
        init_builder = None  # prompt mode anchors to the original crop
        positive = (str(prompt).strip() +
                    ", a single person, matching the scene lighting, perspective, scale, "
                    "camera angle and body orientation")

    negative = options.get("negative") or DEFAULT_NEGATIVE
    try:
        return run_object_edit(
            object_id, bg, foot, params=params, positive=positive, negative=negative,
            slug="person", ckpt=params.get("ckpt", CKPT_NAME),
            context_margin_frac=CONTEXT_MARGIN_FRAC, dilate=MASK_DILATE, feather=MASK_FEATHER,
            timeout=RENDER_TIMEOUT, init_builder=init_builder,
            controlnet={"type": "canny", "strength": 0.45, "end": 0.6},
            style=options.get("style") or "auto", client=client,
            harmonize=bool(options.get("harmonize")), harmonize_strength=options.get("harmonizeStrength"),
            n=options.get("n") or 1, evaluator=bool(options.get("evaluator")),
            extra={"mode": mode},
        )
    except EditError as exc:
        raise PersonError(str(exc))
