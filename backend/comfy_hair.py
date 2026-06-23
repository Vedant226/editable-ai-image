"""
Change Hair — built on the shared AI Editing Engine (comfy_engine.py).

Regenerates ONLY a person's hair from a text prompt (style and/or colour), while
keeping the face, eyes, eyebrows, ears, clothing, body and background untouched.

The hair region is derived from the EXISTING masks (no human-parsing model). The
person footprint gives the head silhouette; the head is the part above the chin
(the linked FACE object's bottom), and the hair is that head region MINUS an
expanded face box (eyes/eyebrows/nose/mouth/ears). What remains — the crown, the
hairline and the hair beside/behind the face — is the only thing SDXL may change.

Phase-2 refactor: this module supplies the hair sub-mask + a `postproc` that
hard-protects the face alpha; the rest is the shared engine. `_find_face` is
imported from comfy_clothes (read, not modified).

Config (env vars):
  COMFY_HAIR_CKPT     — SDXL checkpoint (falls back to the engine default)
  COMFY_HAIR_TIMEOUT  — seconds to wait for a render (default 180)
"""

import os

from comfy_engine import (
    _load_metadata, _background_rgb, _footprint_mask, _bbox_of,
    EditError, run_object_edit, LAYERS_DIR, CKPT_NAME as ENGINE_CKPT,
)
from comfy_clothes import _find_face

CKPT_NAME = os.environ.get("COMFY_HAIR_CKPT", ENGINE_CKPT)
RENDER_TIMEOUT = float(os.environ.get("COMFY_HAIR_TIMEOUT", "180"))

CONTEXT_MARGIN_FRAC = 0.35
# Fallback face geometry (when no face object is linked): central-upper region.
FALLBACK_FACE_W = 0.5
FALLBACK_FACE_H = 0.32
FALLBACK_FACE_TOP = 0.10
FACE_EXPAND_X = 0.16       # widen the preserved face box sideways to cover ears
FACE_EXPAND_TOP = 0.05     # keep eyebrows/upper lids
FACE_EXPAND_BOTTOM = 0.06
CHIN_EXTRA = 0.02          # hair may reach a hair below the chin on the sides
MASK_DILATE = 5
MASK_FEATHER = 9
# Hair is GENERATED from the prompt, so denoise is high.
DEFAULTS = {"steps": 30, "cfg": 7.0, "denoise": 0.85,
            "sampler": "dpmpp_2m", "scheduler": "karras", "seed": 0}

DEFAULT_NEGATIVE = ("different face, changed face, deformed face, altered eyes, hat, "
                    "deformed, blurry, lowres, distorted, artifacts, seam, watermark, text")


class HairError(Exception):
    """A change-hair request that cannot be fulfilled."""


def _hair_mask(meta, person, foot, bbox):
    """Head-region footprint minus an expanded face box = the hair region."""
    ox0, oy0, ox1, oy1 = bbox
    bw, bh = ox1 - ox0, oy1 - oy0
    face = _find_face(meta, person)
    if face is not None:
        fx, fy, fw, fh = int(face["x"]), int(face["y"]), int(face["width"]), int(face["height"])
    else:
        fw, fh = int(bw * FALLBACK_FACE_W), int(bh * FALLBACK_FACE_H)
        fx, fy = ox0 + (bw - fw) // 2, oy0 + int(bh * FALLBACK_FACE_TOP)

    # head region: from the top of the silhouette down to ~chin (clothing/neck kept)
    head_bottom = int(round(fy + fh + CHIN_EXTRA * bh))
    head_bottom = max(oy0 + 1, min(head_bottom, oy1))

    # preserved face box (eyes/eyebrows/nose/mouth/ears)
    ex = int(round(fw * FACE_EXPAND_X))
    px0 = max(0, fx - ex)
    px1 = fx + fw + ex
    py0 = max(0, fy - int(round(fh * FACE_EXPAND_TOP)))
    py1 = fy + fh + int(round(fh * FACE_EXPAND_BOTTOM))

    hair = foot.copy()
    hair[head_bottom:, :] = 0          # keep only the head region (above chin)
    hair[py0:py1, px0:px1] = 0         # subtract the preserved face box
    return hair, (px0, py0, min(px1, ox1), min(py1, oy1)), (face is not None)


def change_hair(object_id, prompt, options=None, client=None):
    """Regenerate `object_id`'s hair from `prompt`. Face/clothing/background kept.

    Returns { objectId, x, y, w, h, png, usedFace } — an RGBA patch for the hair
    region only. Everything outside that region is never touched.
    """
    options = options or {}
    if not prompt or not str(prompt).strip():
        raise HairError("a hair prompt is required")
    params = dict(DEFAULTS)
    for k in ("steps", "cfg", "denoise", "sampler", "scheduler", "seed", "ckpt"):
        if options.get(k) is not None:
            params[k] = options[k]
    if options.get("intensity") is not None and options.get("denoise") is None:
        try:
            iv = max(0.0, min(1.0, float(options["intensity"])))
            params["denoise"] = round(max(0.3, min(0.95, DEFAULTS["denoise"] * (0.7 + 0.6 * iv))), 3)
        except (TypeError, ValueError):
            pass

    meta = _load_metadata()
    obj = meta.get(int(object_id))
    if obj is None:
        raise HairError(f"object {object_id} not found")
    if not os.path.exists(os.path.join(LAYERS_DIR, obj.get("file", ""))):
        raise HairError(f"layer file missing: {obj.get('file')}")

    bg = _background_rgb()
    H, W = bg.shape[:2]
    try:
        foot, bbox = _footprint_mask(obj, H, W)
    except EditError as exc:
        raise HairError(str(exc))

    hm, facebox, had_face = _hair_mask(meta, obj, foot, bbox)
    if _bbox_of(hm > 0) is None:
        raise HairError("could not isolate a hair region for this object")

    # postproc: hard-protect the face box so eyes/eyebrows/ears stay byte-identical
    fbx0, fby0, fbx1, fby1 = facebox

    def _protect_face(patch_rgba, ctx):
        ax0, ay0 = max(0, fbx0 - ctx.gx0), max(0, fby0 - ctx.gy0)
        ax1, ay1 = min(ctx.gw, fbx1 - ctx.gx0), min(ctx.gh, fby1 - ctx.gy0)
        if ax1 > ax0 and ay1 > ay0:
            patch_rgba[ay0:ay1, ax0:ax1, 3] = 0
        return patch_rgba

    positive = (str(prompt).strip() +
                ", hair on the person's head, matching the lighting, head shape and "
                "perspective, natural hair texture and strands")
    negative = options.get("negative") or DEFAULT_NEGATIVE
    try:
        return run_object_edit(
            object_id, bg, hm, params=params, positive=positive, negative=negative,
            slug="hair", ckpt=params.get("ckpt", CKPT_NAME),
            context_margin_frac=CONTEXT_MARGIN_FRAC, dilate=MASK_DILATE, feather=MASK_FEATHER,
            timeout=RENDER_TIMEOUT, postproc=_protect_face,
            controlnet={"type": "canny", "strength": 0.4, "end": 0.5},
            style=options.get("style") or "auto", material_hint="hair", client=client,
            harmonize=bool(options.get("harmonize")), harmonize_strength=options.get("harmonizeStrength"),
            n=options.get("n") or 1, evaluator=bool(options.get("evaluator")),
            extra={"usedFace": had_face},
        )
    except EditError as exc:
        raise HairError(str(exc))
