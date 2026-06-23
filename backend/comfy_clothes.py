"""
Change Clothes — built on the shared AI Editing Engine (comfy_engine.py).

Regenerates ONLY a person's clothing from a text prompt, while keeping the face,
hair, body pose, hands, legs and background untouched.

The clothing region is derived from the EXISTING masks — there is no human-
parsing model installed. The person's footprint alpha gives the silhouette; the
head is excluded using the person's linked FACE object (metadata parent/children,
falling back to a head-fraction), and for tall full-body figures a lower strip is
excluded as legs. What remains is the torso/garment region — the only thing SDXL
is allowed to change.

Phase-2 refactor: the clothing sub-mask is this module's only special input; the
context crop / scaling / SDXL run / feathered patch are the shared engine. The
`_find_face` helper stays here (Change Hair imports it).

Config (env vars):
  COMFY_CLOTHES_CKPT     — SDXL checkpoint (falls back to the engine default)
  COMFY_CLOTHES_TIMEOUT  — seconds to wait for a render (default 180)
"""

import os

from comfy_engine import (
    _load_metadata, _background_rgb, _footprint_mask, _bbox_of,
    EditError, run_object_edit, LAYERS_DIR, CKPT_NAME as ENGINE_CKPT,
)

CKPT_NAME = os.environ.get("COMFY_CLOTHES_CKPT", ENGINE_CKPT)
RENDER_TIMEOUT = float(os.environ.get("COMFY_CLOTHES_TIMEOUT", "180"))

CONTEXT_MARGIN_FRAC = 0.3
HEAD_FRAC = 0.34            # fallback head height (when no face object is linked)
NECK_PAD_FRAC = 0.04       # drop the mask a touch below the face -> keep the neck
LEG_ASPECT = 1.7           # height/width above this => full body -> exclude legs
LEG_FRAC = 0.28            # bottom fraction excluded as legs for full-body figures
MASK_DILATE = 5
MASK_FEATHER = 9
# Clothing is GENERATED from the prompt, so denoise is high (unlike recolor).
DEFAULTS = {"steps": 30, "cfg": 7.0, "denoise": 0.9,
            "sampler": "dpmpp_2m", "scheduler": "karras", "seed": 0}

DEFAULT_NEGATIVE = ("different person, changed face, altered hair, deformed, extra limbs, "
                    "extra fingers, blurry, lowres, distorted, artifacts, seam, watermark, text")


class ClothesError(Exception):
    """A change-clothes request that cannot be fulfilled."""


def _find_face(meta, person):
    """Linked/overlapping FACE object for a person, or None."""
    px, py, pw, ph = person["x"], person["y"], person["width"], person["height"]
    for cid in person.get("children") or []:
        c = meta.get(int(cid))
        if c and str(c.get("category")) == "face":
            return c
    best = None
    for o in meta.values():
        if str(o.get("category")) != "face":
            continue
        cx, cy = o["x"] + o["width"] / 2.0, o["y"] + o["height"] / 2.0
        if px <= cx <= px + pw and py <= cy <= py + ph:
            if best is None or o["height"] > best["height"]:
                best = o
    return best


def _clothing_mask(meta, person, foot, bbox):
    """Footprint minus head (via face) minus legs (tall figures) = garment region."""
    ox0, oy0, ox1, oy1 = bbox
    bw, bh = ox1 - ox0, oy1 - oy0
    face = _find_face(meta, person)
    if face is not None:
        head_line = int(round(face["y"] + face["height"] + NECK_PAD_FRAC * bh))
    else:
        head_line = oy0 + int(round(HEAD_FRAC * bh))
    head_line = max(oy0, min(head_line, oy1 - 1))

    feet_line = oy1
    if bh > LEG_ASPECT * bw:  # full-body figure -> keep the legs
        feet_line = oy1 - int(round(LEG_FRAC * bh))

    cm = foot.copy()
    cm[:head_line, :] = 0
    cm[feet_line:, :] = 0
    return cm, (face is not None)


def change_clothes(object_id, prompt, options=None, client=None):
    """Regenerate `object_id`'s clothing from `prompt`. Face/hair/pose preserved.

    Returns { objectId, x, y, w, h, png, usedFace } — an RGBA patch for the
    clothing region only. Everything outside that region is never touched.
    """
    options = options or {}
    if not prompt or not str(prompt).strip():
        raise ClothesError("a clothing prompt is required")
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
        raise ClothesError(f"object {object_id} not found")
    if not os.path.exists(os.path.join(LAYERS_DIR, obj.get("file", ""))):
        raise ClothesError(f"layer file missing: {obj.get('file')}")

    bg = _background_rgb()
    H, W = bg.shape[:2]
    try:
        foot, bbox = _footprint_mask(obj, H, W)
    except EditError as exc:
        raise ClothesError(str(exc))

    cm, had_face = _clothing_mask(meta, obj, foot, bbox)
    if _bbox_of(cm > 0) is None:
        raise ClothesError("could not isolate a clothing region for this object")

    positive = (str(prompt).strip() +
                ", worn by the person, matching the lighting, perspective and body shape, "
                "natural fabric folds and wrinkles, consistent with the scene")
    negative = options.get("negative") or DEFAULT_NEGATIVE
    try:
        return run_object_edit(
            object_id, bg, cm, params=params, positive=positive, negative=negative,
            slug="clothes", ckpt=params.get("ckpt", CKPT_NAME),
            context_margin_frac=CONTEXT_MARGIN_FRAC, dilate=MASK_DILATE, feather=MASK_FEATHER,
            timeout=RENDER_TIMEOUT,
            controlnet={"type": "canny", "strength": 0.45, "end": 0.55},
            style=options.get("style") or "auto", material_hint="fabric",
            harmonize=bool(options.get("harmonize")), harmonize_strength=options.get("harmonizeStrength"),
            n=options.get("n") or 1, evaluator=bool(options.get("evaluator")),
            client=client, extra={"usedFace": had_face},
        )
    except EditError as exc:
        raise ClothesError(str(exc))
