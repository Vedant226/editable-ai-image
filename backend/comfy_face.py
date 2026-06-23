"""
Face edits — Beard / Smile / Age / Glasses / Skin (Phase 4).

These were the "SOON" buttons in the face panel. They are all the SAME operation
— regenerate a sub-region of a face from a prompt — so they are config entries on
the shared engine (comfy_engine.run_object_edit), differing only in:

  * which sub-region of the face is editable (mask),
  * the prompt + denoise,
  * the ControlNet kind (canny, to lock the face structure / identity).

No new pipeline, no human-parsing model: the sub-regions are derived from the
face object's footprint + bbox (mouth band, eye band, lower face, whole face).
Everything outside the region is preserved exactly; the result is an RGBA patch.

The selected object may be a FACE (category face/head) or a PERSON (we resolve
its linked face). Identity is protected by the canny ControlNet (when installed)
plus low denoise; the editor's "Preserve Identity" toggle still applies on top.

Config (env vars):
  COMFY_FACE_CKPT     — SDXL checkpoint (falls back to the engine default)
  COMFY_FACE_TIMEOUT  — seconds to wait for a render (default 180)
"""

import os

import cv2
import numpy as np

from comfy_engine import (
    _load_metadata, _background_rgb, _footprint_mask, _bbox_of,
    EditError, run_object_edit, LAYERS_DIR, CKPT_NAME as ENGINE_CKPT,
)
from comfy_clothes import _find_face

CKPT_NAME = os.environ.get("COMFY_FACE_CKPT", ENGINE_CKPT)
RENDER_TIMEOUT = float(os.environ.get("COMFY_FACE_TIMEOUT", "180"))

CONTEXT_MARGIN_FRAC = 0.45   # generous face context so edits blend with the head
MASK_DILATE = 5
MASK_FEATHER = 9
_FACE_CATS = ("face", "head", "portrait")

# region: which slice of the face footprint each feature may change.
# denoise: low = subtle/identity-safe (smile/skin/age); higher = adds content (beard/glasses).
FACE_FEATURES = {
    "smile": {
        "region": "mouth", "denoise": 0.42, "cn": 0.45,
        "prompt": "a warm natural smile, smiling mouth, happy expression, white teeth",
        "neg": "frown, sad, open screaming mouth, deformed mouth, distorted teeth",
    },
    "beard": {
        "region": "lower", "denoise": 0.55, "cn": 0.45,
        "prompt": "a natural beard and facial hair, well groomed, realistic hair strands",
        "neg": "clean shaven, deformed jaw, distorted mouth",
    },
    "glasses": {
        "region": "eyes", "denoise": 0.55, "cn": 0.5,
        "prompt": "wearing eyeglasses, glasses with clear lenses resting on the nose, "
                  "natural reflections",
        "neg": "deformed eyes, distorted frames, sunglasses, extra glasses",
    },
    "age": {
        "region": "face", "denoise": 0.4, "cn": 0.55,
        "prompt": "an older face with natural age, fine wrinkles, aged skin texture",
        "neg": "child, distorted features, different person, deformed",
        "younger": "a youthful face, smooth young skin, fewer wrinkles, rejuvenated",
    },
    "skin": {
        "region": "face", "denoise": 0.3, "cn": 0.6,
        "prompt": "smooth even healthy skin, clear complexion, natural skin texture, "
                  "gently retouched",
        "neg": "blemishes, plastic skin, waxy, blurry, distorted features",
    },
}


# material the edited region is made of (drives the planner's texture preservation)
_FEATURE_MATERIAL = {"smile": "skin", "beard": "hair", "glasses": "glass", "age": "skin", "skin": "skin"}


class FaceError(Exception):
    """A face edit that cannot be fulfilled (bad object / no face / bad feature)."""


def _face_kps(bg, bbox):
    """insightface 5-point landmarks (eyes/nose/mouth corners) in full-canvas
    coordinates, or None. Upscales small faces so the detector can find them;
    any failure (no insightface, no face, tiny face) returns None → caller uses
    the geometric fallback, so behavior never regresses."""
    try:
        import identity_manager
        app = identity_manager._get_app()
        if app is None:
            return None
        H, W = bg.shape[:2]
        ox0, oy0, ox1, oy1 = bbox
        pad = int(round(0.35 * max(1, oy1 - oy0)))
        cx0, cy0 = max(0, ox0 - pad), max(0, oy0 - pad)
        cx1, cy1 = min(W, ox1 + pad), min(H, oy1 + pad)
        crop = bg[cy0:cy1, cx0:cx1]
        if crop.size == 0:
            return None
        s = max(1.0, 256.0 / max(1, oy1 - oy0))  # upscale so the detector sees it
        up = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
        faces = app.get(up[:, :, ::-1])  # insightface expects BGR
        if not faces:
            return None
        f = max(faces, key=lambda z: float(getattr(z, "det_score", 0.0)))
        if getattr(f, "kps", None) is None:
            return None
        kps = np.asarray(f.kps, dtype=np.float32) / s
        kps[:, 0] += cx0
        kps[:, 1] += cy0
        return kps
    except Exception:  # noqa: BLE001
        return None


def _region_mask(region, foot, bbox, kps=None):
    """Sub-region of the face footprint the feature may change.

    With insightface landmarks (`kps`) the mouth/eye/jaw regions are placed on
    the ACTUAL face (robust to pose/offset); without them, fixed bbox fractions.
    """
    ox0, oy0, ox1, oy1 = bbox
    fw, fh = ox1 - ox0, oy1 - oy0
    if region == "face":
        return foot.copy()
    m = np.zeros_like(foot)

    if kps is not None and region in ("mouth", "eyes", "lower"):
        le, re, nose, lm, rm = (kps[i] for i in range(5))
        if region == "mouth":        # smile: ellipse over the real mouth
            cx, cy = (lm[0] + rm[0]) / 2.0, (lm[1] + rm[1]) / 2.0
            w = max(8.0, float(np.hypot(lm[0] - rm[0], lm[1] - rm[1])) * 1.8)
            cv2.ellipse(m, (int(round(cx)), int(round(cy))), (int(w / 2), int(w * 0.4)), 0, 0, 360, 255, -1)
        elif region == "eyes":       # glasses: ellipse over the real eye line
            cx, cy = (le[0] + re[0]) / 2.0, (le[1] + re[1]) / 2.0
            w = max(10.0, float(np.hypot(le[0] - re[0], le[1] - re[1])) * 1.9)
            cv2.ellipse(m, (int(round(cx)), int(round(cy))), (int(w / 2), int(w * 0.28)), 0, 0, 360, 255, -1)
        elif region == "lower":      # beard: from just below the nose to the chin
            y0 = max(oy0, int(round(nose[1] + 0.10 * fh)))
            m[y0:oy1, :] = 255
        return cv2.bitwise_and(m, foot)

    # geometric fallback (no landmarks)
    if region == "lower":
        y0 = oy0 + int(round(0.50 * fh))
        m[y0:oy1, :] = foot[y0:oy1, :]
    elif region == "mouth":
        y0, y1 = oy0 + int(round(0.60 * fh)), oy0 + int(round(0.92 * fh))
        x0, x1 = ox0 + int(round(0.18 * fw)), ox0 + int(round(0.82 * fw))
        m[y0:y1, x0:x1] = foot[y0:y1, x0:x1]
    elif region == "eyes":
        y0, y1 = oy0 + int(round(0.24 * fh)), oy0 + int(round(0.55 * fh))
        m[y0:y1, :] = foot[y0:y1, :]
    else:
        raise FaceError(f"unknown face region: {region}")
    return m


def face_edit(object_id, feature, options=None, client=None):
    """Apply a face feature (beard/smile/age/glasses/skin) to `object_id`.

    Returns { objectId, feature, x, y, w, h, png } — an RGBA patch for the edited
    sub-region only. Everything outside it is preserved exactly.
    """
    options = options or {}
    feature = str(feature or "").lower().strip()
    cfg = FACE_FEATURES.get(feature)
    if cfg is None:
        raise FaceError(f"unknown face feature: {feature!r} "
                        f"(expected one of {sorted(FACE_FEATURES)})")

    params = {"steps": 28, "cfg": 7.0, "denoise": cfg["denoise"],
              "sampler": "dpmpp_2m", "scheduler": "karras", "seed": 0}
    for k in ("steps", "cfg", "denoise", "sampler", "scheduler", "seed", "ckpt"):
        if options.get(k) is not None:
            params[k] = options[k]
    # Intensity (0..1) is an edit-strength control: scales denoise around the
    # feature's default (an explicit `denoise` override still wins). Lets the UI
    # offer Subtle…Strong without per-feature magic numbers.
    if options.get("intensity") is not None and options.get("denoise") is None:
        try:
            iv = max(0.0, min(1.0, float(options["intensity"])))
            params["denoise"] = round(max(0.15, min(0.95, cfg["denoise"] * (0.6 + 0.8 * iv))), 3)
        except (TypeError, ValueError):
            pass

    meta = _load_metadata()
    obj = meta.get(int(object_id))
    if obj is None:
        raise FaceError(f"object {object_id} not found")

    # resolve a FACE: the object itself, or the selected person's linked face
    if str(obj.get("category")) in _FACE_CATS:
        face = obj
    else:
        face = _find_face(meta, obj)
    if face is None:
        raise FaceError("no face found for this object — select a face or a person")
    if not os.path.exists(os.path.join(LAYERS_DIR, face.get("file", ""))):
        raise FaceError(f"layer file missing: {face.get('file')}")
    face_id = int(face["id"])

    bg = _background_rgb()
    H, W = bg.shape[:2]
    try:
        foot, bbox = _footprint_mask(face, H, W)
    except EditError as exc:
        raise FaceError(str(exc))

    kps = _face_kps(bg, bbox)  # landmark-aware masking when a face is detected
    region_mask = _region_mask(cfg["region"], foot, bbox, kps=kps)
    if _bbox_of(region_mask > 0) is None:
        raise FaceError(f"could not isolate the '{feature}' region for this face")

    # prompt: explicit override > age-direction variant > feature default
    direction = str(options.get("direction") or "").lower()
    positive = options.get("prompt")
    if not positive:
        positive = cfg["younger"] if (feature == "age" and direction == "younger") else cfg["prompt"]
    positive = positive + ", on the person's face, matching the lighting and head shape"
    negative = options.get("negative") or (cfg["neg"] + ", blurry, lowres, distorted, artifacts, seam, watermark, text")

    try:
        return run_object_edit(
            face_id, bg, region_mask, params=params, positive=positive, negative=negative,
            slug=f"face_{feature}", ckpt=params.get("ckpt", CKPT_NAME),
            context_margin_frac=CONTEXT_MARGIN_FRAC, dilate=MASK_DILATE, feather=MASK_FEATHER,
            timeout=RENDER_TIMEOUT,
            controlnet={"type": "canny", "strength": cfg["cn"], "end": 0.7},
            style=options.get("style") or "auto", material_hint=_FEATURE_MATERIAL.get(feature),
            client=client,
            harmonize=bool(options.get("harmonize")), harmonize_strength=options.get("harmonizeStrength"),
            n=options.get("n") or 1, evaluator=bool(options.get("evaluator")),
            extra={"feature": feature, "faceId": face_id, "landmarks": kps is not None,
                   "denoise": round(float(params["denoise"]), 3)},
        )
    except EditError as exc:
        raise FaceError(str(exc))
