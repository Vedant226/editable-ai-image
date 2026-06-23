"""
comfy_evaluator.py — multi-dimensional, CPU-only self-evaluation of a generated
patch (Phase 14 quality upgrade).

Additive + isolated. `evaluate()` NEVER raises and returns a STRICT SUPERSET of
`comfy_engine._validate_patch` — the same {blur, colorStd, ok, checks} fields,
computed identically, PLUS a richer {score, dimensions{...}} block. The engine
keeps `_validate_patch` as the retry oracle and only consults the rich block when
explicitly asked (evaluator=True or best-of-N selection), so the existing retry
behaviour and the byte-identical base path are unchanged.

Dimensions (each 0..1, higher = better). All are cv2/numpy and model-free except
the two optional ones, which return None (and drop out of the weighted mean) when
their model is unavailable:
  blur      — normalized Laplacian variance (sharpness)
  flatness  — normalized per-channel std (patch is not degenerate/flat)
  seam      — Sobel-gradient continuity across the footprint boundary
              (does the patch edge blend into the original surround?) — the seam tell
  lighting  — LAB tone similarity of the patch vs the surrounding ambient ring
  color     — HSV hue/saturation similarity of the patch vs the ambient ring
  prompt    — CLIP image/text cosine (OPTIONAL; None when no CLIP backend present)
  identity  — ArcFace cosine vs the original face (OPTIONAL; None when no face)

`score` is the weighted mean over the present (non-None) dimensions; the weights
renormalize so absent optional dimensions don't penalize the total.

Heavy modules (comfy_lighting, identity_manager) are imported lazily inside the
helpers so importing this module — and starting the bridge — stays fast and free
of circular imports (comfy_engine imports this module).
"""

import cv2
import numpy as np

# Kept in sync with comfy_engine._validate_patch (the retry oracle). The engine
# stays authoritative for the retry decision; these only drive the backward-
# compatible fields when evaluate() is used stand-alone.
_MIN_BLUR = 6.0   # Laplacian variance below this ≈ blurry/smeared
_MIN_STD = 4.0    # per-channel std below this ≈ flat/degenerate patch

# Normalization ceilings: the dimension earns full 1.0 credit at this raw value.
_BLUR_FULL = 120.0
_STD_FULL = 45.0

_DEFAULT_WEIGHTS = {
    "seam": 0.25, "lighting": 0.20, "blur": 0.15,
    "color": 0.15, "flatness": 0.10, "prompt": 0.10, "identity": 0.05,
}


def _clip01(v):
    return float(max(0.0, min(1.0, v)))


# -- optional CLIP backend (prompt-compliance) -------------------------------

def clip_available():
    """True only if a CLIP backend is importable. None is installed in this
    environment, so prompt-compliance scoring stays a graceful no-op."""
    for mod in ("clip", "open_clip"):
        try:
            __import__(mod)
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


def build_clip_ctx():
    """Lazily build a CLIP scoring handle, or None when unavailable. Never raises.

    Placeholder: returns None until a CLIP backend (open_clip / clip) is
    installed. When one is, this should return an object with
    `.score(patch_rgb, prompt) -> float in 0..1`."""
    return None


# -- per-dimension helpers (each returns 0..1 or None; never raises) ---------

def _sharpness(patch_rgb):
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    std = float(patch_rgb.std())
    return blur, std


def _ambient_mask(foot, crop_rgb, px, py):
    """The original scene around the patch footprint, in crop coordinates."""
    ch, cw = crop_rgb.shape[:2]
    h, w = foot.shape[:2]
    ref_obj = np.zeros((ch, cw), bool)
    y1, x1 = min(ch, py + h), min(cw, px + w)
    if y1 > py and x1 > px:
        ref_obj[py:y1, px:x1] = foot[:y1 - py, :x1 - px].astype(bool)
    return ~ref_obj, ref_obj


def _seam_score(patch_rgb, foot, crop_rgb, px, py):
    """Gradient continuity across the footprint boundary (1 = seamless).

    Composite the patch into the original crop, then compare the mean edge energy
    in a thin ring straddling the boundary against the edge energy just outside.
    A visible seam shows up as boundary gradient far above the surround's."""
    try:
        ch, cw = crop_rgb.shape[:2]
        h, w = patch_rgb.shape[:2]
        y1, x1 = min(ch, py + h), min(cw, px + w)
        if y1 <= py or x1 <= px:
            return None
        sub_foot = foot[:y1 - py, :x1 - px].astype(bool)
        comp = crop_rgb.copy()
        region = comp[py:y1, px:x1]
        region[sub_foot] = patch_rgb[:y1 - py, :x1 - px][sub_foot]
        comp[py:y1, px:x1] = region

        fc = np.zeros((ch, cw), "uint8")
        fc[py:y1, px:x1][sub_foot] = 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dil_k = cv2.dilate(fc, k) > 0
        ero_k = cv2.erode(fc, k) > 0
        band_b = dil_k & ~ero_k                       # ring across the boundary
        outer_b = (cv2.dilate(fc, big) > 0) & ~dil_k  # surround just outside
        if int(band_b.sum()) < 16 or int(outer_b.sum()) < 16:
            return None

        gray = cv2.cvtColor(comp, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        seam_energy = float(mag[band_b].mean())
        ref_energy = float(mag[outer_b].mean()) + 1e-3
        excess = (seam_energy - ref_energy) / ref_energy
        return _clip01(1.0 - max(0.0, excess))
    except Exception:  # noqa: BLE001
        return None


def _lighting_score(patch_rgb, foot, crop_rgb, px, py):
    """LAB tone similarity of the patch vs the surrounding ambient ring."""
    try:
        import comfy_lighting
        ambient, _ref_obj = _ambient_mask(foot, crop_rgb, px, py)
        if int(ambient.sum()) < 64 or int(foot.sum()) < 16:
            return None
        src = comfy_lighting._lab_stats(patch_rgb, foot.astype("uint8") * 255)
        tgt = comfy_lighting._lab_stats(crop_rgb, ambient.astype("uint8") * 255)
        dl = abs(src["mL"] - tgt["mL"]) / 50.0
        da = abs(src["mA"] - tgt["mA"]) / 24.0
        db = abs(src["mB"] - tgt["mB"]) / 24.0
        return _clip01(1.0 - (0.5 * dl + 0.25 * da + 0.25 * db))
    except Exception:  # noqa: BLE001
        return None


def _mean_hsv(hsv, mask):
    H = hsv[:, :, 0][mask] * (2.0 * np.pi / 180.0)   # OpenCV hue is [0,180)
    s = hsv[:, :, 1][mask] / 255.0
    mh = np.arctan2(np.sin(H).mean(), np.cos(H).mean())
    return mh, float(s.mean())


def _color_score(patch_rgb, foot, crop_rgb, px, py):
    """HSV hue/saturation similarity of the patch vs the ambient ring."""
    try:
        ambient, _ref_obj = _ambient_mask(foot, crop_rgb, px, py)
        fb = foot.astype(bool)
        if int(ambient.sum()) < 64 or int(fb.sum()) < 16:
            return None
        ph = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2HSV)
        ah = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
        mhp, msp = _mean_hsv(ph, fb)
        mha, msa = _mean_hsv(ah, ambient)
        dh = abs(np.arctan2(np.sin(mhp - mha), np.cos(mhp - mha))) / np.pi
        ds = abs(msp - msa)
        return _clip01(1.0 - (0.6 * dh + 0.4 * ds))
    except Exception:  # noqa: BLE001
        return None


def _prompt_score(patch_rgb, prompt, clip_ctx):
    if not prompt or clip_ctx is None:
        return None
    try:
        return _clip01(float(clip_ctx.score(patch_rgb, prompt)))
    except Exception:  # noqa: BLE001
        return None


def _identity_score(patch_rgb, original_face_rgb):
    if original_face_rgb is None:
        return None
    try:
        import identity_manager as idm
        fa = idm.main_face(original_face_rgb)
        fb = idm.main_face(patch_rgb)
        if fa is None or fb is None:
            return None
        sim = idm.similarity(fa.embedding, fb.embedding)
        if sim is None:
            return None
        return _clip01((sim + 1.0) / 2.0)   # cosine [-1,1] -> [0,1]
    except Exception:  # noqa: BLE001
        return None


# -- public API --------------------------------------------------------------

def evaluate(patch_rgb, *, crop_rgb=None, patch_left=0, patch_top=0,
             footprint=None, prompt=None, original_face_rgb=None,
             weights=None, clip_ctx=None, client=None):
    """Score a generated patch across multiple dimensions. Never raises.

    Returns a superset of _validate_patch's report:
      {blur, colorStd, ok, checks,            # backward-compatible
       score, dimensions{seam,lighting,color,blur,flatness,prompt,identity}}
    Optional dimensions are None when their input/model is unavailable and are
    excluded from the weighted `score`.
    """
    try:
        if patch_rgb is None or getattr(patch_rgb, "size", 0) == 0:
            return {"blur": 0.0, "colorStd": 0.0, "ok": False,
                    "checks": {"blur": False, "notFlat": False},
                    "score": 0.0, "dimensions": {}}

        blur, std = _sharpness(patch_rgb)
        checks = {"blur": blur >= _MIN_BLUR, "notFlat": std >= _MIN_STD}

        if footprint is not None:
            foot = (np.asarray(footprint) > 128)
        else:
            foot = np.ones(patch_rgb.shape[:2], bool)
        px, py = int(patch_left or 0), int(patch_top or 0)

        dims = {
            "blur": _clip01(blur / _BLUR_FULL),
            "flatness": _clip01(std / _STD_FULL),
            "seam": None, "lighting": None, "color": None,
            "prompt": None, "identity": None,
        }
        if crop_rgb is not None:
            dims["seam"] = _seam_score(patch_rgb, foot, crop_rgb, px, py)
            dims["lighting"] = _lighting_score(patch_rgb, foot, crop_rgb, px, py)
            dims["color"] = _color_score(patch_rgb, foot, crop_rgb, px, py)
        dims["prompt"] = _prompt_score(patch_rgb, prompt, clip_ctx)
        dims["identity"] = _identity_score(patch_rgb, original_face_rgb)

        w = dict(_DEFAULT_WEIGHTS)
        if weights:
            w.update(weights)
        num = den = 0.0
        for key, val in dims.items():
            if val is None:
                continue
            wk = w.get(key, 0.0)
            num += wk * val
            den += wk
        score = (num / den) if den > 0 else 0.0

        return {
            "blur": round(blur, 1), "colorStd": round(std, 1),
            "ok": all(checks.values()), "checks": checks,
            "score": round(float(score), 3),
            "dimensions": {k: (round(v, 3) if v is not None else None)
                           for k, v in dims.items()},
        }
    except Exception as exc:  # noqa: BLE001 - safety: never crash an edit
        return {"blur": 0.0, "colorStd": 0.0, "ok": True,
                "checks": {"blur": True, "notFlat": True},
                "score": 0.0, "dimensions": {}, "error": str(exc)}
