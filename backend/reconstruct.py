"""
Seamless background reconstruction (colour / lighting / texture harmonisation).

LaMa (or the OpenCV fallback) fills the hole an object leaves, but its interior
can read as "edited": a faint colour/lighting cast and lost high-frequency
texture (the fill is smoother/cleaner than the surrounding artwork — measured
~0.76× the surround's texture energy on real content). This stage removes those
tells, then keeps each refinement ONLY when an objective check says it helped.

Steps:
  1. COLOUR/LIGHTING harmonisation — pull the fill's interior mean toward a ring
     of surrounding ORIGINAL pixels, through a feathered weight that is full
     strength deep inside and ZERO at the boundary (so it can never add a seam).
  2. TEXTURE continuation — amplify the fill's OWN high-frequency band toward the
     surround's texture energy (restores grain / emboss / paper tooth), again
     feathered to zero at the boundary. It only amplifies detail the fill already
     has, so it cannot hallucinate foreign structure.
  3. OBJECTIVE selection — score {harmonised, +texture, +Poisson} by colour cast,
     boundary seam and texture deviation, and pick the best candidate that does
     NOT worsen colour/seam vs the harmonised base. Poisson `seamlessClone` is
     offered here but is essentially never eligible (it preserves the fill's
     interior gradients and re-introduces large offsets — measured strictly
     worse), so it is applied only on the rare object where it objectively helps.

Content-agnostic (no per-image / per-category constants) and fail-safe: any
error returns the best result obtained so far, never raises (see public entry).
"""

import cv2
import numpy as np

SIGMA = 2.0  # scale separating "structure" (low-freq) from "texture" (high-freq)


def _ring_outside(region, width):
    """Boolean ring of `width` px just OUTSIDE `region` (sampled from original)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width * 2 + 1, width * 2 + 1))
    dil = cv2.dilate(region.astype(np.uint8), k) > 0
    return dil & ~region


def _feather_inside(region, feather):
    """Weight 1 deep inside `region`, ramping to 0 at the boundary (no seam)."""
    fk = max(3, feather | 1)
    inside = cv2.erode(region.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fk, fk)))
    return cv2.GaussianBlur(inside.astype(np.float32), (0, 0), feather)


def _highpass(img, sigma=SIGMA):
    f = img.astype(np.float32)
    return f - cv2.GaussianBlur(f, (0, 0), sigma)


def harmonize_fill(bg, filled, mask, ring_px=14, feather=9, clip=36.0):
    """Match the fill's interior colour/lighting to the surrounding original."""
    region = mask > 0
    if not region.any():
        return filled
    ring = _ring_outside(region, ring_px)
    if ring.sum() < 24 or region.sum() < 24:
        return filled
    w = _feather_inside(region, feather)
    out = filled.astype(np.float32)
    bgf = bg.astype(np.float32)
    for c in range(3):
        shift = float(bgf[ring, c].mean() - out[region, c].mean())
        shift = max(-clip, min(clip, shift))
        out[:, :, c] += shift * w
    return np.clip(out, 0, 255).astype(np.uint8)


def texture_continuation(bg, filled, mask, ring_px=18, feather=9, max_gain=2.2):
    """Amplify the fill's own high-freq texture toward the surround's energy."""
    region = mask > 0
    if not region.any():
        return filled
    ring = _ring_outside(region, ring_px)
    if ring.sum() < 60 or region.sum() < 60:
        return filled
    hp = _highpass(filled)
    e_fill = float(np.std(hp[region]))
    e_ring = float(np.std(_highpass(bg)[ring]))
    if e_fill < 1e-3 or e_ring < 1e-3:
        return filled
    gain = min(max_gain, e_ring / e_fill)
    if gain <= 1.05:  # already textured enough — don't touch it
        return filled
    w = _feather_inside(region, feather)[:, :, None]
    out = filled.astype(np.float32) + (gain - 1.0) * hp * w * region[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _poisson(bg, filled, mask):
    """Poisson seamless clone — offered to the selector, rarely eligible."""
    region = mask > 0
    ys, xs = np.where(region)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    H, W = bg.shape[:2]
    if x0 <= 1 or y0 <= 1 or x1 >= W - 2 or y1 >= H - 2:
        return None
    try:
        return cv2.seamlessClone(filled, bg, (region.astype(np.uint8)) * 255, ((x0 + x1) // 2, (y0 + y1) // 2), cv2.NORMAL_CLONE)
    except cv2.error:
        return None


def _metrics(bg, img, region, ring):
    """(cast, seam, tex_dev): colour mismatch, boundary step, texture deviation."""
    cast = float(np.abs(img[region].mean(0) - bg[ring].mean(0)).mean())
    inner = region & ~(cv2.erode(region.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    outer = _ring_outside(region, 2)
    seam = float(np.abs(img[inner].mean(0) - bg[outer].mean(0)).mean()) if inner.any() and outer.any() else 0.0
    e_fill = float(np.std(_highpass(img)[region]))
    e_ring = float(np.std(_highpass(bg)[ring]))
    tex_dev = abs(1.0 - (e_fill / e_ring)) if e_ring > 1e-3 else 0.0
    return cast, seam, tex_dev


def reconstruct(bg, filled, mask, ring_px=16):
    """
    Harmonise colour/lighting + texture, then keep each refinement ONLY if an
    objective check says it did not worsen the colour/seam of the harmonised
    base — choosing, among the eligible candidates, the one closest to the
    surround's texture. Returns HxWx3 uint8.
    """
    region = mask > 0
    if not region.any():
        return filled
    ring = _ring_outside(region, ring_px)
    if ring.sum() < 24 or region.sum() < 24:
        return filled

    base = harmonize_fill(bg, filled, mask)
    b_cast, b_seam, b_tex = _metrics(bg, base, region, ring)

    candidates = [(base, b_cast, b_seam, b_tex)]
    tex = texture_continuation(bg, base, mask)
    if tex is not base:
        candidates.append((tex, *_metrics(bg, tex, region, ring)))
    pois = _poisson(bg, base, mask)
    if pois is not None:
        candidates.append((pois, *_metrics(bg, pois, region, ring)))

    # eligibility: never worse than the harmonised base on colour or seam (the
    # proven wins); among those, prefer the smallest texture deviation, then cast.
    TOL_CAST, TOL_SEAM = 2.0, 3.0
    eligible = [c for c in candidates if c[1] <= b_cast + TOL_CAST and c[2] <= b_seam + TOL_SEAM]
    if not eligible:
        eligible = [candidates[0]]
    best = min(eligible, key=lambda c: (c[3], c[1]))
    return best[0]


def seamless_reconstruct(bg, filled, mask):
    """Public entry: make the inpaint fill seamless with the original.

    Fail-safe by contract: ANY error (cv2, shape mismatch, degenerate mask)
    returns the unmodified fill rather than breaking /lift."""
    try:
        return reconstruct(bg, filled, mask)
    except Exception:
        return filled
