"""
Seamless background reconstruction (colour / lighting harmonisation).

LaMa (or the OpenCV fallback) fills the hole an object leaves, but its interior
can carry a faint global colour cast or lighting offset relative to the
surrounding artwork — a tell that reads as "edited". This step removes that
mismatch so the repaired region sits in the original's colour/lighting:

  - measure the per-channel mean of a RING of surrounding ORIGINAL pixels and
    the mean of the FILLED region, and pull the fill toward the original by the
    (clamped) difference;
  - apply that correction through a FEATHERED weight that is full strength deep
    inside the region and ramps to ZERO at the boundary — so the fill stays
    continuous with the original at the seam (the correction can never create a
    step), while the interior colour/lighting is matched.

Why not Poisson `seamlessClone`: measured on real content it preserves the
fill's interior gradients and re-introduces large offsets (it is built for
compositing a *different* image, not harmonising an in-place inpaint), so it
made the mismatch worse. A feathered low-frequency correction is both safer
(clamped, no boundary step) and measurably better.

Content-agnostic (no per-image / per-category logic) and fail-safe: any
degenerate input returns the unmodified fill, never raises.
"""

import cv2
import numpy as np


def _ring_outside(region, width):
    """Boolean ring of `width` px just OUTSIDE `region` (sampled from original)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width * 2 + 1, width * 2 + 1))
    dil = cv2.dilate(region.astype(np.uint8), k) > 0
    return dil & ~region


def harmonize_fill(bg, filled, mask, ring_px=14, feather=9, clip=36.0, strength=1.0):
    """
    bg     : HxWx3 uint8 — untouched original.
    filled : HxWx3 uint8 — bg with the masked region inpainted (== bg elsewhere).
    mask   : HxW uint8   — the reconstructed region (>0).
    Returns HxWx3 uint8 — `filled` with its interior colour/lighting matched to
    the surrounding original, feathered to zero at the boundary (no seam).
    """
    region = mask > 0
    if not region.any():
        return filled
    ring = _ring_outside(region, ring_px)
    if ring.sum() < 24 or region.sum() < 24:
        return filled

    # weight: 1 deep inside the region, ramping to 0 at the boundary, so the
    # correction never introduces a step where the fill meets the original.
    fk = max(3, feather | 1)
    inside = cv2.erode(region.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fk, fk)))
    w = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), feather)

    out = filled.astype(np.float32)
    bgf = bg.astype(np.float32)
    for c in range(3):
        shift = float(bgf[ring, c].mean() - out[region, c].mean())
        shift = max(-clip, min(clip, shift))  # clamp so a wild fill can't be tinted absurdly
        out[:, :, c] += strength * shift * w
    return np.clip(out, 0, 255).astype(np.uint8)


def seamless_reconstruct(bg, filled, mask):
    """Public entry: make the inpaint fill seamless with the original.

    Fail-safe by contract: ANY error (cv2, shape mismatch, degenerate mask)
    returns the unmodified fill rather than breaking /lift."""
    try:
        return harmonize_fill(bg, filled, mask)
    except Exception:
        return filled
