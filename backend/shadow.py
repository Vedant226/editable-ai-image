"""
Soft shadow synthesis.

The product rule is that a lifted object carries a believable, grounded shadow
and leaves none behind. We:

  - SYNTHESIZE an attached soft shadow from the object's alpha (blur + offset +
    darken), returned as a black RGBA matte the frontend animates (grows on
    lift, grounds on settle), and
  - expose the shadow REGION so the caller can fold it into the inpaint mask,
    which removes the object's *original* cast shadow from the base.

v1 uses sensible default direction/softness; observed-shadow estimation is a
later upgrade.
"""

import cv2
import numpy as np


def _odd(n):
    n = max(1, int(round(n)))
    return n if n % 2 == 1 else n + 1


def synthesize_shadow(object_alpha, *, blur, dx, dy):
    """
    object_alpha : HxW float 0..1 (full canvas)
    returns      : HxW float 0..1 — the soft, offset shadow alpha (pre-opacity)
    """
    k = _odd(blur)
    sh = cv2.GaussianBlur(object_alpha.astype(np.float32), (k, k), 0)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    sh = cv2.warpAffine(sh, M, (object_alpha.shape[1], object_alpha.shape[0]))
    return np.clip(sh, 0.0, 1.0)


def shadow_params(w, h):
    """Default direction/softness/opacity scaled to object size."""
    size = max(w, h)
    return {
        "blur": max(5, round(size * 0.18)),
        "dx": round(size * 0.05),
        "dy": round(size * 0.07),
        "opacity": 0.40,
    }
