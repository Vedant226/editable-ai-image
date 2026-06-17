"""
Matte refinement + foreground color decontamination.

The SAM-derived object mask is a hard, imperfect cut. To make a lifted object
read as a continuous part of the original (no fringe, no halo) we:

  1. refine the binary mask into a SOFT, edge-aware alpha matte (guided filter,
     guided by the original luminance), and
  2. DECONTAMINATE the foreground colour in the boundary band, removing the
     original background's bleed — using the inpaint result as the background
     estimate B in the matting equation  I = αF + (1-α)B  ->  F = (I-(1-α)B)/α.

Dependency-light: only numpy + core cv2 (box/Gaussian filters).
"""

import cv2
import numpy as np


def guided_filter(guide, src, radius, eps):
    """Edge-aware smoothing of `src` guided by `guide` (both float32 0..1)."""
    r = max(1, int(radius))
    ksize = (r, r)
    mean_g = cv2.boxFilter(guide, -1, ksize)
    mean_s = cv2.boxFilter(src, -1, ksize)
    corr_g = cv2.boxFilter(guide * guide, -1, ksize)
    corr_gs = cv2.boxFilter(guide * src, -1, ksize)
    var_g = corr_g - mean_g * mean_g
    cov_gs = corr_gs - mean_g * mean_s
    a = cov_gs / (var_g + eps)
    b = mean_s - a * mean_g
    mean_a = cv2.boxFilter(a, -1, ksize)
    mean_b = cv2.boxFilter(b, -1, ksize)
    return mean_a * guide + mean_b


def refine_matte(crop_rgb, binary, radius=8, eps=1e-3):
    """
    crop_rgb : HxWx3 uint8 (original pixels)
    binary   : HxW float/uint8, 1 inside the object
    returns  : HxW float32 alpha in [0,1], soft at the silhouette
    """
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    p = (binary > 0).astype(np.float32)
    alpha = guided_filter(gray, p, radius, eps)
    # gentle remap: keep confident interior/exterior crisp, preserve a soft band
    alpha = np.clip((alpha - 0.15) / 0.70, 0.0, 1.0)
    return alpha


def decontaminate(crop_rgb, alpha, bg_rgb):
    """
    Estimate true foreground colour, removing background bleed in the edge band.
    crop_rgb, bg_rgb : HxWx3 uint8 ; alpha : HxW float 0..1
    returns RGBA uint8 (straight alpha).
    """
    I = crop_rgb.astype(np.float32)
    B = bg_rgb.astype(np.float32)
    a = alpha[:, :, None]
    F = (I - (1.0 - a) * B) / np.clip(a, 0.15, 1.0)
    F = np.where(a > 0.95, I, F)  # untouched interior
    F = np.clip(F, 0, 255)
    rgba = np.dstack([F, np.clip(alpha * 255.0, 0, 255)]).astype(np.uint8)
    return rgba
