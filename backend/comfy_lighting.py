"""
Lighting & Shadow Harmonization Engine — Phase 10.

An OPTIONAL, fully isolated post-process that runs AFTER an AI edit. It compares
the AI-generated crop against the original scene and harmonises ONLY the
generated crop's lighting — brightness, exposure, white balance, scene colour
temperature, contrast, gamma and saturation — to match the surrounding scene,
then handles contact shadows and edge blending. Geometry, identity, texture,
shape, mask and alpha are preserved (the colour transform is a global affine in
LAB, so detail is never altered); only lighting changes.

It never re-renders SDXL — it is pure CPU/OpenCV and targets <2 s on a crop. It
reuses the shared data-url helpers and the lighting_manager's config + analysis.
On ANY failure it returns the original patch unchanged, so enabling lighting can
never crash or block editing.

Endpoint contract (POST /comfyui/lighting):
  patch      : RGBA data URL — the AI patch (footprint alpha), or a full image
  reference  : RGB data URL  — the original scene (margined crop around the patch)
  patchLeft  : x offset of the patch within the reference (px)
  patchTop   : y offset of the patch within the reference (px)
  strength   : optional global blend override (0..1)
"""

import cv2
import numpy as np

from comfy_replace import _decode_data_url, _data_url
import lighting_manager as lm


class LightingError(Exception):
    pass


# -- LAB statistics for the transform (float precision) ----------------------

def _lab_stats(rgb_u8, mask):
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    sel = mask.astype(bool)
    if int(sel.sum()) < 16:
        sel = np.ones(rgb_u8.shape[:2], bool)
    L = lab[:, :, 0][sel]
    A = lab[:, :, 1][sel] - 128.0
    B = lab[:, :, 2][sel] - 128.0
    C = np.sqrt(A * A + B * B)
    return {"mL": float(L.mean()), "sL": float(L.std() + 1e-3), "mA": float(A.mean()),
            "mB": float(B.mean()), "mC": float(C.mean() + 1e-3), "medL": float(np.median(L))}


def _apply_corrections(rgb_f, src, tgt, gstr):
    """Global LAB tone/colour match toward the target stats (texture preserved)."""
    lab = cv2.cvtColor(rgb_f.clip(0, 255).astype("uint8"), cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]
    A = lab[:, :, 1] - 128.0
    B = lab[:, :, 2] - 128.0

    ex = lm.stage("exposure", {"enabled": True, "strength": 0.6, "max_shift": 48.0})
    if ex.get("enabled", True):
        s = gstr * ex.get("strength", 0.6)
        L = L + float(np.clip(s * (tgt["mL"] - src["mL"]), -ex.get("max_shift", 48), ex.get("max_shift", 48)))

    co = lm.stage("contrast", {"enabled": True, "strength": 0.5, "min_scale": 0.7, "max_scale": 1.4})
    if co.get("enabled", True):
        s = gstr * co.get("strength", 0.5)
        scale = float(np.clip(1 + s * (tgt["sL"] / src["sL"] - 1), co.get("min_scale", 0.7), co.get("max_scale", 1.4)))
        L = (L - tgt["mL"]) * scale + tgt["mL"]

    ga = lm.stage("gamma", {"enabled": True, "strength": 0.4, "min": 0.6, "max": 1.7})
    if ga.get("enabled", True) and src["medL"] > 1 and tgt["medL"] > 1:
        s = gstr * ga.get("strength", 0.4)
        g = np.log(max(1e-3, tgt["medL"] / 255.0)) / np.log(max(1e-3, src["medL"] / 255.0))
        g = float(np.clip(g, ga.get("min", 0.6), ga.get("max", 1.7)))
        g = 1 + s * (g - 1)
        L = np.power(np.clip(L / 255.0, 0, 1), g) * 255.0

    wb = lm.stage("white_balance", {"enabled": True, "strength": 0.7, "max_shift": 24.0})
    if wb.get("enabled", True):
        s = gstr * wb.get("strength", 0.7); mx = wb.get("max_shift", 24.0)
        A = A + float(np.clip(s * (tgt["mA"] - src["mA"]), -mx, mx))
        B = B + float(np.clip(s * (tgt["mB"] - src["mB"]), -mx, mx))

    sa = lm.stage("saturation", {"enabled": True, "strength": 0.5, "min_scale": 0.6, "max_scale": 1.6})
    if sa.get("enabled", True):
        s = gstr * sa.get("strength", 0.5)
        sc = float(np.clip(1 + s * (tgt["mC"] / src["mC"] - 1), sa.get("min_scale", 0.6), sa.get("max_scale", 1.6)))
        A = A * sc; B = B * sc

    out = np.dstack([np.clip(L, 0, 255), np.clip(A + 128, 0, 255), np.clip(B + 128, 0, 255)]).astype("uint8")
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB).astype(np.float32)


def _contact_shadow(out_rgba, light_dir_deg, gstr):
    """Conservatively extend a soft contact shadow into empty ground below the
    object. Only paints currently-transparent pixels — never darkens content,
    never invents an unrealistic shadow (no-op when there is no ground below)."""
    sh = lm.stage("shadow", {"enabled": True, "strength": 0.35, "softness": 0.6,
                             "max_extend_frac": 0.12, "min_opacity": 0.04, "max_opacity": 0.5})
    h, w = out_rgba.shape[:2]
    alpha = out_rgba[:, :, 3].astype(np.float32)
    rgb = out_rgba[:, :, :3].astype(np.float32)
    foot = (alpha > 128).astype("uint8")
    info = {"applied": False, "direction_deg": round(float(light_dir_deg), 1),
            "softness": sh.get("softness", 0.6), "intensity": 0.0}
    ext = int(round(sh.get("max_extend_frac", 0.12) * h))
    if not sh.get("enabled", True) or int(foot.sum()) < 64 or ext < 2:
        return out_rgba, info

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, (ext // 2) * 2 + 1), ext | 1))
    grown = cv2.dilate(foot, kernel)
    dx = -np.cos(np.radians(light_dir_deg))   # shadow falls away from the light
    M = np.float32([[1, 0, dx * ext * 0.3], [0, 1, ext * 0.45]])
    grown = cv2.warpAffine(grown, M, (w, h), flags=cv2.INTER_NEAREST)
    shadow = ((grown > 0) & (foot == 0) & (alpha < 8)).astype(np.float32)
    if float(shadow.sum()) < 16:
        return out_rgba, info

    k = int(max(3, ext * float(sh.get("softness", 0.6)))) | 1
    shadow = cv2.GaussianBlur(shadow, (k, k), 0)
    op = float(np.clip(gstr * sh.get("strength", 0.35), sh.get("min_opacity", 0.04), sh.get("max_opacity", 0.5)))
    sh3 = (shadow * op)[:, :, None]
    new_rgb = rgb * (1 - sh3) + np.array([18, 16, 15], np.float32) * sh3
    new_alpha = np.clip(alpha + shadow * op * 255.0, 0, 255)
    info.update({"applied": True, "intensity": round(op, 3), "extend_px": ext})
    return np.dstack([np.clip(new_rgb, 0, 255), new_alpha]).astype("uint8"), info


def _edge_blend(rgba):
    """Soften the patch/original boundary and remove colour fringe (no seam)."""
    eb = lm.stage("edge_blend", {"enabled": True, "feather_px": 3, "decontaminate": True})
    if not eb.get("enabled", True):
        return rgba
    rgb = rgba[:, :, :3].astype(np.float32)
    a = rgba[:, :, 3].astype(np.float32)
    f = int(eb.get("feather_px", 3)) | 1
    a2 = cv2.GaussianBlur(a, (f, f), 0)
    if eb.get("decontaminate", True):
        edge = (a > 16) & (a < 240)
        if edge.any():
            blur_rgb = cv2.GaussianBlur(rgb, (f, f), 0)
            m = (edge[:, :, None].astype(np.float32)) * 0.5
            rgb = rgb * (1 - m) + blur_rgb * m
    return np.dstack([np.clip(rgb, 0, 255), np.clip(a2, 0, 255)]).astype("uint8")


def _brief(a):
    return {k: a[k] for k in ("brightness", "color_temperature", "contrast", "saturation",
                              "white_balance", "light_direction_deg", "median_L") if k in a}


def harmonize_patch(patch_rgba, reference_rgb, patch_left=0, patch_top=0, options=None):
    """Core harmonization. Returns (out_rgba, report)."""
    options = options or {}
    cfg = lm.config()
    h, w = patch_rgba.shape[:2]
    rgb = patch_rgba[:, :, :3].astype(np.float32)
    alpha = patch_rgba[:, :, 3] if patch_rgba.shape[2] == 4 else np.full((h, w), 255, "uint8")
    foot = alpha > 128
    full_image = int(foot.sum()) < 16 or bool((alpha > 250).all())
    if full_image:
        foot = np.ones((h, w), bool)
        gstr = float(options.get("strength", cfg.get("full_image_strength", 0.4)))
    else:
        gstr = float(options.get("strength", cfg.get("strength", 0.7)))

    # ambient = the original scene around the object (reference minus the footprint)
    rh, rw = reference_rgb.shape[:2]
    ref_obj = np.zeros((rh, rw), bool)
    y0, x0 = int(patch_top), int(patch_left)
    y1, x1 = min(rh, y0 + h), min(rw, x0 + w)
    if y1 > y0 and x1 > x0:
        ref_obj[y0:y1, x0:x1] = foot[:y1 - y0, :x1 - x0]
    ambient = ~ref_obj
    if int(ambient.sum()) < 64:
        ambient = np.ones((rh, rw), bool)

    before = lm.analyze(patch_rgba[:, :, :3], (foot.astype("uint8") * 255))
    target = lm.analyze(reference_rgb, (ambient.astype("uint8") * 255))
    src_s = _lab_stats(rgb.clip(0, 255).astype("uint8"), foot.astype("uint8"))
    tgt_s = _lab_stats(reference_rgb, ambient.astype("uint8"))

    out_rgb = _apply_corrections(rgb, src_s, tgt_s, gstr)
    out_rgba = np.dstack([out_rgb, alpha]).astype("uint8")

    shadow_info = {"applied": False}
    if not full_image:
        out_rgba, shadow_info = _contact_shadow(out_rgba, target.get("light_direction_deg", 0.0), gstr)
    out_rgba = _edge_blend(out_rgba)

    after = lm.analyze(out_rgba[:, :, :3], ((out_rgba[:, :, 3] > 128).astype("uint8") * 255))
    report = {
        "strength": gstr, "full_image": full_image, "shadow": shadow_info,
        "before": _brief(before), "after": _brief(after), "target": _brief(target),
        "histogram_distance_before": lm.histogram_distance(before["histogram"], target["histogram"]),
        "histogram_distance_after": lm.histogram_distance(after["histogram"], target["histogram"]),
    }
    return out_rgba, report


# -- endpoint-facing API (never raises) --------------------------------------

def harmonize(patch_image, reference_image, patch_left=0, patch_top=0, options=None):
    """Endpoint wrapper. On ANY failure returns the original patch unchanged."""
    if not patch_image or not reference_image:
        return {"png": patch_image, "ok": False, "report": {"error": "patch and reference required"}}
    try:
        patch = _decode_data_url(patch_image)                 # RGBA
        reference = _decode_data_url(reference_image)[:, :, :3]
        out, report = harmonize_patch(patch, reference, int(patch_left or 0), int(patch_top or 0), options)
        return {"png": _data_url(out), "ok": True, "report": report,
                "capabilities": {"engine": "opencv_lighting", "rerender_sdxl": False}}
    except Exception as exc:  # noqa: BLE001 - safety: never crash, never block
        return {"png": patch_image, "ok": False, "report": {"error": str(exc)}}
