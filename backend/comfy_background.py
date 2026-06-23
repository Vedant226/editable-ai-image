"""
Background Replacement — built on the shared AI Editing Engine (comfy_engine.py).

Preserves EVERY foreground object and regenerates only the background, conditioned
on an uploaded background image. Unlike the object features this is a full-canvas
operation, so it keeps its own flow — but it now uses the engine's shared helpers,
graph injector and run-loop (no duplicated transport code) and the shared SDXL
inpaint template.

Pipeline:
  1. FOREGROUND mask = union of every object layer's alpha (metadata).
  2. Cover-fit the uploaded image to the canvas -> the new background.
  3. init = new background where foreground is absent, ORIGINAL pixels where it is.
  4. SetLatentNoiseMask over the BACKGROUND region only -> SDXL re-renders the
     backdrop, harmonising it to the (fixed) foreground latent.
  5. Re-composite the crisp ORIGINAL foreground at full resolution -> foreground
     preserved exactly; only the backdrop changed.
  6. Return the full canvas image (data URL) as the new base layer.

Config (env vars):
  COMFY_REPLACE_CKPT — SDXL checkpoint (falls back to the engine default)
  COMFY_BG_TIMEOUT   — seconds to wait for a render (default 240)
"""

import os

import cv2
import numpy as np

from comfy_client import ComfyUIClient
import comfy_capabilities
import comfy_style
from comfy_engine import (
    _round8, _data_url, _decode_rgb, _load_metadata_list, _background_rgb,
    _load_graph, inject_inpaint_graph, submit_and_fetch, apply_style, SHARED_TEMPLATE,
    LAYERS_DIR, CKPT_NAME as ENGINE_CKPT,
)
from comfy_recolor import _parse_color   # reuse the shared colour parser (color mode)

CKPT_NAME = os.environ.get("COMFY_REPLACE_CKPT", ENGINE_CKPT)
RENDER_TIMEOUT = float(os.environ.get("COMFY_BG_TIMEOUT", "240"))

FG_ALPHA_THRESH = 20    # alpha above this counts as foreground
FG_DILATE = 5           # px; tighten the seam so no original-bg halo survives
SEAM_FEATHER = 5        # px; soft foreground edge when re-compositing
TARGET_LONG = 1024      # SDXL working long-side (background is re-composited crisp)
DEFAULTS = {"steps": 28, "cfg": 7.0, "denoise": 0.6,
            "sampler": "dpmpp_2m", "scheduler": "karras", "seed": 0}

DEFAULT_POSITIVE = ("a cohesive background scene, consistent lighting, colour and "
                    "perspective with the foreground, depth, high detail, realistic")
DEFAULT_NEGATIVE = ("blurry, lowres, distorted, deformed, artifacts, seam, harsh edges, "
                    "extra objects, watermark, text")


class BackgroundError(Exception):
    """A background-replace request that cannot be fulfilled."""


def _foreground_alpha(meta, H, W):
    """Union of every object layer's alpha -> full-canvas foreground mask (0..255)."""
    fg = np.zeros((H, W), dtype="uint8")
    used = 0
    for o in meta:
        if int(o.get("id", -1)) == 99999:  # the base background itself
            continue
        x, y, w, h = int(o["x"]), int(o["y"]), int(o["width"]), int(o["height"])
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        path = os.path.join(LAYERS_DIR, o.get("file", ""))
        if not os.path.exists(path):
            continue
        from PIL import Image
        png = Image.open(path)
        if "A" in png.getbands():
            alpha = np.array(png.split()[-1].resize((w, h)))
            local = (alpha > FG_ALPHA_THRESH).astype("uint8") * 255
        else:
            local = np.full((h, w), 255, dtype="uint8")
        region = fg[y0:y1, x0:x1]
        np.maximum(region, local[(y0 - y):(y1 - y), (x0 - x):(x1 - x)], out=region)
        used += 1
    if used == 0 or fg.max() == 0:
        raise BackgroundError("no foreground objects found to preserve")
    return fg


def _cover_fit(img_rgb, W, H):
    """Resize+center-crop the uploaded image to exactly WxH (cover, no distortion)."""
    ih, iw = img_rgb.shape[:2]
    scale = max(W / iw, H / ih)
    nw, nh = max(W, int(round(iw * scale))), max(H, int(round(ih * scale)))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    ox, oy = (nw - W) // 2, (nh - H) // 2
    return resized[oy:oy + H, ox:ox + W]


def _full_result(image, mode, engine="cpu", prompt_id=None):
    """Full-canvas result descriptor (RGB or RGBA), shaped like the replace path."""
    return {"x": 0, "y": 0, "w": int(image.shape[1]), "h": int(image.shape[0]),
            "png": _data_url(image), "engine": engine, "promptId": prompt_id,
            "full": True, "mode": mode}


def _cpu_background(mode, bg, fg, options):
    """Fast CPU background edits — no SDXL, no upload, foreground preserved exactly:
    blur (depth-of-field), color (solid fill), remove (transparent background)."""
    H, W = bg.shape[:2]
    fk = SEAM_FEATHER | 1
    fg_soft = (cv2.GaussianBlur(fg.astype(np.float32), (fk, fk), 0) / 255.0)[:, :, None]
    if mode == "blur":
        strength = options.get("blur")
        if strength is None:
            strength = options.get("strength")
        strength = max(0.05, min(1.0, float(strength if strength is not None else 0.5)))
        k = int(max(3, round(strength * 0.05 * max(H, W)))) | 1
        blurred = cv2.GaussianBlur(bg, (k, k), 0)
        final = (blurred * (1 - fg_soft) + bg * fg_soft).astype("uint8")
        return _full_result(final, mode)
    if mode == "color":
        try:
            r, g, b = _parse_color(options.get("targetColor") or options.get("color") or "#ffffff")
        except Exception:  # noqa: BLE001
            r, g, b = (255, 255, 255)
        fill = np.empty_like(bg); fill[:] = (r, g, b)
        final = (fill * (1 - fg_soft) + bg * fg_soft).astype("uint8")
        return _full_result(final, mode)
    # remove -> foreground over a transparent background (RGBA)
    alpha = cv2.GaussianBlur(fg.astype(np.float32), (fk, fk), 0).clip(0, 255).astype("uint8")
    rgba = np.dstack([bg, alpha]).astype("uint8")
    return _full_result(rgba, mode)


def replace_background(replacement_data_url, options=None, client=None):
    """Replace OR otherwise edit the background, preserving all foreground objects.

    `options["mode"]` selects: replace (default, upload) / generate (prompt) /
    blur / color / remove. Returns { x, y, w, h, png, engine, promptId, full, mode }
    where png is the full canvas; foreground objects are always kept intact.
    """
    options = options or {}
    mode = (options.get("mode") or "replace").lower()
    params = dict(DEFAULTS)
    for k in ("steps", "cfg", "denoise", "sampler", "scheduler", "seed", "ckpt"):
        if options.get(k) is not None:
            params[k] = options[k]

    meta = _load_metadata_list()
    bg = _background_rgb()
    H, W = bg.shape[:2]

    # foreground mask = union of every object's alpha (shared by every mode)
    fg = _foreground_alpha(meta, H, W)

    # ---- CPU modes (no SDXL, no upload): foreground always preserved exactly ----
    if mode in ("blur", "color", "remove"):
        return _cpu_background(mode, bg, fg, options)

    client = client or ComfyUIClient()
    soft_active = bool(comfy_capabilities.detect(client).get("differential_diffusion"))

    # 1) background region SDXL may change (everything the foreground does not cover)
    fg_tight = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (FG_DILATE, FG_DILATE)))
    bg_mask = 255 - fg_tight

    # 2/3) init behind the (fixed) foreground: the uploaded backdrop for "replace",
    #      or the original backdrop for "generate" (SDXL re-renders it from the prompt).
    if mode == "generate":
        if options.get("denoise") is None:
            params["denoise"] = 0.9   # regenerate the backdrop fully from the prompt
        init = bg.copy()
    else:  # "replace" (default): cover-fit the uploaded image behind the foreground
        if not replacement_data_url:
            raise BackgroundError("background replace requires an uploaded image")
        new_bg = _cover_fit(_decode_rgb(replacement_data_url), W, H)
        a = (fg_tight.astype(np.float32) / 255.0)[:, :, None]
        init = (new_bg * (1 - a) + bg * a).astype("uint8")

    # 4) downscale init + bg mask to a working size, upload, run SDXL
    scale = TARGET_LONG / float(max(H, W))
    nw, nh = _round8(W * scale), _round8(H * scale)
    init_w = cv2.resize(init, (nw, nh), interpolation=cv2.INTER_AREA)
    if soft_active:  # soft boundary -> DifferentialDiffusion blends fg/bg seam
        fsoft = (SEAM_FEATHER * 4) | 1
        soft = cv2.GaussianBlur(bg_mask, (fsoft, fsoft), 0)
        mask_w = cv2.resize(soft, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        mask_w = cv2.resize(bg_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    import io
    from PIL import Image
    init_png = io.BytesIO(); Image.fromarray(init_w).save(init_png, format="PNG")
    mask_rgb = np.dstack([mask_w, mask_w, mask_w])
    mask_png = io.BytesIO(); Image.fromarray(mask_rgb).save(mask_png, format="PNG")

    up_init = client.upload_image(init_png.getvalue(), "background_init.png", overwrite=True)
    up_mask = client.upload_image(mask_png.getvalue(), "background_mask.png", overwrite=True)

    _style = options.get("style") or "auto"
    _base_prompt = options.get("prompt") or DEFAULT_POSITIVE
    if _style == "auto":  # match the new backdrop to the (preserved) artwork's style
        _sk, _ = comfy_style.detect_style(bg)
        _suf = comfy_style.STYLE_SUFFIXES.get(_sk, "")
        positive = f"{_base_prompt.rstrip(' ,')}, {_suf}" if _suf else _base_prompt
    else:
        positive = apply_style(_base_prompt, _style)
    negative = options.get("negative") or DEFAULT_NEGATIVE
    graph = inject_inpaint_graph(
        _load_graph(SHARED_TEMPLATE),
        init_name=up_init["name"], mask_name=up_mask["name"],
        positive=positive, negative=negative, params=params,
        ckpt=params.get("ckpt", CKPT_NAME), soft_inpaint=soft_active,
    )
    result_w, prompt_id = submit_and_fetch(client, graph, RENDER_TIMEOUT)

    # 5) upscale the harmonised background to canvas, then re-composite the CRISP
    #    original foreground so it is preserved exactly (only the backdrop changed)
    result = cv2.resize(result_w, (W, H), interpolation=cv2.INTER_CUBIC)
    fk = SEAM_FEATHER | 1
    fg_soft = (cv2.GaussianBlur(fg.astype(np.float32), (fk, fk), 0) / 255.0)[:, :, None]
    final = (result * (1 - fg_soft) + bg * fg_soft).astype("uint8")

    return {
        "x": 0, "y": 0, "w": W, "h": H,
        "png": _data_url(final),
        "engine": "sdxl",
        "promptId": prompt_id,
        "full": True,
        "mode": mode,
    }
