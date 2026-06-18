"""
Thin FastAPI inpaint service for the editable-AI-image editor.

The frontend lifts an object as a Smart Object and asks this service to fill the
hole it leaves in the original image. We build the inpaint mask from the
object's own PNG alpha footprint (the Visual Object Resolver decides *which*
object; the pixel mask is rasterised here), run LaMa (or OpenCV fallback) over a
context crop around the object for speed, and return the patch cropped to the
object's footprint as a base64 PNG.

  POST /inpaint { "objectId": <int> } -> { x, y, w, h, png }
  GET  /health                        -> { engine, device, objects }

Run:  ./run.sh        (from backend/)   — serves on 127.0.0.1:8000
"""

import base64
import io
import json
import os

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from lama_engine import InpaintEngine
from matting import refine_matte, decontaminate
from shadow import synthesize_shadow, shadow_params

LAYERS_DIR = os.path.abspath(
    os.environ.get(
        "LAYERS_DIR",
        os.path.join(os.path.dirname(__file__), "..", "editable-editor", "public", "layers"),
    )
)
MASK_DILATE = int(os.environ.get("MASK_DILATE", "7"))  # px, hides anti-aliased edges
CONTEXT_MARGIN = int(os.environ.get("CONTEXT_MARGIN", "64"))  # px of context for LaMa
FILL_FEATHER = int(os.environ.get("FILL_FEATHER", "9"))  # px collar that blends fill into original

app = FastAPI(title="Editable AI — Inpaint")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = InpaintEngine()

with open(os.path.join(LAYERS_DIR, "metadata.json")) as fh:
    META = {int(o["id"]): o for o in json.load(fh)}

_background = None


def background_rgb():
    global _background
    if _background is None:
        img = Image.open(os.path.join(LAYERS_DIR, "background.png")).convert("RGB")
        _background = np.array(img)
    return _background


def _data_url(rgba_or_rgb):
    buf = io.BytesIO()
    Image.fromarray(rgba_or_rgb).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _bbox_of(boolmask):
    ys, xs = np.where(boolmask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def build_object_mask(obj, H, W):
    """Full-canvas binary mask from the object's PNG alpha (full rect if opaque)."""
    x, y, w, h = int(obj["x"]), int(obj["y"]), int(obj["width"]), int(obj["height"])
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    png = Image.open(os.path.join(LAYERS_DIR, obj["file"]))
    has_alpha = "A" in png.getbands()
    mask = np.zeros((H, W), dtype="uint8")
    if has_alpha:
        alpha = np.array(png.split()[-1].resize((w, h)))
        local = (alpha > 12).astype("uint8") * 255
    else:
        local = np.full((h, w), 255, dtype="uint8")
    mask[y0:y1, x0:x1] = local[(y0 - y) : (y1 - y), (x0 - x) : (x1 - x)]
    return mask, has_alpha, (x0, y0, x1, y1)


def inpaint_full(bg, mask, margin=CONTEXT_MARGIN):
    """Inpaint the masked region over a context crop; return a full-canvas copy."""
    H, W = bg.shape[:2]
    box = _bbox_of(mask > 0)
    if box is None:
        return bg.copy()
    x0, y0, x1, y1 = box
    cx0, cy0 = max(0, x0 - margin), max(0, y0 - margin)
    cx1, cy1 = min(W, x1 + margin), min(H, y1 + margin)
    res = engine.inpaint(bg[cy0:cy1, cx0:cx1], mask[cy0:cy1, cx0:cx1])
    filled = bg.copy()
    filled[cy0:cy1, cx0:cx1] = res
    return filled


class InpaintRequest(BaseModel):
    objectId: int


@app.get("/health")
def health():
    return {"engine": engine.engine, "device": engine.device, "objects": len(META)}


@app.post("/inpaint")
def inpaint(req: InpaintRequest):
    obj = META.get(req.objectId)
    if obj is None:
        raise HTTPException(status_code=404, detail=f"object {req.objectId} not found")
    if not os.path.exists(os.path.join(LAYERS_DIR, obj.get("file", ""))):
        raise HTTPException(status_code=404, detail=f"layer file missing: {obj.get('file')}")

    bg = background_rgb()
    H, W = bg.shape[:2]
    x, y, w, h = int(obj["x"]), int(obj["y"]), int(obj["width"]), int(obj["height"])

    # footprint clipped to the canvas
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        raise HTTPException(status_code=400, detail="object footprint is empty")

    # mask: object alpha (or full rectangle for opaque crops like text)
    png = Image.open(os.path.join(LAYERS_DIR, obj["file"]))
    mask = np.zeros((H, W), dtype="uint8")
    if "A" in png.getbands():
        alpha = np.array(png.split()[-1].resize((w, h)))
        local = (alpha > 12).astype("uint8") * 255
    else:
        local = np.full((h, w), 255, dtype="uint8")
    mask[y0:y1, x0:x1] = local[(y0 - y) : (y1 - y), (x0 - x) : (x1 - x)]

    if MASK_DILATE > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_DILATE, MASK_DILATE))
        mask = cv2.dilate(mask, kernel)

    # context crop around the footprint — LaMa on a small region is much faster
    cx0, cy0 = max(0, x0 - CONTEXT_MARGIN), max(0, y0 - CONTEXT_MARGIN)
    cx1, cy1 = min(W, x1 + CONTEXT_MARGIN), min(H, y1 + CONTEXT_MARGIN)
    crop_result = engine.inpaint(bg[cy0:cy1, cx0:cx1], mask[cy0:cy1, cx0:cx1])

    # extract just the object's footprint from the inpainted crop
    patch = crop_result[(y0 - cy0) : (y1 - cy0), (x0 - cx0) : (x1 - cx0)]

    buf = io.BytesIO()
    Image.fromarray(patch).save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "png": data_url}


class LiftRequest(BaseModel):
    objectId: int


@app.post("/lift")
def lift(req: LiftRequest):
    """
    Full Lift Package for the Editing Illusion Engine:
      cutout  — ORIGINAL pixels × refined, decontaminated matte (the float)
      fill    — inpaint patch for footprint (+ shadow region), feathered collar
      shadow  — synthesized soft black shadow matte the frontend animates
    """
    obj = META.get(req.objectId)
    if obj is None:
        raise HTTPException(status_code=404, detail=f"object {req.objectId} not found")
    if not os.path.exists(os.path.join(LAYERS_DIR, obj.get("file", ""))):
        raise HTTPException(status_code=404, detail=f"layer file missing: {obj.get('file')}")

    bg = background_rgb()
    H, W = bg.shape[:2]
    obj_mask, has_alpha, (ox0, oy0, ox1, oy1) = build_object_mask(obj, H, W)
    if ox1 <= ox0 or oy1 <= oy0:
        raise HTTPException(status_code=400, detail="object footprint is empty")
    w, h = ox1 - ox0, oy1 - oy0
    obj_alpha01 = obj_mask.astype(np.float32) / 255.0

    # synthesize an attached shadow (real cutouts only; not opaque text rects)
    sp = shadow_params(w, h)
    shadow_full = (
        synthesize_shadow(obj_alpha01, blur=sp["blur"], dx=sp["dx"], dy=sp["dy"])
        if has_alpha
        else None
    )

    # inpaint mask = dilated footprint (∪ shadow region, so the original shadow goes too)
    dil = cv2.dilate(obj_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MASK_DILATE, MASK_DILATE)))
    inpaint_mask = dil.copy()
    if shadow_full is not None:
        inpaint_mask = np.maximum(inpaint_mask, (shadow_full > 0.06).astype("uint8") * 255)
    filled = inpaint_full(bg, inpaint_mask)

    # fill patch: opaque over the region, feathering to 0 just outside → blends into original
    fk = FILL_FEATHER | 1
    expanded = cv2.dilate(inpaint_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fk, fk)))
    collar = cv2.GaussianBlur(expanded.astype(np.float32), (fk, fk), 0) / 255.0
    fx0, fy0, fx1, fy1 = _bbox_of(collar > 0.02)
    fill_rgba = np.dstack(
        [filled[fy0:fy1, fx0:fx1], np.clip(collar[fy0:fy1, fx0:fx1] * 255.0, 0, 255)]
    ).astype("uint8")

    # cutout: refined matte × original, decontaminated using the inpaint as background B
    pad = FILL_FEATHER + 4
    mx0, my0 = max(0, ox0 - pad), max(0, oy0 - pad)
    mx1, my1 = min(W, ox1 + pad), min(H, oy1 + pad)
    crop_rgb = bg[my0:my1, mx0:mx1]
    alpha = refine_matte(crop_rgb, obj_mask[my0:my1, mx0:mx1], radius=max(4, round(min(w, h) * 0.04)))
    cutout_pad = decontaminate(crop_rgb, alpha, filled[my0:my1, mx0:mx1])
    # crop the cutout to the exact footprint so the frontend can drop it in at
    # the object's transform (same bbox the old SAM PNG used)
    fyo, fxo = oy0 - my0, ox0 - mx0
    cutout_rgba = cutout_pad[fyo : fyo + h, fxo : fxo + w]

    resp = {
        "objectId": req.objectId,
        "engine": engine.engine,
        "footprint": {"x": ox0, "y": oy0, "w": w, "h": h},
        "cutout": {
            "x": ox0, "y": oy0, "w": w, "h": h,
            "png": _data_url(cutout_rgba),
        },
        "fill": {
            "x": fx0, "y": fy0, "w": fx1 - fx0, "h": fy1 - fy0,
            "png": _data_url(fill_rgba),
        },
        "shadow": None,
    }
    if shadow_full is not None:
        sbox = _bbox_of(shadow_full > 0.02)
        if sbox:
            sx0, sy0, sx1, sy1 = sbox
            sh_a = np.clip(shadow_full[sy0:sy1, sx0:sx1] * 255.0, 0, 255).astype("uint8")
            sh_rgba = np.dstack([np.zeros((*sh_a.shape, 3), "uint8"), sh_a])
            resp["shadow"] = {
                "x": sx0, "y": sy0, "w": sx1 - sx0, "h": sy1 - sy0,
                "opacity": sp["opacity"],
                "png": _data_url(sh_rgba),
            }
    return resp
