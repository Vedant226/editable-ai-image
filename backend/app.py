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

LAYERS_DIR = os.path.abspath(
    os.environ.get(
        "LAYERS_DIR",
        os.path.join(os.path.dirname(__file__), "..", "editable-editor", "public", "layers"),
    )
)
MASK_DILATE = int(os.environ.get("MASK_DILATE", "7"))  # px, hides anti-aliased edges
CONTEXT_MARGIN = int(os.environ.get("CONTEXT_MARGIN", "64"))  # px of context for LaMa

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
