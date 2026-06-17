"""
Phase X1 — universal proposal generation.

Two complementary sources, fused into one candidate pool:
  - SAM SamAutomaticMaskGenerator: class-agnostic "segment everything" (RECALL —
    this is what surfaces decorative elements the prompt never named).
  - GroundingDINO boxes → SAM box-prompted masks: SEMANTIC anchor (named objects
    with a label, used later for hierarchy + labeling).

Dedup/labeling/refinement happen in later phases. Each candidate keeps its
`source` so downstream stages can reason about provenance.
"""

import json
import os

import numpy as np
import torch
from PIL import Image

from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
from groundingdino.util.inference import load_model, load_image, predict

from . import config as C

_state = {}


def load_models():
    if "amg" not in _state:
        sam = sam_model_registry[C.SAM_TYPE](checkpoint=C.SAM_CHECKPOINT).to(C.DEVICE)
        _state["amg"] = SamAutomaticMaskGenerator(sam, **C.AMG_PARAMS)
        _state["predictor"] = SamPredictor(sam)
        _state["dino"] = load_model(C.DINO_CONFIG, C.DINO_WEIGHTS)
    return _state["amg"], _state["predictor"], _state["dino"]


def _bbox_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def sam_amg_proposals(image_rgb):
    amg, _, _ = load_models()
    out = []
    for m in amg.generate(image_rgb):
        seg = m["segmentation"]
        bb = m.get("bbox")
        bb = [int(v) for v in bb] if bb else _bbox_from_mask(seg)
        if bb is None or bb[2] <= 0 or bb[3] <= 0:
            continue
        out.append(
            {
                "mask": seg,
                "bbox": bb,
                "area": int(m["area"]),
                "score": float(m.get("predicted_iou", 0.0)),
                "stability": float(m.get("stability_score", 0.0)),
                "source": "sam",
                "label": None,
            }
        )
    return out


def dino_proposals(image_rgb, image_tensor):
    _, predictor, dino = load_models()
    H, W = image_rgb.shape[:2]
    boxes, logits, phrases = predict(
        model=dino,
        image=image_tensor,
        caption=C.DINO_PROMPT,
        box_threshold=C.DINO_BOX_THRESHOLD,
        text_threshold=C.DINO_TEXT_THRESHOLD,
    )
    predictor.set_image(image_rgb)
    out = []
    for box, logit, phrase in zip(boxes, logits, phrases):
        cx, cy, bw, bh = box.tolist()
        x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
        x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        masks, _, _ = predictor.predict(box=np.array([x1, y1, x2, y2]), multimask_output=False)
        seg = masks[0]
        bb = _bbox_from_mask(seg)
        if bb is None:
            continue
        out.append(
            {
                "mask": seg,
                "bbox": bb,
                "area": int(seg.sum()),
                "score": float(logit),
                "stability": 1.0,
                "source": "dino",
                "label": str(phrase),
            }
        )
    return out


def generate(image_path=None):
    image_path = image_path or C.IMAGE_PATH
    image_rgb = np.array(Image.open(image_path).convert("RGB"))
    _, image_tensor = load_image(image_path)
    sam = sam_amg_proposals(image_rgb)
    if C.DEVICE == "cuda":
        torch.cuda.empty_cache()  # release AMG activations before the DINO->SAM pass
    dino = dino_proposals(image_rgb, image_tensor)
    return image_rgb, sam, dino


def save_candidates(candidates, work_dir=None):
    work_dir = work_dir or C.WORK_DIR
    masks_dir = os.path.join(work_dir, "cand_masks")
    os.makedirs(masks_dir, exist_ok=True)
    manifest = []
    for i, c in enumerate(candidates):
        x, y, w, h = c["bbox"]
        crop = (c["mask"][y : y + h, x : x + w].astype("uint8")) * 255
        Image.fromarray(crop, "L").save(os.path.join(masks_dir, f"cand_{i}.png"))
        manifest.append(
            {
                "id": i,
                "source": c["source"],
                "bbox": c["bbox"],
                "area": c["area"],
                "score": round(c["score"], 3),
                "stability": round(c.get("stability", 0.0), 3),
                "label": c["label"],
            }
        )
    with open(os.path.join(work_dir, "candidates.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest
