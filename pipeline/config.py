"""
Configuration for the Universal Object Extraction Pipeline.

Paths are resolved relative to the repo root so the pipeline can be run with
`python -m pipeline.run_x1` from the repo root.
"""

import os

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_PATH = os.path.join(ROOT, "uploads", "test.png")
WORK_DIR = os.path.join(ROOT, "pipeline", "_work")  # debug/intermediate artifacts (gitignored)

SAM_CHECKPOINT = os.path.join(ROOT, "models", "sam_vit_h_4b8939.pth")
SAM_TYPE = "vit_h"

DINO_CONFIG = os.path.join(
    ROOT, "venv/lib/python3.11/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
DINO_WEIGHTS = os.path.join(ROOT, "grounding_weights", "groundingdino_swint_ogc.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- SAM "segment everything": tuned for recall within an ~8GB GPU budget ---
# crop_n_layers=0: a crop pass (=1) adds extra vit_h encodes and OOMs an 8GB GPU.
# Small-decor recall via crops is added later as a memory-bounded tiling pass.
# points_per_batch kept low to cap the decode-time memory spike.
AMG_PARAMS = dict(
    points_per_side=32,
    points_per_batch=32,
    pred_iou_thresh=0.82,
    stability_score_thresh=0.90,
    min_mask_region_area=64,
    crop_n_layers=0,
    box_nms_thresh=0.7,
)

# --- GroundingDINO: semantic anchor (inclusive thresholds; SAM-AMG drives recall) ---
DINO_BOX_THRESHOLD = 0.20
DINO_TEXT_THRESHOLD = 0.18
DINO_PROMPT = " . ".join(
    [
        "person", "man", "woman", "child", "face", "head", "hair", "eye", "hand",
        "crown", "hat", "helmet", "halo",
        "sword", "weapon", "spear", "shield", "armor", "staff", "scepter",
        "clothing", "robe", "cape", "dress", "collar",
        "book", "scroll", "banner", "flag",
        "flower", "plant", "tree", "leaf", "animal", "horse", "bird", "lion",
        "logo", "emblem", "badge", "crest", "coat of arms",
        "ornament", "decorative border", "frame", "embroidery", "pattern",
        "symbol", "icon", "jewelry", "necklace", "ring", "gem",
        "throne", "chair", "table", "furniture", "building", "pillar",
        "text", "title", "letter", "number",
        "cloud", "sky", "background",
    ]
)

# --- taxonomy used by CLIP labeling (X2) and normalization ---
TAXONOMY = [
    "person", "face", "head", "hair", "hand", "eye",
    "crown", "hat", "helmet",
    "sword", "weapon", "shield", "armor", "staff",
    "clothing", "robe", "cape", "collar",
    "book", "scroll", "banner",
    "flower", "plant", "tree", "animal",
    "logo", "emblem", "crest", "badge",
    "ornament", "border", "frame", "embroidery", "pattern",
    "symbol", "icon", "jewelry", "gem", "decoration",
    "throne", "furniture", "building", "column",
    "text",
    "sky", "cloud", "background", "wall", "floor", "texture",
]

# categories that should default to non-editable (retained, flagged)
IGNORE_CATEGORIES = {"background", "sky", "cloud", "wall", "floor", "texture"}
