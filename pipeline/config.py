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

# CLIP zero-shot prompt per category (default: "a photo of a {cat}")
CLIP_PROMPTS = {
    "text": "text, typography or lettering",
    "border": "a decorative border",
    "frame": "an ornate picture frame",
    "embroidery": "gold embroidery thread",
    "ornament": "a decorative ornament",
    "emblem": "an emblem",
    "crest": "a heraldic crest",
    "symbol": "a small symbol or icon",
    "icon": "an icon",
    "pattern": "a decorative pattern",
    "crown": "a royal crown",
    "face": "a human face",
    "person": "a person or portrait",
    "robe": "a robe",
    "gem": "a gemstone or jewel",
    "texture": "a plain background texture",
    "background": "a plain background",
}

# DINO phrase tokens -> taxonomy category (for tokens that aren't categories)
SYNONYMS = {
    "man": "person", "woman": "person", "child": "person", "boy": "person", "girl": "person",
    "portrait": "person", "figure": "person",
    "robe": "clothing", "dress": "clothing", "coat": "clothing", "cloak": "cape", "garment": "clothing",
    "necklace": "jewelry", "ring": "jewelry", "pendant": "jewelry",
    "spear": "weapon", "title": "text", "letter": "text", "number": "text", "word": "text",
    "halo": "ornament", "flourish": "ornament", "decorative": "ornament",
    "coat of arms": "crest", "arms": "crest", "badge": "emblem", "shield": "shield",
    "leaf": "plant", "lion": "animal", "horse": "animal", "bird": "animal",
}

# importance weight per category (drives the importance score; default 0.6)
IMPORTANCE_WEIGHTS = {
    "person": 1.0, "face": 0.95, "head": 0.7, "hair": 0.6, "hand": 0.5, "eye": 0.4,
    "crown": 0.9, "hat": 0.7, "helmet": 0.7,
    "sword": 0.85, "weapon": 0.85, "shield": 0.8, "armor": 0.75, "staff": 0.7,
    "clothing": 0.6, "robe": 0.6, "cape": 0.6, "collar": 0.6,
    "book": 0.75, "scroll": 0.7, "banner": 0.7,
    "flower": 0.7, "plant": 0.6, "tree": 0.5, "animal": 0.7,
    "logo": 0.8, "emblem": 0.82, "crest": 0.82, "badge": 0.78,
    "ornament": 0.72, "border": 0.65, "frame": 0.5, "embroidery": 0.72, "pattern": 0.6,
    "symbol": 0.7, "icon": 0.7, "jewelry": 0.75, "gem": 0.7, "decoration": 0.6,
    "throne": 0.6, "furniture": 0.5, "building": 0.4, "column": 0.4,
    "text": 0.85,
    "sky": 0.1, "cloud": 0.1, "background": 0.05, "wall": 0.1, "floor": 0.1, "texture": 0.1,
}

EDITABLE_MIN_AREA_FRAC = 0.0003  # below this = dust/noise
EDITABLE_MAX_AREA_FRAC = 0.60    # above this = whole-image/background
UNCERTAIN_THRESHOLD = 0.50       # confidence below this = flagged uncertain
