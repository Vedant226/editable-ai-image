import cv2
import numpy as np
import torch
import os
import json
import easyocr

from groundingdino.util.inference import (
    load_model,
    load_image,
    predict
)

from segment_anything import (
    sam_model_registry,
    SamPredictor,
    SamAutomaticMaskGenerator
)

# ---------------------------------
# CONFIG
# ---------------------------------

IMAGE_PATH = "uploads/test.png"

TEXT_PROMPT = (
    "face . person . human . "
    "portrait . historical portrait . "
    "painting portrait . painting face . "
    "historical figure . illustrated face . "
    "royal portrait . head . "
    "text . title . subtitle . "
    "book title . heading . "
    "typography . author name . "
    "letters . words . sentence . "
    "ornament . decorative border . "
    "frame . emblem . logo . "
    "symbol . flower . decoration . "
    "illustration . building . "
    "object . clothing . animal . "
    "crown . sword . sky . "
    "clouds . paper texture . "
    "background"
)

BOX_THRESHOLD = 0.15
TEXT_THRESHOLD = 0.10

GROUNDING_MODEL_CONFIG = (
    "venv/lib/python3.11/site-packages/"
    "groundingdino/config/"
    "GroundingDINO_SwinT_OGC.py"
)

GROUNDING_MODEL_WEIGHTS = (
    "grounding_weights/"
    "groundingdino_swint_ogc.pth"
)

SAM_MODEL_PATH = (
    "models/sam_vit_h_4b8939.pth"
)

OUTPUT_DIR = "final_layers"

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ---------------------------------
# CREATE OUTPUT
# ---------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ---------------------------------
# LOAD MODELS
# ---------------------------------

print("Loading GroundingDINO...")

grounding_model = load_model(
    GROUNDING_MODEL_CONFIG,
    GROUNDING_MODEL_WEIGHTS
)

print("Loading SAM...")

sam = sam_model_registry[
    "vit_h"
](checkpoint=SAM_MODEL_PATH)

sam.to(device=DEVICE)

predictor = SamPredictor(sam)

mask_generator = (
    SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=100
    )
)

print("Loading OCR...")

reader = easyocr.Reader(
    ["en"]
)

# ---------------------------------
# LOAD IMAGE
# ---------------------------------

image_source, image = load_image(
    IMAGE_PATH
)

image_bgr = cv2.imread(
    IMAGE_PATH
)

image_rgb = cv2.cvtColor(
    image_bgr,
    cv2.COLOR_BGR2RGB
)

height, width, _ = image_rgb.shape

predictor.set_image(
    image_rgb
)

metadata = []

# ---------------------------------
# PART 1
# GroundingDINO detection
# ---------------------------------

print("Detecting semantic objects...")

boxes, logits, phrases = predict(
    model=grounding_model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)

print(
    f"Found {len(phrases)} semantic objects"
)

for idx, box in enumerate(boxes):

    x_center, y_center, bw, bh = box

    x1 = int(
        (x_center - bw / 2)
        * width
    )

    y1 = int(
        (y_center - bh / 2)
        * height
    )

    x2 = int(
        (x_center + bw / 2)
        * width
    )

    y2 = int(
        (y_center + bh / 2)
        * height
    )

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    input_box = np.array([
        x1,
        y1,
        x2,
        y2
    ])

    masks, _, _ = predictor.predict(
        box=input_box,
        multimask_output=False
    )

    mask = masks[0]

    rgba = np.zeros(
        (height, width, 4),
        dtype=np.uint8
    )

    rgba[:, :, :3] = image_rgb
    rgba[:, :, 3] = mask * 255

    cropped = rgba[
        y1:y2,
        x1:x2
    ]

    label = (
        phrases[idx]
        .replace(" ", "_")
        .replace("/", "_")
    )

    filename = (
        f"{idx}_{label}.png"
    )

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    cv2.imwrite(
        path,
        cv2.cvtColor(
            cropped,
            cv2.COLOR_RGBA2BGRA
        )
    )

    metadata.append({
        "id": idx,
        "file": filename,
        "type": label,
        "x": x1,
        "y": y1,
        "width": x2 - x1,
        "height": y2 - y1,
        "rotation": 0,
        "zIndex": 5
    })

# ---------------------------------
# PART 2
# OCR TEXT EXTRACTION
# ---------------------------------

print("Extracting text...")

results = reader.readtext(
    IMAGE_PATH
)

for i, result in enumerate(results):

    bbox, text, confidence = result

    if confidence < 0.2:
        continue

    x1 = int(bbox[0][0])
    y1 = int(bbox[0][1])

    x2 = int(bbox[2][0])
    y2 = int(bbox[2][1])

    crop = image_rgb[
        y1:y2,
        x1:x2
    ]

    filename = (
        f"text_{i}.png"
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            filename
        ),
        cv2.cvtColor(
            crop,
            cv2.COLOR_RGB2BGR
        )
    )

    metadata.append({
    "id": 1000 + i,

    "file": filename,

    "type": "text",

    "text": text,

    # typography
    "fontFamily":
        "Cinzel",

    "fontWeight":
        "bold",

    "fontColor":
        "#d8b36a",

    "strokeColor":
        "#5a2e12",

    "strokeWidth":
        1.5,

    "textAlign":
        "center",

    "letterSpacing":
        1,

    "shadowBlur":
        2,

    # placement
    "x": x1,
    "y": y1,
    "width": x2 - x1,
    "height": y2 - y1,

    "rotation": 0,

    "zIndex": 10
})

# ---------------------------------
# PART 3
# BACKGROUND LAYER
# ---------------------------------

cv2.imwrite(
    os.path.join(
        OUTPUT_DIR,
        "background.png"
    ),
    image_bgr
)

metadata.append({
    "id": 99999,
    "file": "background.png",
    "type": "background",
    "x": 0,
    "y": 0,
    "width": width,
    "height": height,
    "rotation": 0,
    "zIndex": 0
})

# ---------------------------------
# SAVE METADATA
# ---------------------------------

metadata.sort(
    key=lambda x:
    x.get("zIndex", 0)
)

with open(
    os.path.join(
        OUTPUT_DIR,
        "metadata.json"
    ),
    "w"
) as f:
    json.dump(
        metadata,
        f,
        indent=4
    )

print("Done!")