import cv2
import numpy as np
import torch
import os
import json

from groundingdino.util.inference import (
    load_model,
    load_image,
    predict
)

from segment_anything import (
    sam_model_registry,
    SamPredictor
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
    "book title . subtitle . "
    "author name . text . "
    "portrait frame . decorative emblem . "
    "ornament . ornament border . "
    "decoration . illustration . "
    "symbol . logo . book cover text"
)

BOX_THRESHOLD = 0.18
TEXT_THRESHOLD = 0.12

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

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ---------------------------------
# OUTPUT FOLDERS
# ---------------------------------

OUTPUT_DIR = "final_layers"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ---------------------------------
# LOAD MODELS
# ---------------------------------

print("Loading Grounding DINO...")

grounding_model = load_model(
    GROUNDING_MODEL_CONFIG,
    GROUNDING_MODEL_WEIGHTS
)

print("Grounding DINO loaded!")

print("Loading SAM...")

sam = sam_model_registry[
    "vit_h"
](checkpoint=SAM_MODEL_PATH)

sam.to(device=DEVICE)

predictor = SamPredictor(sam)

print("SAM loaded!")

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

predictor.set_image(image_rgb)

height, width, _ = image_rgb.shape

# ---------------------------------
# DETECT OBJECTS
# ---------------------------------

print("Detecting objects...")

boxes, logits, phrases = predict(
    model=grounding_model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)

print(
    f"Found {len(phrases)} objects"
)

metadata = []

# ---------------------------------
# EXTRACT OBJECTS
# ---------------------------------

for idx, box in enumerate(boxes):

    x_center, y_center, bw, bh = box

    x1 = int(
        (x_center - bw / 2) * width
    )

    y1 = int(
        (y_center - bh / 2) * height
    )

    x2 = int(
        (x_center + bw / 2) * width
    )

    y2 = int(
        (y_center + bh / 2) * height
    )

    # Clamp bounds
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

    masks, scores, _ = predictor.predict(
        box=input_box,
        multimask_output=False
    )

    mask = masks[0]

    # Create RGBA
    rgba = np.zeros(
        (
            height,
            width,
            4
        ),
        dtype=np.uint8
    )

    rgba[:, :, :3] = image_rgb
    rgba[:, :, 3] = mask * 255

    # Crop exact object region
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

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            cropped,
            cv2.COLOR_RGBA2BGRA
        )
    )

    metadata.append({
        "id": idx,
        "file": filename,
        "x": x1,
        "y": y1,
        "width": x2 - x1,
        "height": y2 - y1,
        "type": label,
        "rotation": 0,
        "scaleX": 1,
        "scaleY": 1
    })

    print(
        f"Saved {filename}"
    )

# ---------------------------------
# SAVE JSON
# ---------------------------------

metadata_path = os.path.join(
    OUTPUT_DIR,
    "metadata.json"
)

with open(
    metadata_path,
    "w"
) as f:
    json.dump(
        metadata,
        f,
        indent=4
    )

print(
    f"Metadata saved -> {metadata_path}"
)

print("Done!")