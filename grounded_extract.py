import cv2
import torch
import os

from groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
)

# ------------------------
# CONFIG
# ------------------------

IMAGE_PATH = "uploads/test.png"

TEXT_PROMPT = (
    "book . portrait . face . "
    "ornament . text . border"
)

BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25

MODEL_CONFIG = (
    "venv/lib/python3.11/site-packages/"
    "groundingdino/config/"
    "GroundingDINO_SwinT_OGC.py"
)

MODEL_WEIGHTS = (
    "grounding_weights/"
    "groundingdino_swint_ogc.pth"
)

# ------------------------
# LOAD MODEL
# ------------------------

print("Loading Grounding DINO...")

model = load_model(
    MODEL_CONFIG,
    MODEL_WEIGHTS
)

print("Model loaded!")

# ------------------------
# LOAD IMAGE
# ------------------------

image_source, image = load_image(
    IMAGE_PATH
)

print("Detecting objects...")

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD
)

print(f"Found {len(phrases)} objects")

# ------------------------
# SAVE CROPS
# ------------------------

os.makedirs(
    "detected_objects",
    exist_ok=True
)

h, w, _ = image_source.shape

for idx, box in enumerate(boxes):

    x_center, y_center, bw, bh = box

    x1 = int((x_center - bw / 2) * w)
    y1 = int((y_center - bh / 2) * h)

    x2 = int((x_center + bw / 2) * w)
    y2 = int((y_center + bh / 2) * h)

    crop = image_source[
        max(0, y1):min(h, y2),
        max(0, x1):min(w, x2)
    ]

    label = phrases[idx]

    filename = (
        f"detected_objects/"
        f"{idx}_{label}.png"
    )

    cv2.imwrite(
        filename,
        cv2.cvtColor(
            crop,
            cv2.COLOR_RGB2BGR
        )
    )

    print(
        f"Saved: {filename}"
    )

print("Done!")