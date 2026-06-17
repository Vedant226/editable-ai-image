import cv2
import os
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

MODEL_PATH = "models/sam_vit_h_4b8939.pth"

print("Loading SAM model...")

sam = sam_model_registry["vit_h"](
    checkpoint=MODEL_PATH
)

mask_generator = SamAutomaticMaskGenerator(sam)

print("Model loaded successfully!")

IMAGE_PATH = "uploads/test.png"

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Image not found!")
    exit()

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

print("Generating masks...")

masks = mask_generator.generate(image_rgb)

# Sort biggest objects first
masks = sorted(
    masks,
    key=lambda x: x['area'],
    reverse=True
)

print(f"Found {len(masks)} objects")

os.makedirs("outputs", exist_ok=True)

# Keep only top 10 biggest objects
top_masks = masks[:10]

for i, mask in enumerate(top_masks):

    segmented = image_rgb.copy()

    segmented[~mask['segmentation']] = 0

    output_path = f"outputs/object_{i}.png"

    cv2.imwrite(
        output_path,
        cv2.cvtColor(
            segmented,
            cv2.COLOR_RGB2BGR
        )
    )

print("Saved top 10 objects only!")