import cv2
import easyocr
import os

# Load image
IMAGE_PATH = "uploads/test.png"

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Image not found!")
    exit()

print("Image loaded!")

# Create folders
os.makedirs("text_layers", exist_ok=True)

# OCR Reader
reader = easyocr.Reader(['en'])

print("Detecting text...")

results = reader.readtext(IMAGE_PATH)

print(f"Found {len(results)} text regions")

for i, result in enumerate(results):

    bbox, text, confidence = result

    print(
        f"{i+1}. Text: {text} "
        f"(Confidence: {confidence:.2f})"
    )

    # Bounding box coordinates
    top_left = tuple(map(int, bbox[0]))
    bottom_right = tuple(map(int, bbox[2]))

    # Crop text region
    cropped = image[
        top_left[1]:bottom_right[1],
        top_left[0]:bottom_right[0]
    ]

    output_path = (
        f"text_layers/text_{i}.png"
    )

    cv2.imwrite(output_path, cropped)

print("Text extraction complete!")