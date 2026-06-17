import cv2
import numpy as np
from PIL import Image
from pathlib import Path

BACKGROUND_IMAGE = "final_layers/background.png"
MASK_FOLDER = "masks"
OUTPUT_PATH = "final_layers/clean_background.png"


def combine_masks(mask_folder):
    mask_files = list(
        Path(mask_folder).glob("*.png")
    )

    combined = None

    for mask_file in mask_files:
        mask = cv2.imread(
            str(mask_file),
            cv2.IMREAD_GRAYSCALE
        )

        if combined is None:
            combined = np.zeros_like(mask)

        _, thresh = cv2.threshold(
            mask,
            20,
            255,
            cv2.THRESH_BINARY
        )

        combined = cv2.bitwise_or(
            combined,
            thresh
        )

    return combined


def clean_background():
    image = cv2.imread(
        BACKGROUND_IMAGE
    )

    combined_mask = combine_masks(
        MASK_FOLDER
    )

    cleaned = cv2.inpaint(
        image,
        combined_mask,
        7,
        cv2.INPAINT_TELEA
    )

    cv2.imwrite(
        OUTPUT_PATH,
        cleaned
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    clean_background()