import cv2

# load original image
image = cv2.imread(
    "editable-editor/public/layers/background.png"
)

# load mask
mask = cv2.imread(
    "masks/mask_1.png",
    0
)

# inpaint deleted area
result = cv2.inpaint(
    image,
    mask,
    7,
    cv2.INPAINT_TELEA
)

# save output
cv2.imwrite(
    "outputs/result.png",
    result
)

print("Done!")