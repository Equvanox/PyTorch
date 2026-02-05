from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np

# 1. Setup Model
sam_checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# 2. Load Image
image = cv2.imread('images.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
predictor.set_image(image)

# -----------------------------
# Interactive state
# -----------------------------
points = []
labels = []

def draw_overlay(img, mask):
    overlay = img.copy()
    overlay[mask] = (
        overlay[mask] * 0.4 + np.array([0, 255, 0]) * 0.6
    ).astype(np.uint8)
    return overlay

def mouse_callback(event, x, y, flags, param):
    global points, labels

    # 🔥 RESET segmentation on new foreground click
    if event == cv2.EVENT_LBUTTONDOWN:
        points = [[x, y]]
        labels = [1]

    # Background click refines current object only
    elif event == cv2.EVENT_RBUTTONDOWN and len(points) > 0:
        points.append([x, y])
        labels.append(0)

    else:
        return

    masks, scores, _ = predictor.predict(
        point_coords=np.array(points),
        point_labels=np.array(labels),
        multimask_output=False,
    )

    mask = masks[0]
    display = draw_overlay(image, mask)

    # Draw current points
    for (px, py), label in zip(points, labels):
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        cv2.circle(display, (px, py), 6, color, -1)

    cv2.imshow("SAM Live Segmentation", cv2.cvtColor(display, cv2.COLOR_RGB2BGR))

# -----------------------------
# Open window
# -----------------------------
cv2.namedWindow("SAM Live Segmentation")
cv2.setMouseCallback("SAM Live Segmentation", mouse_callback)

print("Left click = NEW object | Right click = refine | ESC = exit")

cv2.imshow("SAM Live Segmentation", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

while True:
    if cv2.waitKey(1) == 27:  # ESC
        break

cv2.destroyAllWindows()
