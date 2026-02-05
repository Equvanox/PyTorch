import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# 1. Setup Model (Ensure paths to .pt and .yaml are correct)
checkpoint = "sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml" # Note: config should match your checkpoint size
sam2_model = build_sam2(model_cfg, checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
predictor = SAM2ImagePredictor(sam2_model)

# 2. Load and Set Image
image_path = "images.jpg"
image = Image.open(image_path)
image_np = np.array(image)
predictor.set_image(image_np)

# 3. Define Prompts
input_points = np.array([[500, 375], [600, 400]])
input_labels = np.array([1, 0]) # 1 = foreground, 0 = background

# 4. Predict
masks, scores, _ = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True # Returns 3 variations (e.g., "just the arm" vs "whole person")
)

# 5. DISPLAY RESULTS
def show_mask(mask, ax, random_color=False):
    color = np.concatenate([np.random.random(3), [0.6]], axis=0) if random_color else np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=200, edgecolor='white')
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=200, edgecolor='white')

# Create a grid to see all 3 mask suggestions
plt.figure(figsize=(15, 5))
for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.subplot(1, 3, i+1)
    plt.imshow(image_np)
    show_mask(mask, plt.gca())
    show_points(input_points, input_labels, plt.gca())
    plt.title(f"Mask {i+1} (Score: {score:.2f})")
    plt.axis('off')

plt.tight_layout()
plt.show()