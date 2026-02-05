import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from PIL import Image
from torchvision.transforms import functional as F

# 1. Load pre-trained model
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
model.eval()

# 2. Load and prep image
img = Image.open("images.jpg").convert("RGB")
img_tensor = F.to_tensor(img).unsqueeze(0)

# 3. Run inference
with torch.no_grad():
    prediction = model(img_tensor)

# 4. Filter predictions
score_threshold = 0.8
scores = prediction[0]["scores"]
masks = prediction[0]["masks"][scores > score_threshold]  # [N, 1, H, W]

print(f"Detected {len(masks)} objects with >80% confidence.")

# 5. Convert image to numpy
img_np = np.array(img)

# 6. Overlay masks
overlay = img_np.copy()
alpha = 0.5  # transparency

for mask in masks:
    mask = mask.squeeze(0).cpu().numpy()
    mask = mask > 0.5  # binarize

    # random color per instance
    color = np.random.randint(0, 255, size=3)

    overlay[mask] = (
        overlay[mask] * (1 - alpha) + color * alpha
    ).astype(np.uint8)

# 7. Display result
plt.figure(figsize=(10, 10))
plt.imshow(overlay)
plt.axis("off")
plt.title("Segmented Image")
plt.show()
