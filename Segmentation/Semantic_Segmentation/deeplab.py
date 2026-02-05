# pip install segmentation-models-pytorch
import segmentation_models_pytorch as smp
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T

# 1. Create the DeepLabV3+ model
model = smp.DeepLabV3Plus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=19,   # Cityscapes-style
)
model.eval()

# 2. Image preprocessing (ImageNet normalization)
preprocess = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])

# 3. Load image
img = Image.open("city_street.jpg").convert("RGB")
input_tensor = preprocess(img).unsqueeze(0)

# 4. Inference
with torch.no_grad():
    output = model(input_tensor)      # [1, 19, H, W]
    seg_map = torch.argmax(output, dim=1)[0].cpu().numpy()  # [H, W]

# 5. Create a color palette (19 classes)
COLORS = np.array([
    [128, 64, 128],   # road
    [244, 35, 232],   # sidewalk
    [70, 70, 70],     # building
    [102, 102, 156],  # wall
    [190, 153, 153],  # fence
    [153, 153, 153],  # pole
    [250, 170, 30],   # traffic light
    [220, 220, 0],    # traffic sign
    [107, 142, 35],   # vegetation
    [152, 251, 152],  # terrain
    [70, 130, 180],   # sky
    [220, 20, 60],    # person
    [255, 0, 0],      # rider
    [0, 0, 142],      # car
    [0, 0, 70],       # truck
    [0, 60, 100],     # bus
    [0, 80, 100],     # train
    [0, 0, 230],      # motorcycle
    [119, 11, 32],    # bicycle
])

# 6. Map classes → colors
seg_color = COLORS[seg_map]

# 7. Display
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(seg_color)
plt.title("DeepLabV3+ Segmentation")
plt.axis("off")

plt.tight_layout()
plt.show()
