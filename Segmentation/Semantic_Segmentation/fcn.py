import torch
import torchvision.transforms as T
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 1. Load model with pre-trained weights
weights = FCN_ResNet50_Weights.DEFAULT
model = fcn_resnet50(weights=weights).eval()

# 2. Pre-processing pipeline
preprocess = weights.transforms()

# 3. Load and transform image
img = Image.open("city_street.jpg").convert("RGB")
input_tensor = preprocess(img).unsqueeze(0)

# 4. Inference
with torch.no_grad():
    output = model(input_tensor)['out'][0]

# 5. Convert output to class labels [H, W]
seg_map = torch.argmax(output, dim=0).cpu().numpy()

# 6. Create a color map for Pascal VOC (21 classes)
VOC_COLORS = np.array([
    [0,   0,   0],     # background
    [128, 0,   0],     # aeroplane
    [0,   128, 0],     # bicycle
    [128, 128, 0],     # bird
    [0,   0,   128],   # boat
    [128, 0,   128],   # bottle
    [0,   128, 128],   # bus
    [128, 128, 128],   # car
    [64,  0,   0],     # cat
    [192, 0,   0],     # chair
    [64,  128, 0],     # cow
    [192, 128, 0],     # diningtable
    [64,  0,   128],   # dog
    [192, 0,   128],   # horse
    [64,  128, 128],   # motorbike
    [192, 128, 128],   # person
    [0,   64,  0],     # potted plant
    [128, 64,  0],     # sheep
    [0,   192, 0],     # sofa
    [128, 192, 0],     # train
    [0,   64,  128],   # tv/monitor
])

# 7. Map class indices to colors
seg_color = VOC_COLORS[seg_map]

# 8. Display original image + segmentation
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(seg_color)
plt.title("Semantic Segmentation (FCN-ResNet50)")
plt.axis("off")

plt.tight_layout()
plt.show()
