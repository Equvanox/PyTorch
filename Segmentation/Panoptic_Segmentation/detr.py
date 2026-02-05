import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
import io
import torchvision.transforms as T

# 1. Load Model
model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', 
                                     pretrained=True, return_postprocessor=True)
model.eval()

# 2. Prep Image
# url = "http://images.cocodataset.org/val2017/000000281759.jpg"
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
image = Image.open("images.jpg")
img_tensor = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])(image).unsqueeze(0)

# 3. Inference
with torch.no_grad():
    out = model(img_tensor)
result = postprocessor(out, torch.as_tensor(img_tensor.shape[-2:]).unsqueeze(0))[0]

# 4. Convert the PNG string to a Segment ID Map
# The 'png_string' is a special encoded format where RGB values represent IDs
panoptic_seg = Image.open(io.BytesIO(result['png_string']))
panoptic_seg_array = np.array(panoptic_seg, dtype=np.uint8)

# Calculate the actual ID for each pixel: ID = R + G*256 + B*256^2
# This is why your image looked black! The values were encoded.
id_map = (panoptic_seg_array[:, :, 0].astype(np.uint32) + 
          panoptic_seg_array[:, :, 1].astype(np.uint32) * 256 + 
          panoptic_seg_array[:, :, 2].astype(np.uint32) * 256 * 256)

# 5. GENERATE COLORFUL OVERLAY
# We create a random color for every unique ID found in the image
unique_ids = np.unique(id_map)
rgb_mask = np.zeros((*id_map.shape, 3), dtype=np.uint8)

for val in unique_ids:
    if val == 0: continue # Skip background
    rgb_mask[id_map == val] = np.random.randint(0, 255, size=3)

# 6. Display Side-by-Side
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(rgb_mask)
plt.title("Panoptic Segments (Colorized)")
plt.axis('off')

plt.show()