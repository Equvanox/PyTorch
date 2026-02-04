# Implementing R-CNN and Fast R-CNN from scratch is complex 
# because they involve multi-stage pipelines (Selective Search, CNN feature extraction, and SVMs/Regressors).
# Modern libraries like torchvision primarily provide Faster R-CNN (the industry standard)

import torch
import torchvision
from PIL import Image
import torchvision.transforms as T
import cv2
import numpy as np

# Load pretrained Faster R-CNN (ResNet50-FPN)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True)
model.eval()

# Preprocess
transform = T.Compose([T.ToTensor()])  # model expects [0,1] tensor
img = Image.open("D:\VisualStudioCode\Python\AI\DeepLearning\Pytorch\Detectors\FasterRCNN\Screenshot 2025-12-17 125515.png").convert("RGB")
img_tensor = transform(img)  # [C,H,W]
with torch.no_grad():
    preds = model([img_tensor])[0]   # list -> take first

# preds contains 'boxes','labels','scores'
boxes = preds["boxes"].cpu().numpy()
scores = preds["scores"].cpu().numpy()
labels = preds["labels"].cpu().numpy()

# Draw boxes with threshold
threshold = 0.7
img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
for box, score, label in zip(boxes, scores, labels):
    if score < threshold:
        continue
    x1,y1,x2,y2 = map(int, box)
    cv2.rectangle(img_cv, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(img_cv, f"{label}:{score:.2f}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

cv2.imwrite("fasterrcnn_out.jpg", img_cv)
print("Saved fasterrcnn_out.jpg")
