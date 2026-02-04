# Implementing R-CNN and Fast R-CNN from scratch is complex 
# because they involve multi-stage pipelines (Selective Search, CNN feature extraction, and SVMs/Regressors).
# Modern libraries like torchvision primarily provide Faster R-CNN (the industry standard)

import torch
import torch.nn as nn
import torchvision
from torchvision.ops import roi_align
from PIL import Image

# -----------------------------
# Backbone (shared conv)
# -----------------------------
backbone = torchvision.models.resnet50(pretrained=True)
modules = list(backbone.children())[:-2]  # remove avgpool & fc
backbone = nn.Sequential(*modules)
backbone.eval()

# -----------------------------
# Fast R-CNN Heads
# -----------------------------
class FastRCNNHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(2048 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        self.cls_score = nn.Linear(1024, num_classes)      # classification
        self.bbox_pred = nn.Linear(1024, num_classes * 4) # bbox regression

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


head = FastRCNNHead(num_classes=2)  # background + person
head.eval()

# -----------------------------
# Load Image
# -----------------------------
img = Image.open("D:\VisualStudioCode\Python\AI\DeepLearning\Pytorch\Detectors\FastRCNN\Screenshot 2025-12-17 125515.png").convert("RGB")
img_t = torchvision.transforms.ToTensor()(img).unsqueeze(0)

# -----------------------------
# Proposals (candidate boxes)
# -----------------------------
proposals = torch.tensor([
    [0, 50, 50, 200, 200],
    [0, 120, 80, 260, 220],
], dtype=torch.float)

# -----------------------------
# Forward Pass
# -----------------------------
with torch.no_grad():
    feat_map = backbone(img_t)

    roi_feats = roi_align(
        feat_map,
        proposals,
        output_size=(7, 7),
        spatial_scale=feat_map.shape[-1] / img_t.shape[-1]
    )

    class_scores, bbox_deltas = head(roi_feats)

print("RoI features:", roi_feats.shape)
print("Class scores:", class_scores.shape)
print("BBox deltas:", bbox_deltas.shape)
