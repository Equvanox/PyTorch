# SSD = Single Shot Multibox Detector

import torch, torchinfo
import torch.nn as nn
import torch.nn.functional as F

class SimpleSSD(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # Base CNN
        self.base = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )

        # Additional feature maps
        self.extra1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.extra2 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        # Anchor count per feature map
        self.k = 6

        # Localization prediction
        self.loc = nn.ModuleList([
            nn.Conv2d(64,  self.k * 4, 3, padding=1),
            nn.Conv2d(128, self.k * 4, 3, padding=1),
            nn.Conv2d(256, self.k * 4, 3, padding=1),
        ])

        # Class prediction
        self.conf = nn.ModuleList([
            nn.Conv2d(64,  self.k * num_classes, 3, padding=1),
            nn.Conv2d(128, self.k * num_classes, 3, padding=1),
            nn.Conv2d(256, self.k * num_classes, 3, padding=1),
        ])

    def forward(self, x):
        locs = []
        confs = []

        # Feature map 1
        x = self.base(x)
        locs.append(self.loc[0](x).permute(0, 2, 3, 1).contiguous())
        confs.append(self.conf[0](x).permute(0, 2, 3, 1).contiguous())

        # Feature map 2
        x = self.extra1(x)
        locs.append(self.loc[1](x).permute(0, 2, 3, 1).contiguous())
        confs.append(self.conf[1](x).permute(0, 2, 3, 1).contiguous())

        # Feature map 3
        x = self.extra2(x)
        locs.append(self.loc[2](x).permute(0, 2, 3, 1).contiguous())
        confs.append(self.conf[2](x).permute(0, 2, 3, 1).contiguous())

        # Flatten
        locs = torch.cat([o.view(o.size(0), -1, 4) for o in locs], dim=1)
        confs = torch.cat([o.view(o.size(0), -1, self.num_classes) for o in confs], dim=1)

        return locs, confs


## Dummy Test
model = SimpleSSD(num_classes=21)   # VOC dataset
torchinfo.summary(model=model)
# print(model.parameters())
x = torch.randn(1, 3, 300, 300)
locs, confs = model(x)

print(locs.shape)   # [batch, num_boxes, 4]
print(confs.shape)  # [batch, num_boxes, num_classes]
