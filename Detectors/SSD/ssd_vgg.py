import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

class SSDVGG(nn.Module):
    def __init__(self, num_classes=2):  # person + background
        super().__init__()
        self.num_classes = num_classes

        # --------------------------
        # 1. VGG Backbone
        # --------------------------
        self.vgg = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, ceil_mode=True),

            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 5
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1, padding=1)
        )

        # --------------------------
        # 2. SSD Extra Layers
        # --------------------------
        self.extra1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.extra2 = nn.Conv2d(1024, 512, 3, stride=2, padding=1)
        self.extra3 = nn.Conv2d(512, 256, 3, stride=2, padding=1)

        self.k = 6  # anchors per feature map

        # --------------------------
        # Localization heads
        # --------------------------
        self.loc = nn.ModuleList([
            nn.Conv2d(512,  self.k * 4, 3, padding=1),    # fmap 1
            nn.Conv2d(1024, self.k * 4, 3, padding=1),    # fmap 2
            nn.Conv2d(512,  self.k * 4, 3, padding=1),    # fmap 3
            nn.Conv2d(256,  self.k * 4, 3, padding=1)     # fmap 4
        ])

        # --------------------------
        # Classification heads
        # --------------------------
        self.conf = nn.ModuleList([
            nn.Conv2d(512,  self.k * num_classes, 3, padding=1),
            nn.Conv2d(1024, self.k * num_classes, 3, padding=1),
            nn.Conv2d(512,  self.k * num_classes, 3, padding=1),
            nn.Conv2d(256,  self.k * num_classes, 3, padding=1),
        ])

    def forward(self, x):
        locs = []
        confs = []

        # ----------- Feature Map 1 -----------
        x = self.vgg(x)
        locs.append(self.loc[0](x).permute(0, 2, 3, 1))
        confs.append(self.conf[0](x).permute(0, 2, 3, 1))

        # ----------- Feature Map 2 -----------
        x = self.extra1(x)
        locs.append(self.loc[1](x).permute(0, 2, 3, 1))
        confs.append(self.conf[1](x).permute(0, 2, 3, 1))

        # ----------- Feature Map 3 -----------
        x = self.extra2(x)
        locs.append(self.loc[2](x).permute(0, 2, 3, 1))
        confs.append(self.conf[2](x).permute(0, 2, 3, 1))

        # ----------- Feature Map 4 -----------
        x = self.extra3(x)
        locs.append(self.loc[3](x).permute(0, 2, 3, 1))
        confs.append(self.conf[3](x).permute(0, 2, 3, 1))

        # Flatten
        locs = torch.cat([o.reshape(o.size(0), -1, 4) for o in locs], 1)
        confs = torch.cat([o.reshape(o.size(0), -1, self.num_classes) for o in confs], 1)

        return locs, confs


model = SSDVGG(num_classes=2)
torchinfo.summary(model, input_size=(1, 3, 300, 300))

x = torch.randn(1, 3, 300, 300) # instead of random tensor, any image can also be passed after converting it to tensor
locs, confs = model(x)

print(locs.shape)   # [1, num_boxes, 4]
print(confs.shape)  # [1, num_boxes, 2]
