### Transfer Learning for the ComputerVision-CustomDataset problem
## Imports
import torch
import random
import matplotlib.pyplot as plt
from torch import nn
from pathlib import Path
from PIL import Image
from torchvision import datasets, transforms, models
from Modularization.code import data_setup, engine
from .SampleData import CustomFood101
from torch.utils.data import dataloader


## Device Agnostic code and random seeds
device= "cuda" if torch.cuda.is_available() else "cpu"
random.seed(50)
torch.manual_seed(50)
torch.cuda.manual_seed(50)


## Creating transforms in 2 different ways 
# MANUALLY -> since we decided to use EfficientNet_B0 model, we have to take care of the data format that model accepts
# size atleast 224, values b/w 0-1, normalization should be mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
manual_transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# AUTOMATICALLY -> copying the pretrained model's transform
weights = models.EfficientNet_B0_Weights.DEFAULT
automatic_transform = weights.transforms()

print(manual_transform, automatic_transform)