"""
This file contains the function to predict images on a provided model.
"""
import torch
import torchvision
from torch import nn
from PIL import Image
from torchvision import transforms
from typing import List, Tuple
from matplotlib import pyplot as plt

device="cuda" if torch.cuda.is_available() else "cpu"

def predict_a_image(path:str,
                model:nn.Module,
                class_names:List[str],
                image_size:Tuple[int]=(224,224),
                transform:torchvision.transforms=None,
                device=device):
    img = Image.open(path)
    if not transform:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
    transformed_img = transform(img).to(device)
    model.eval()
    with torch.inference_mode():
        y_logits = model(transformed_img.unsqueeze(dim=0))
        y_pred_probs = torch.softmax(y_logits, dim=1)
        y_pred_labels = y_pred_probs.argmax(dim=1)

    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred {class_names[y_pred_labels]}, Prob {y_pred_probs.max():.3f}")
    plt.axis(False)
    plt.show()