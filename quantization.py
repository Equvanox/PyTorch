import torch
import torch.nn as nn
from torchinfo import summary

model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

summary(model)

quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)
print(quantized_model)
# summary(quantized_model)