"""
This file has the Purpose of creating a Model class, to aid in Reusability.
"""
import torch
from torch import nn

class TinyVGG(nn.Module):
  """This is a model replicating the TinyVGG architechture of CNN Explainer website

     Args:
        inputs : An Integer to be fed into the input layer of model.
        hidden_layers : An Integer determing the number of hidden units on each layer of the model.
        output : An Integer representing the count of total classes/labels in problem.
  """
  def __init__(self,
               inputs: int,
               hidden_layers: int,
               output: int):
    super().__init__()
    self.convBlock1 = nn.Sequential(
        nn.Conv2d(in_channels= inputs,
                  out_channels= hidden_layers,
                  kernel_size= 3,
                  stride= 1,
                  padding= 1),
        nn.ReLU(),
        nn.Conv2d(in_channels= hidden_layers,
                  out_channels= hidden_layers,
                  kernel_size= 3,
                  stride= 1,
                  padding= 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,
                     stride=2)      # default stride value will be equal to kernel_size
    )
    self.convBlock2 = nn.Sequential(
        nn.Conv2d(in_channels= hidden_layers,
                  out_channels= hidden_layers,
                  kernel_size= 3,
                  stride= 1,
                  padding= 1),
        nn.ReLU(),
        nn.Conv2d(in_channels= hidden_layers,
                  out_channels= hidden_layers,
                  kernel_size= 3,
                  stride= 1,
                  padding= 1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size= 2,
                     stride= 2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features= hidden_layers * 16*16,
                  out_features= output)
    )


  def forward(self, X:torch.tensor) -> torch.tensor :
    # print(f"Shape of X before model {X.shape}")
    X = self.convBlock1(X)
    # print(f"Shape of X after ConvBlock1 {X.shape}")
    X = self.convBlock2(X)
    # print(f"Shape of X after ConvBlock2 {X.shape}")
    X = self.classifier(X)
    # print(f"Shape of X after Classifier {X.shape}")
    return X
