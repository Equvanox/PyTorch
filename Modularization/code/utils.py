"""
This file contains all the helper functions, utilities and extra functions
"""
import torch
from torch import nn
from pathlib import Path

def saving_model(model: nn.Module,
                 dirPath: str,
                 model_name: str):
  """Saves model or astate_dict into a directory

  This function will save the model or it's state dictionary in
  the provided path.

  Args: 
    model : An Neural Network Class, working as a nn Model.
    dirPath : (str)A path to the directory in which the model has to be saved.
    model_name : (str)A name by which the model will be saved.
  Returns:
    A file in models/ directory with .pt or .pth extension.
  """
  dirPath = Path(dirPath)
  # if dirPath.is_dir():
  #   print("Directory Already exists, Saving model..")
  # else:
  #   print("Directory is Missing, Creating now..")
  #   Path.mkdir(dirPath)
  dirPath.mkdir(parents=True,
                exist_ok=True)
  
  assert model_name.endswith(".pt") or model_name.endswith(".pth"), "The Model Path must end with proper file extension as either (.pt) or (.pth)"
  modelPath = dirPath/model_name

  torch.save(obj=model,f=modelPath)
  
