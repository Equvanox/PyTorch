"""
The Purpose of this file is to create dataset out of the raw data and further convert it into dataloaders for processing.
"""
import os
import torchvision
from torchvision import datasets
from torch.utils.data import dataloader

def convert_data_to_datasets(trainPath: str,
                             testPath: str,
                             batch_size: int,
                             transform: torchvision.transforms.Compose):

  """Creates training and testing dataloaders

  This Method will take training and testing paths as inputs and
  will return the datasets and dataloaders created from the raw data in files.

  Args:
    trainPath : the path where training data lies.
    testPath : the path where testing data lies.
    batch_size : the size in which the datasets are to be splitted into.
    transform : the transformation to be applied on thr data.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names)
    where the class_names will be a list of target labels.
  """

  train_dataset = datasets.ImageFolder(root=trainPath,
                                       transform=transform,
                                       target_transform=None)
  test_dataset = datasets.ImageFolder(root=testPath,
                                      transform=transform)

  class_names = train_dataset.classes

  train_dataloader = dataloader.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=1)
  test_dataloader = dataloader.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=1)
  return train_dataloader, test_dataloader, class_names
