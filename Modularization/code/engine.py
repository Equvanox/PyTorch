"""
This File contains the Functionized loops for training, evaluating and full training-evaluating an ML model
"""
import torch
from torch import nn 
from tqdm.auto import tqdm
from typing import Tuple, Dict
from torch.utils.data import dataloader
from timeit import default_timer as timer

device= "cuda" if torch.cuda.is_available() else "cpu" 

def train_loop(model: nn.Module,
               train_dataloader: dataloader.DataLoader,
               loss_ftn: nn.Module,
               optimizer_ftn: torch.optim.Optimizer,
               accuracy_ftn,
               device=device) -> Tuple[float,float]:
  """Loops the Dataloader to train model
  
  Executes Looping structure to train model using loss functions and optimizer, Also calculates
  Accuracy and training loss.

  Args: 
    model : An Neural Network Class, working as a nn Model
    dataloader : A batched version of dataset for the model to be trained upon.
    loss_ftn : The Loss/Cost function to be used in Back Propogation.
    optimizer_ftn : An Optimizer to be used in Gradient Decent Algorithm.
    accuracy_ftn : An torchmetrics/custom function to calculate bias and hence accuracy of model.
    device : A string static parameter populated with the device name on which the experiment is being executed. 

  Returns:
    A tuple of (training_loss_avg, training_acc_avg)
    where training_loss_avg is the average loss during training and 
    training_acc_avg is the average accuracy during training. 
  """
  model.train()
  train_loss = 0
  train_acc = 0
  for batch, (X_train, y_train) in enumerate(train_dataloader):
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    y_logits = model(X_train)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    loss = loss_ftn(y_logits, y_train)
    train_loss += loss
    acc = accuracy_ftn(y_train, y_pred)
    train_acc += acc
    optimizer_ftn.zero_grad()
    loss.backward()
    optimizer_ftn.step()

  training_loss_avg = train_loss/ len(train_dataloader)
  training_acc_avg = train_acc/ len(train_dataloader)
  #print(f"Batches Trained {batch+1}")
  return training_loss_avg, training_acc_avg

def test_loop(model: nn.Module,
              test_dataloader: dataloader.DataLoader,
              loss_ftn: nn.Module,
              accuracy_ftn,
              device=device) -> Tuple[float,float]:
  """Loops the Dataloader to evaluate model
  
  Executes Looping structure to test/validate model using loss function, Also calculates
  Accuracy and testing loss.

  Args: 
    model : An Neural Network Class, working as a nn Model
    dataloader : A batched version of dataset for the model to be evaluated upon.
    loss_ftn : The Loss/Cost function to be used in Back Propogation.
    accuracy_ftn : An torchmetrics/custom function to calculate bias and hence accuracy of model.
    device : A string static parameter populated with the device name on which the experiment is being executed. 

  Returns:
    A tuple of (tesing_loss_avg, tesing_acc_avg)
    where tesing_loss_avg is the average loss during tesing and 
    tesing_acc_avg is the average accuracy during tesing/validation. 
  """
  model.eval()
  test_loss = 0
  test_acc = 0
  with torch.inference_mode():
    for batch, (X_test, y_test) in enumerate(test_dataloader):
      X_test = X_test.to(device)
      y_test = y_test.to(device)
      y_logits = model(X_test)
      y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
      loss = loss_ftn(y_logits, y_test)
      test_loss += loss
      acc = accuracy_ftn(y_test, y_pred)
      test_acc += acc

    testing_loss_avg = test_loss/ len(test_dataloader)
    testing_acc_avg = test_acc/ len(test_dataloader)
    #print(f"Batches Evaluated {batch+1}")
  return testing_loss_avg, testing_acc_avg

def full_loop(epochs: int,
              model: nn.Module,
              train_dataloader: dataloader.DataLoader,
              test_dataloader: dataloader.DataLoader,
              loss_fn: nn.Module,
              optimizer: torch.optim.Optimizer,
              accuracy,
              device=device) -> Tuple[Dict[str,float]]:
  """Encapsulates Training and Testing loops to run complete simulation
  
  Incases the independent train_loop() and test_loop() functions to create a complete
  functionality of iterable training and testing/validation and returns a dictionary 
  of training and testing, loss and accuracies.
  
  Args:
    epochs : An Integer to represent how many iterations must be executed.
    model : An Neural Network Class, working as a nn Model
    train_dataloader : A batched version of dataset for the model to be trained upon.
    test_dataloader : A batched version of dataset for the model to be evaluated upon.
    loss_ftn : The Loss/Cost function to be used in Back Propogation.
    optimizer_ftn : An Optimizer to be used in Gradient Decent Algorithm.
    accuracy_ftn : An torchmetrics/custom function to calculate bias and hence accuracy of model.
    device : A static string parameter populated with the device name on which the experiment is being executed.

  Returns:
    A Dictionary of collective loss and accuracy,
    for each iteration of training and testing.
    """
  epochs = epochs
  results = {"training_loss":[],
             "training_accuracy":[],
             "testing_loss":[],
             "testing_accuracy":[]
             }
  start_time = timer()
  for epoch in tqdm(range(epochs)):
    print("\n----------------------------------------------------------------------------------------------------------------------")
    train_loss, train_acc = train_loop(model=model,
                                       train_dataloader=train_dataloader,
                                       loss_ftn=loss_fn,
                                       optimizer_ftn=optimizer,
                                       accuracy_ftn=accuracy)
    
    test_loss, test_acc = test_loop(model=model,
                                    test_dataloader=test_dataloader,
                                    loss_ftn=loss_fn,
                                    accuracy_ftn=accuracy)
    print(f"Epoch - {epoch}")
    print(f"Training Loss {train_loss:.4f} , Training Accuracy {train_acc:.2f}")
    print(f"Testing Loss {test_loss:.4f} , Testing Accuracy {test_acc:.2f}")
    results["training_loss"].append(train_loss.cpu().detach().numpy().item())
    results["training_accuracy"].append(train_acc)
    results["testing_loss"].append(test_loss.cpu().detach().numpy().item())
    results["testing_accuracy"].append(test_acc)
  print("----------------------------------------------------------------------------------------------------------------------")
  end_time = timer()
  print(f"Time taken by the experiment is {(end_time-start_time):.2f} sec, on device {device}")
  return results
