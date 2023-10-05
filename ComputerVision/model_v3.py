### Creating a ConvNet Model
## Imports
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import dataloader
from torchmetrics import ConfusionMatrix
from timeit import default_timer as timer
from tqdm.auto import tqdm
from mlxtend.plotting import plot_confusion_matrix

## Random Seeds
random.seed(50)
torch.manual_seed(50)
torch.cuda.manual_seed(50)

## Device Agnostic Code
device= "cuda" if torch.cuda.is_available() else "cpu"

## Creating Sample Data(FashionMNIST)
train = datasets.FashionMNIST(root=r"D:\VisualStudioCode\Python\ML\pyTorch\ComputerVision\SampleData",
                              train=True,
                              transform=transforms.ToTensor(),
                              download=False)
test = datasets.FashionMNIST(root=r"D:\VisualStudioCode\Python\ML\pyTorch\ComputerVision\SampleData",
                             train=False,
                             transform=transforms.ToTensor(),
                             download=False)
print(train, test)
print(train.classes)
print(train.class_to_idx)


## Understanding Dataset
'''train and test are of 60000 and 10000 elements, of type(FashionMNIST)
from which each element can be accessed using index and can be splitted into image(X) and label(y)
hence train[78] -> element
      train[78][0] -> image or feature or X -> shape[1, 28, 28] -> [Color, Weight, Bias]
      train[78][1] -> label of y'''
print(train[78])
X,y = train[78]    # called destructuring | same as X= train[78][0] and y= train[78][1]
print(X.shape, y)  # cannot print X coz very large and cannot print y.shape coz y is of type Int and Int don't have .shape attribute


## Visualize Samples
plt.imshow(train[78][0].squeeze(), cmap="gray")  # since to plot a image it should be in [_,_] shape, hence _.squeeze()          
plt.title(train.classes[train[78][1]])
plt.axis(False)
#plt.show()
'''plotting random samples from train dataset'''
fig=plt.figure(figsize=(5,5))
for i in range(1, 21):
    random_idx = torch.randint(0, len(train), size=[1]).item()
    X,y = train[random_idx]
    fig.add_subplot(4,5,i)
    plt.imshow(X.squeeze(), cmap="gray")
    plt.title(train.classes[y])
    plt.axis(False)
#plt.show()


## Splitting into batches, Creating Dataloaders
train_dataloader = dataloader.DataLoader(dataset=train,
                                         batch_size=32,
                                         shuffle=True)
test_dataloader = dataloader.DataLoader(dataset=test,
                                        batch_size=32,
                                        shuffle=False)
print(f"The Train Dataset of Length {len(train)} has been splitted into {len(train_dataloader)} batches of Size {train_dataloader.batch_size} each.")
print(f"The Test Dataset of Length {len(test)} has been splitted into {len(test_dataloader)} batches of Size {test_dataloader.batch_size} each.")


## Create a ConvNet(CNN) model | architecture -> TinyVGG
class ComputerVisionV3(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_units,
                 output_size):
        super().__init__()
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                     out_channels=hidden_units,
                     kernel_size=3,
                     stride=1,
                     padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units *7*7,
                      out_features=output_size
                      )
        )

    def forward(self, X:torch.tensor) -> torch.tensor :
        #print(f"Shape of X : {X.shape}")
        X = self.convBlock1(X)
        #print(f"Shape of X after ConvBlock1 : {X.shape}")
        X = self.convBlock2(X)
        #print(f"Shape of X after ConvBlock2 : {X.shape}")
        X = self.classifier(X)
        #print(f"Shape of X after Classifier : {X.shape}")
        return X
        
model_v3 = ComputerVisionV3(input_size=1,
                         hidden_units=10,
                         output_size=len(train.classes)).to(device)
print(model_v3)
# print(model.parameters())
# print(model.state_dict())


## Find out the value of in_features in classifier layer, by experimenting
model_v3.eval()
with torch.inference_mode():
    y_logit = model_v3(X.unsqueeze(dim=0)).to(device)



## Select Loss/Cost Function and Optimizer also Create Accuracy fn and TimeTracker
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_v3.parameters(),
                            lr=0.1)
def accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    accuracy = correct/ len(y_pred) * 100
    return accuracy
timeTracker = lambda start, end : end - start

## create functized training and testing loop
def loopsFunc(epochs : int,
              model : nn.Module,
              train_dataloader : dataloader,
              test_dataloader : dataloader,
              loss_ftn : nn.Module,
              optimizer_ftn : torch.optim,
              accuracy_ftn):
    
    training_loss_values = list()
    testing_loss_values = list()
    start_time = timer()
    print("-----------------------------------------------------------------------------------------------------------------------------")
    for epoch in tqdm(range(epochs)):
        
        train_loss = 0
        train_acc = 0
        for train_batch, (X_train,y_train) in enumerate(train_dataloader):
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            model.train()
            y_logit_train = model(X_train).to(device)
            y_pred_train = torch.softmax(y_logit_train, dim=1).argmax(dim=1)
            
            loss_train = loss_ftn(y_logit_train, y_train)
            train_loss += loss_train
            acc = accuracy_ftn(y_train, y_pred_train)
            train_acc += acc

            optimizer_ftn.zero_grad()
            loss_train.backward()
            optimizer_ftn.step()
        training_loss_avg = train_loss/ len(train_dataloader)
        training_acc_avg = train_acc / len(train_dataloader)

        test_loss = 0
        test_acc = 0
        model.eval()
        with torch.inference_mode():
            for test_batch, (X_test,y_test) in enumerate(test_dataloader):
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                y_logit_test = model(X_test).to(device)
                y_pred_test = torch.softmax(y_logit_test, dim=1).argmax(dim=1)

                loss_test = loss_ftn(y_logit_test, y_test)
                test_loss += loss_test
                acc = accuracy_ftn(y_test, y_pred_test)
                test_acc += acc
            testing_loss_avg = test_loss / len(test_dataloader)
            testing_acc_avg = test_acc / len(test_dataloader)

        if epoch % 1 == 0:
            print(f"\nEpoch : {epoch}")
            print(f"Training Batches Processed : {train_batch+1}")
            print(f"Testing Batches Processed : {test_batch+1}")
            print(f"Training Loss {training_loss_avg:.4f} , Testing Loss {testing_loss_avg:.4f}")
            print(f"Training Accuracy {training_acc_avg:.2f} , Testing Accuracy {testing_acc_avg:.2f}")
            training_loss_values.append(training_loss_avg.cpu().detach().numpy())
            testing_loss_values.append(testing_loss_avg.cpu().detach().numpy())
        print("-----------------------------------------------------------------------------------------------------------------------------")
    end_time = timer()
    print(f"The Time taken for Training is {timeTracker(start_time, end_time):.2f} sec. on Device {str(next(model.parameters()).device)}")
    return timeTracker(start_time, end_time), training_loss_values, testing_loss_values


## Calling the funtionized loop for model    
model_v3_time, training_loss_values, testing_loss_values = loopsFunc(epochs=3,
          model=model_v3,
          train_dataloader=train_dataloader,
          test_dataloader=test_dataloader,
          loss_ftn=loss_fn,
          optimizer_ftn=optimizer,
          accuracy_ftn=accuracy)


## creating prediction function
def prediction(model,
               dataloader,
               loss_ftn,
               accuracy_ftn):
    loss = 0
    acc = 0 
    model.eval()
    with torch.inference_mode():
        for X,y in dataloader:
            X = X.to(device)
            y = y.to(device)
            y_logits = model(X).to(device)
            y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

            loss_val = loss_ftn(y_logits, y)
            loss += loss_val
            accuracy = accuracy_ftn(y, y_pred)
            acc += accuracy

        loss_avg = loss/ len(dataloader)
        acc_avg = acc / len(dataloader)    
    
    return {"Model":model.__class__.__name__,
            "Loss": loss_avg.item(),
            "Accuracy":acc_avg}

## Making prediction of model
print(prediction(model=model_v3,
                 dataloader=test_dataloader,
                 loss_ftn=loss_fn,
                 accuracy_ftn=accuracy))


## Predict some random sample from original "test" FashionMNIST dataset and plot it
random_sample_idx = random.randint(0, len(test))
X, y = test[random_sample_idx]
#print(X.shape, y)
model_v3.eval()
with torch.inference_mode():
    y_logit = model_v3(X.unsqueeze(dim=0)).to(device)
    y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
#print(y , y_pred)
plt.figure()
plt.imshow(X.squeeze(), cmap="gray")
plt.title(f"True : {test.classes[y]} | Pred : {test.classes[y_pred]}")
plt.axis(False)
#plt.show()


## Make predictions on a random set of samples from "test_dataloader" and plot them
preds = list()
model_v3.eval()
with torch.inference_mode():
    for batch, (X,y) in enumerate(test_dataloader):
        X = X.to(device)
        y = y.to(device)
        y_logit = model_v3(X).to(device)
        y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)

        preds.append(y_pred)

y_preds = torch.Tensor(torch.concat(tensors=preds))
#print(y_preds, len(y_preds))
fig=plt.figure(figsize=(12,12))
for i in range(1,21):
    random_idx = torch.randint(0, len(test), size=[1]).item()
    image, label = test[random_idx]
    pred_label = y_preds[random_idx]
    image = image.cpu().detach().numpy()
    fig.add_subplot(4,5,i)
    plt.imshow(image.squeeze(), cmap="gray")
    if pred_label == label:
        plt.title(f"Label : {test.classes[label]} | Pred : {test.classes[pred_label]}", fontsize=10, c="g")
    else:
        plt.title(f"Label : {test.classes[label]} | Pred : {test.classes[pred_label]}", fontsize=10, c="r")
    plt.axis(False)
#plt.show()


## plotting loss curves
plt.figure()
plt.plot(training_loss_values, label="Training Loss")
plt.plot(testing_loss_values, label="Testing Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
#plt.show()


## plotting another Evaluation Metric(Confusion Matrix)
confmat = ConfusionMatrix(task="multiclass", num_classes=len(test.classes))
confmat_tensor = confmat(preds= y_preds,
                         target= test.targets)
fig, ax = plot_confusion_matrix(conf_mat=confmat_tensor.cpu().detach().numpy(),
                      figsize=(8,8),
                      class_names=test.classes)
plt.show()


## Save Model and state_dict()
torch.save(model_v3, r"D:\VisualStudioCode\Python\ML\pyTorch\ComputerVision\Saved_Models\CVConvNetModel.pt")
torch.save(model_v3.state_dict(), r"D:\VisualStudioCode\Python\ML\pyTorch\ComputerVision\Saved_Models\CVConvNetModel_state_dict.pt")