### Creating a Base Linear Model
## imports
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import dataset, dataloader 
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from timeit import default_timer as timer

## device agnostic code and randomness
torch.manual_seed(50)
torch.cuda.manual_seed(50)
device="cuda" if torch.cuda.is_available() else "cpu"


## create sample dataset(FashionMNIST)
train = datasets.FashionMNIST(root=r"D:\VisualStudioCode\Python\ML\pyTorch\ComputerVision\SampleData",
                              train=True,
                              transform=transforms.ToTensor(),
                              target_transform=None,
                              download=True)
test = datasets.FashionMNIST(root=r"D:\VisualStudioCode\Python\ML\pyTorch\ComputerVision\SampleData",
                             train=False,
                             transform=transforms.ToTensor(),
                             target_transform=None,
                             download=True)
print(len(train),len(test))
print(train.data, test.data)
print(train.classes)                # returns list of labels
print(train.class_to_idx)           # returns dictionary of labels
print(train.data.shape, test.data.shape)


## visualize data
plt.imshow(train.data.squeeze()[0])
fig = plt.figure(figsize=(7,7))
for i in range(1,17):
    index = torch.randint(0, len(train), size=[1]).item()
    fig.add_subplot(4,4,i)
    train_feature, train_label = train[index]
    plt.imshow(train.data.squeeze()[index], cmap="gray")
    plt.title(train.classes[train_label])
    plt.axis(False)
#plt.show()


## create Dataloader for train and test data
train_dataloader = dataloader.DataLoader(dataset=train,
                                         batch_size=32,
                                         shuffle=True)
test_dataloader = dataloader.DataLoader(dataset=test,
                                        batch_size=32,
                                        shuffle=False)
print(f"The Train Data of {len(train)} has been divided into {len(train_dataloader)} batches of {train_dataloader.batch_size} each.")
print(f"The Test Data of {len(test)} has been divided into {len(test_dataloader)} batches of {test_dataloader.batch_size} each.")
# shape of features and labels of train dataloader -> torch.Size([32, 1, 28, 28]) torch.Size([32])
                                                                # [N, C, W, B]
print(f"Train Feature Shape : {next(iter(train_dataloader))[0].shape} and Train Label Shape : {next(iter(train_dataloader))[1].shape}")
train_feature, train_label = next(iter(train_dataloader))


## creating a Linear model
# torch.Size([32, 1, 28, 28]) -> Flatten Layer -> torch.Size([32, 784])
class ComputerVisionV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=784, out_features=16),
            nn.Linear(in_features=16, out_features=10)              
            # out_features of last layer will be the probability of all possible labels, hence No. of labels(10)
       )
    
    def forward(self,X:torch.tensor) -> torch.tensor:
        return self.layers(X)

model = ComputerVisionV1().to(device)
model.eval()
with torch.inference_mode():
    y_logit = model(train_feature)
print(f"Model Predicted Logit Shape : {y_logit.shape}")


## Selecting Loss/Cost function, Optimizer, Evaluation Metric and time tracker
# Since this is a multiclass classification, we will use CrossEntropyLoss
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.01)
def accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    accuracy = correct/ len(y_pred) * 100
    return accuracy
def timeTrack(start: float,
              end: float):
    return end - start 


## creating training and testing loop
print("-----------------------------------------------------------------------------------------------------------------------------")
epochs = 3
training_loss_values = list()
testing_loss_values = list()
start_time = timer() 
for epoch in tqdm(range(epochs)):
    train_loss = 0
    train_acc = 0
    for batch, (X_train, y_train) in tqdm(enumerate(train_dataloader)):
        model.train()
        y_logit_train = model(X_train)
        training_loss = loss_fn(y_logit_train, y_train)
        train_loss += training_loss
        train_acc += accuracy(y_train, torch.softmax(y_logit_train, dim=1).argmax(dim=1))
        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()

    training_loss_avg = train_loss / len(train_dataloader)
    training_accuracy_avg = train_acc / len(train_dataloader)

    test_loss = 0
    test_acc = 0
    model.eval()
    with torch.inference_mode():
        for batch,(X_test, y_test) in tqdm(enumerate(test_dataloader)):
            y_logit_test = model(X_test)
            testing_loss = loss_fn(y_logit_test, y_test)
            test_loss += testing_loss
            test_acc += accuracy(y_test, torch.softmax(y_logit_test, dim=1).argmax(dim=1))

        testing_loss_avg = test_loss / len(test_dataloader)
        testing_accuracy_avg = test_acc / len(test_dataloader)

    if epoch % 1 == 0:
        print(f"Epoch {epoch}")
        print(f"Training Loss {training_loss_avg:.4f} , Testing Loss {testing_loss_avg:.4f}")
        print(f"Training Accuracy {training_accuracy_avg:.2f} , Testing Accuracy {testing_accuracy_avg:.2f}")
        training_loss_values.append(training_loss_avg.cpu().detach().numpy())
        testing_loss_values.append(testing_loss_avg.cpu().detach().numpy())
    print("-----------------------------------------------------------------------------------------------------------------------------")
end_time = timer()
print(f"Time Taken : {timeTrack(start_time, end_time):.2f} sec")
 
## save model and state_dict()
torch.save(model, r"D:\VisualStudioCode\Python\ML\pyTorch\ComputerVision\Saved_Models\CVLinearModel.pt")
torch.save(model.state_dict(), r"D:\VisualStudioCode\Python\ML\pyTorch\ComputerVision\Saved_Models\CVLinearModel_state_dict.pt")