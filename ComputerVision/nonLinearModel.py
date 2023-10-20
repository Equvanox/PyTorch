### Creating a Non Linear Model
## imports
import torch 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn import datasets, model_selection
from torch import nn    
from torchvision import datasets, transforms
from torch.utils.data import dataloader, dataset
from timeit import default_timer as timer
from tqdm.cli import tqdm


## device agnostic code and random seeds
torch.manual_seed(50)
torch.cuda.manual_seed(50)
device= "cuda" if torch.cuda.is_available() else "cpu"


## sample datasets
train = datasets.FashionMNIST(root=r"D:\VisualStudioCode\Python\ML\pyTorch\ComputerVision\SampleData",
                              train=True,
                              transform=transforms.ToTensor(),
                              download=False)
test = datasets.FashionMNIST(root=r"D:\VisualStudioCode\Python\ML\pyTorch\ComputerVision\SampleData",
                             train=False,
                             transform=transforms.ToTensor(),
                             download=False)
print(train, test)
print(len(train), len(test))
print(train.data.shape, test.data.shape)
print(train.classes)
print(train.class_to_idx)


## visualize the datasets
print(train[7])
plt.imshow(train.data[7])
# visualizing loop
no_of_pics = 20
fig = plt.figure(figsize=(7,7))
for i in range(1, no_of_pics+1):
    random_idx = torch.randint(0, len(train.data), size=[1])
    # print(random_idx, random_idx.item())
    random_idx = random_idx.item()
    X_train, y_train = train[random_idx]
    fig.add_subplot(4,5,i)
    plt.title(train.classes[y_train])
    plt.imshow(X_train.squeeze(), cmap="gray")
    plt.axis(False)
#plt.show()


## splitting the dataset into batches by creating dataloaders 
train_dataloader = dataloader.DataLoader(dataset=train,
                                         batch_size=32,
                                         shuffle=True)
test_dataloader = dataloader.DataLoader(dataset=test,
                                        batch_size=32,
                                        shuffle=False)
print(train_dataloader, test_dataloader)
# train_dataloader is the collection of 1875 batches, and test_dataloader is collection of 313 batches
print(len(train_dataloader), len(test_dataloader))

# next(iter) gives 1st batch out of 1875, batch length will be 32
print(next(iter(train_dataloader)))

# 1 batch will have 32 elements, every element will have a 2 values, a set of features(x) and a label(y)
train_features, train_labels = next(iter(train_dataloader))
# print(train_features)
# print(train_labels)
print(train_features.shape, train_labels.shape)

print(f"Train Dataset of length {len(train)} has been splitted into {len(train_dataloader)} batches of size {train_dataloader.batch_size} each.")
print(f"Test Dataset of length {len(test)} has been splitted into {len(test_dataloader)} batches of size {test_dataloader.batch_size} each.")


## create a non linear model
class ComputerVisionV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=28*28, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=10)
        )

    def forward(self, X:torch.tensor) -> torch.tensor :
        return self.layers(X)
    
model = ComputerVisionV2().to(device)
print(model)
print(model.state_dict())
model.eval()
with torch.inference_mode():
    y_logits = model(train_features)
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
print(y_pred, train_labels)


## selecting Loss/Cost function, Optimizer, Evaluation Metric and creating Time tracker
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.01)
def accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    return correct/ len(y_pred) * 100
def trackTime(start:float, end:float):
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
    for batch,(X_train, y_train) in enumerate(train_dataloader):
        model.train()
        y_logits_train = model(X_train)
        y_pred_train = torch.softmax(y_logits_train, dim=1).argmax(dim=1)
        loss = loss_fn(y_logits_train, y_train)
        train_loss += loss
        acc = accuracy(y_train, y_pred_train)
        train_acc += acc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    training_loss_avg = train_loss/ len(train_dataloader)
    training_acc_avg = train_acc/ len(train_dataloader)

    test_loss = 0
    test_acc = 0
    model.eval()
    with torch.inference_mode():
        for batch, (X_test, y_test) in tqdm(enumerate(test_dataloader)):
            y_logits_test = model(X_test)
            y_pred_test = torch.softmax(y_logits_test, dim=1).argmax(dim=1)
            loss = loss_fn(y_logits_test, y_test)
            test_loss += loss
            acc = accuracy(y_test, y_pred_test)
            test_acc += acc

            testing_loss_avg = test_loss/ len(test_dataloader)
            testing_acc_avg = test_acc / len(test_dataloader)

    if epoch % 1 == 0:
        print(f"Epoch {epoch}")
        print(f"Training Loss {training_loss_avg:.4f} , Testing Loss {testing_loss_avg:.4f}")
        print(f"Training Accuracy {training_acc_avg:.2f} , Testing Accuracy {testing_acc_avg:.2f}")
        training_loss_values.append(training_loss_avg.cpu().detach().numpy())
        testing_loss_values.append(testing_loss_avg.cpu().detach().numpy()) 
    print("-----------------------------------------------------------------------------------------------------------------------------")

end_time = timer()
print(f"Time Taken : {trackTime(start_time, end_time):.2f} sec")

## save model and state_dict()
torch.save(model, r"D:\VisualStudioCode\Python\ML\pyTorch\ComputerVision\Saved_Models\CVNonLinearModel.pt")
torch.save(model.state_dict(), r"D:\VisualStudioCode\Python\ML\pyTorch\ComputerVision\Saved_Models\CVNonLinearModel_state_dict.pt")