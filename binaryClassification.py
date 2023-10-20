### creating a pytorch model to train over Binary Classification 
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import nn  
from sklearn import datasets, model_selection

## device agnostic code
device="cuda" if torch.cuda.is_available() else "cpu"

## random seed
torch.manual_seed(50)
torch.cuda.manual_seed(50)

## create sample data
X, y = datasets.make_circles(n_samples=1000,
                             noise=0.03,
                             random_state=50)

## pandas dataframe
df = pd.DataFrame({"X":X[:,0],
                    "y":X[:,1],
                    "Label":y})
print(df)


## plot using matplotlib
def plotting2D(X1,X2,title,c):
    plt.figure(figsize=(6,6))
    plt.title(title)
    plt.scatter(x=X1,y=X2,s=3,c=c,cmap="winter")
    plt.xlabel("Feature(X1)")
    plt.ylabel("Feature(X2)")

def plotting3D(X1,X2,y,title):
    plt.figure(figsize=(8,6))
    plt.title(title)
    ax = plt.axes(projection="3d")
    ax.scatter3D(xs=X1,ys=X2,zs=y,s=3,c=y,cmap="winter")
    ax.set_xlabel("Feature(X1)")
    ax.set_ylabel("Feature(X2)")
    ax.set_zlabel("Label(y)")

plotting2D(X[:,0],X[:,1],"2D Data",y)
plotting3D(X[:,0],X[:,1],y,"3D Data")
# plt.show()


## convert numpy to pytorch
print(isinstance(X,np.ndarray), isinstance(y,np.ndarray))             # true if numpy array
print(X.dtype, y.dtype)
X = torch.from_numpy(X).type(torch.float).to(device)
y = torch.from_numpy(y).type(torch.float).to(device)
print(torch.is_tensor(X), torch.is_tensor(y))             # true if pytorch tensor
print(X.dtype, y.dtype)


## split into training and testing datasets using sklearn
X_training_dataset, X_testing_dataset = model_selection.train_test_split(X,
                                                                         test_size=0.2,
                                                                         train_size=0.8,
                                                                         random_state=50)

y_training_dataset, y_testing_dataset = model_selection.train_test_split(y,
                                                                         test_size=0.2,
                                                                         train_size=0.8,
                                                                         random_state=50)
print(len(X_training_dataset), len(X_testing_dataset))
print(len(y_training_dataset), len(y_testing_dataset))
print(X_training_dataset.size() , X_testing_dataset.size())


## plot training and testing datasets
plotting2D(X_training_dataset[:,0], X_training_dataset[:,1], "Training Data", y_training_dataset)
plotting2D(X_testing_dataset[:,0], X_testing_dataset[:,1], "Testing Data", y_testing_dataset)

plotting3D(X_training_dataset[:,0], X_training_dataset[:,1], y_training_dataset, "Training Data")
plotting3D(X_testing_dataset[:,0], X_testing_dataset[:,1], y_testing_dataset, "Testing Data")
# plt.show()


## create a model class
class BinaryClassification(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()         # ReLU -> Rectified Linear Unit -> nullifys all the negative values
                                      # adds non linearity to a linear model 

    def forward(self, X:torch.tensor) -> torch.tensor :
        return  self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(X)))))

model = BinaryClassification().to(device)
print(model)


## make a raw prediction
with torch.inference_mode():
    pred_y = model(X_testing_dataset.to(device))


## selecting loss functions, optimizers and accuracy
loss_fn = nn.BCEWithLogitsLoss()           # activation function -> round(sigmoid())
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.1)
def accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    accuracy = correct / len(y_pred) * 100
    return accuracy


## bringing training and testing datasets onto target device
X_training_dataset , X_testing_dataset = X_training_dataset.to(device), X_testing_dataset.to(device)
y_training_dataset, y_testing_dataset = y_training_dataset.to(device), y_testing_dataset.to(device) 


## creating a training and testing loop
epochs = 1500
training_loss_values = list()
testing_loss_values = list()
for epoch in range(epochs):
    model.train()
    y_logit_train = model(X_training_dataset).squeeze()
    y_pred_train = torch.round(torch.sigmoid(y_logit_train))
    training_loss = loss_fn(y_logit_train, y_training_dataset)
    training_accuracy = accuracy(y_training_dataset, y_pred_train)
    optimizer.zero_grad()
    training_loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        y_logit_test = model(X_testing_dataset).squeeze()
        y_pred_test = torch.round(torch.sigmoid(y_logit_test))
        testing_loss = loss_fn(y_logit_test, y_testing_dataset)
        testing_accuracy = accuracy(y_testing_dataset, y_pred_test)

    if epoch % 100 == 0 :
        print(f"Epoch : {epoch}")
        print(f"Training Loss {training_loss:.4f} | Testing Loss {testing_loss:.4f}")
        print(f"Training Accuracy {training_accuracy:.2f} | Testing Accuracy {testing_accuracy:.2f}")
        training_loss_values.append(training_loss.detach().numpy())
        testing_loss_values.append(testing_loss.detach().numpy())


## make prediction after training model and plot
with torch.inference_mode():
    y_pred = model(X_testing_dataset)

print(X_testing_dataset.size() , y_pred.size())

plotting3D(X_training_dataset[:,0].cpu().detach().numpy(),
            X_training_dataset[:,1].cpu().detach().numpy(),
              y_pred_train.cpu().detach().numpy(),
                "3D Training Data Predictions")
plotting3D(X_testing_dataset[:,0].cpu().detach().numpy(),
            X_testing_dataset[:,1].cpu().detach().numpy(),
              y_pred_test.cpu().detach().numpy(),
                "3D Testing Data Predictions")


## plot loss function graph
plt.figure(figsize=(8,5))
plt.plot(training_loss_values,label="Training Loss")
plt.plot(testing_loss_values,label="Testing Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


## saving model and state_dict
torch.save(model, r"D:\VisualStudioCode\Python\ML\pyTorch\Saved_Models\model_v1.pt")
# model._save_to_state_dict(r"D:\VisualStudioCode\Python\ML\pyTorch\Saved_Models\model_v1_state_dict.pt")
torch.save(model.state_dict(), r"D:\VisualStudioCode\Python\ML\pyTorch\Saved_Models\model_v1_state_dict.pt")