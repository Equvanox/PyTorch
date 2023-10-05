### creating a pytorch model to train over Multiclass Classification
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, model_selection
from torch import nn    

## device agnostic code
device= "cuda" if torch.cuda.is_available() else "cpu"

## random seeds
torch.manual_seed(50)
torch.cuda.manual_seed(50)

## sample data
# let's take toy dataset as 2D blobs with (X) having 2 features and (y) having 6 classes/values)
X, y = datasets.make_blobs(n_samples=1000,
                           n_features=2,
                           centers=6,
                           cluster_std=0.7,
                           random_state=50)
print(X[:10], y[:10])
# plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.RdYlBu)
plt.axes(projection="3d").scatter3D(X[:,0], X[:,1], y, s=40, c=y, cmap=plt.cm.RdYlBu)
# plt.show()

## let's visualize using pandas dataframe 
df = pd.DataFrame({"X1":X[:,0],
                   "X2":X[:,1],
                   "label":y})
print(df)

## lets convert this numpy data into pytorch data
X = torch.from_numpy(X).type(torch.float).to(device)
y = torch.from_numpy(y).type(torch.float).to(device)
print(torch.is_tensor(X), X.dtype)
print(isinstance(X, np.ndarray), X.dtype)

## split into training and testing datasets using sklearn
X_training_dataset, X_testing_dataset, y_training_dataset, y_testing_dataset = model_selection.train_test_split(X,
                                                                                                                y,
                                                                                                                train_size=0.8,
                                                                                                                test_size=0.2,
                                                                                                                random_state=50)
print(len(X_training_dataset), len(X_testing_dataset), len(y_training_dataset), len(y_testing_dataset))


## create a model class
class MulticlassClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=2, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=16),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=6)
        )

    def forward(self, X:torch.tensor) -> torch.tensor :
        return self.layers(X)
    
model = MulticlassClassification().to(device)
print(model)
print(model.state_dict())


## making a raw prediction
model.eval()
with torch.inference_mode():
    y_logit = model(X_testing_dataset)
print(y_logit[:10], y_testing_dataset[:10])


## selecting loss/cost function and optimizer, create accuracy
loss_fn = nn.CrossEntropyLoss()   # activation function -> softmax().argmax()
optimizer = torch.optim.SGD(params=model.parameters(),
                         lr=0.1)
def accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    accuracy = correct / len(y_pred) * 100
    return accuracy

## making plot prediction method
def plotting(model, X_testing_dataset, y_testing_dataset):
    # everything on cpu
    X_testing_dataset = X_testing_dataset.to("cpu")
    y_testing_dataset = y_testing_dataset.to("cpu")

    # finding min and max
    x1_min , x1_max = X_testing_dataset[:,0].min() , X_testing_dataset[:,1].max()
    x2_min , x2_max = X_testing_dataset[:,1].min() , X_testing_dataset[:,1].max()

    # making equal parts of range
    x = np.linspace(x1_min, x1_max, 101)
    y = np.linspace(x2_min, x2_max, 101)

    # making grid of range
    X, Y = np.meshgrid(x,y)

    # find x value for the meshgrid and numpy to pytorch
    x_pred = torch.tensor(np.column_stack((X.ravel(), Y.ravel()))).type(torch.float).to(device)

    # make raw predictions(logits) with new x
    model.eval()
    with torch.inference_mode():
        y_logit = model(x_pred).to(device)
        
    # convert logits to pred_probs and then to pred_labels
    y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)

    # plot
    plt.figure()
    plt.contourf(X,Y,y_pred.reshape(X.shape).to(device))
    plt.scatter(x=X_testing_dataset[:,0], y=X_testing_dataset[:,1], s=40, c=y_testing_dataset, cmap=plt.cm.RdYlBu)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())

plotting(model, X_testing_dataset, y_testing_dataset)
# plt.show()


## create training and testing loop
# bring everything on target device
X_training_dataset, X_testing_dataset = X_training_dataset.to(device), X_testing_dataset.to(device)
y_training_dataset, y_testing_dataset = y_training_dataset.type(torch.long).to(device), y_testing_dataset.type(torch.long).to(device)

# loops
epochs = 1000
training_loss_values = list()
testing_loss_values = list()
for epoch in range(epochs):
    model.train()
    y_logit_train = model(X_training_dataset).to(device)
    y_pred_train = torch.softmax(y_logit_train, dim=1).argmax(dim=1).squeeze()
    training_loss = loss_fn(y_logit_train,y_training_dataset)
    training_accuracy = accuracy(y_training_dataset, y_pred_train)
    optimizer.zero_grad()
    training_loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        y_logit_test = model(X_testing_dataset).to(device)
        y_pred_test = torch.softmax(y_logit_test, dim=1).argmax(dim=1).squeeze()
        testing_loss = loss_fn(y_logit_test, y_testing_dataset)
        testing_accuracy = accuracy(y_testing_dataset, y_pred_test)
    
    if epoch % 100 == 0 :
        print(f"Epoch {epoch}")
        print(f"Training Loss {training_loss:.4f} , Testing Loss {testing_loss:.4f}")
        print(f"Training Accuracy {training_accuracy:.2f} , Testing Accuracy {testing_accuracy:.2f}")
        training_loss_values.append(training_loss.cpu().detach().numpy())
        testing_loss_values.append(testing_loss.cpu().detach().numpy())


## make predictions after training 
plotting(model, X_testing_dataset, y_testing_dataset)

## plot loss curve
plt.figure()
plt.title("Loss Curve Graph")
plt.xlabel("Epochs")
plt.plot(training_loss_values, label="Training Loss")
plt.ylabel("Loss")
plt.plot(testing_loss_values, label="Testing Loss")
plt.legend()
plt.show()

## saving model and state_dict()
torch.save(model, r"D:\VisualStudioCode\Python\ML\pyTorch\Saved_Models\model_v2.pt")
torch.save(model.state_dict(), r"D:\VisualStudioCode\Python\ML\pyTorch\Saved_Models\model_v2_state_dict.pt")
