### creating a pytorch model to train over Linear Regression 
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

## setting manual seed
torch.manual_seed(80)

## device agnostic code
torch.device = "cuda" if torch.cuda.is_available() else "cpu"

## lets take Linear Regression Formula as a known pattern and create ideal model 
# y = mx + c
slope = 0.9             # m
intercept = 0.1         # c

## create dataset within a desired range
X = torch.arange(100,537,2.7).unsqueeze(dim=1)
print(f'X {X}')
y = slope * X + intercept
print(len(y),len(X))

## lets split the dataset into training and testing by ratio 80:20
# finding 80% marker
X_split_marker = int(len(X) * 0.8)
y_split_marker = int(len(y) * 0.8)

# splitting
X_training , X_testing = X[:X_split_marker] , X[X_split_marker:]
y_training , y_testing = y[:y_split_marker] , y[y_split_marker:]
print(len(X_training), len(X_testing), len(y_training), len(y_testing))

## lets create a method to plot graphs
def plotting(training_data=X_training,
             testing_data=X_testing,
             training_label=y_training,
             testing_label=y_testing,
             predictions=None):
    # plot figure size
    plt.figure(figsize=(7,5))
    # plotting training dataset
    plt.scatter(x=training_data,y=training_label,s=3,c="r",label="Training Data")
    # plotting testing dataset
    plt.scatter(x=testing_data,y=testing_label,s=3,c="b",label="Testing Data")
    # plotting predictions if available
    if predictions != None:
        plt.scatter(x=testing_data,y=predictions,s=3,c="g",label="Prediction Data")
    # plotting legend
    plt.legend(prop={"size":10})

#NOTE - predictions parameter will contain the y_testing value generated at any given time by x_testing
##lets plot a graph of the ideal Linear Regression graph without any predictions
plotting()
plt.title("Ideal values for Linear Regression")

## lets create a model class for Liner Regression to train respect to ideal data
class LinearRegression(nn.Module):
    # constructor
    def __init__(self):
        super().__init__()
        self.intercept = nn.Parameter(torch.randn(1,
                                     requires_grad=True,
                                     dtype=torch.float))
        self.slope = nn.Parameter(torch.randn(1,
                                requires_grad=True,
                                dtype=torch.float))  
    # override forward method
    def forward(self,X:torch.tensor) -> torch.tensor:
        return self.intercept + self.slope * X

## create object for the class to be used further in code
model = LinearRegression()
print(model.parameters())              # will return a generator
print(list(model.parameters()))
print(model.state_dict())

## lets plot graph using ideal values and model generated random variable(slope and intercept) values
with torch.inference_mode():
    y_testing = model(X_testing)
plotting(predictions=y_testing)
plt.title("Ideal values with Random test Inputs")

## lets select loss function and optimizer
# loss function
loss_fn = nn.L1Loss()

# optimizer
optimizer = torch.optim.SGD(params=model.parameters(),
                            lr=0.0001)

## lets create loop for training and testing
epochs = 101
epoch_states = list()
epoch_count = list()
training_loss_values = list()
testing_loss_values = list()
for epoch in range(epochs):
 ## training
    # state of the model to train
    model.train()

    # training forward pass 
    y_train = model(X_training)

    # check loss
    training_loss = loss_fn(y_train, y_training)

    # make optimizer to zero grad 
    optimizer.zero_grad()

    # loss backward
    training_loss.backward()

    # optimizer step
    optimizer.step()

 ## testing
    # make model into testing mode
    model.eval()

    with torch.inference_mode():
        # testing forward pass
        y_test = model(X_testing)

        # check loss
        testing_loss = loss_fn(y_test, y_testing)

 ## printing epochs and logging states
    if epoch % 10 == 0:
        epoch_states.append(model.state_dict())
        epoch_count.append(epoch)
        training_loss_values.append(training_loss.detach().numpy())
        testing_loss_values.append(testing_loss.detach().numpy())
        print(f"epoch : {epoch}")
        print(f"training loss : {training_loss}")
        print(f"testing loss : {testing_loss}")
        print(model.state_dict())

## printing the learnings of model and plotting the graph
print(f'expected values - {slope}, {intercept}')
print(f'predicted values - {model.state_dict()["slope"]}, {model.state_dict()["intercept"]}')
with torch.inference_mode():
    y_testing = model(X_testing)

plotting(predictions=y_testing)
plt.title("Predicted testing values with Ideal values")

## plotting the loss curves
plt.figure(figsize=(7,5))
plt.plot(epoch_count, training_loss_values, label="Training Loss")
plt.plot(epoch_count, testing_loss_values, label="Testing Loss")
plt.title("Training and Testing Loss curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.show()               # to display matplotlib charts while running on console

## Saving Model, Saving state_dict()
torch.save(model,r"D:\VisualStudioCode\Python\ML\pyTorch\Saved_Models\model_v0.pt")
torch.save(model.state_dict(),r"D:\VisualStudioCode\Python\ML\pyTorch\Saved_Models\model_v0_state_dict.pt")



