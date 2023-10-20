### Working with Custom Data and Augmentation
## Imports 
import os
import torch
import torchinfo
import torchvision
import requests
import zipfile 
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch import nn
from PIL import Image
from torch.utils.data import dataloader, dataset
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from typing import List, Dict, Tuple
from timeit import default_timer as timer


## Device Agnostic Code
device= "cuda" if torch.cuda.is_available() else "cpu"


## Requesting http GET for Git Repo, to fetch smaller Food101 dataset and putting into folders of `Standard Image Classification Format`
# making base folder in root directory
dirPath = Path(r"D:\VisualStudioCode\Python\ML\pyTorch\ComputerVision\SampleData\CustomFood101")
imagePath = dirPath / "Images"
if imagePath.is_dir():
    print("Images Folder Already Exists, Moving for Downloading..")
else:
    Path.mkdir(imagePath)
    print("Images Folder Created, Moving for Downloading...")

# creating a file and overwriting it with the one we fetched from git
with open(dirPath/"SampleData.zip", "r") as e:
    if e.__sizeof__() > 0:
        print("File Already Exists, Skipping Download..")
    else:
        with open(dirPath/ "SampleData.zip", "wb") as f:
            data = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
            f.write(data.content)
            print("Downloading Complete, Sample Data Acquired.")

# extracting the contents of zipfile into Images folder
with zipfile.ZipFile(dirPath/"SampleData.zip", "r") as ref_zip:
    ref_zip.extractall(imagePath)
    print("Data has been extracted from Zipfile.")
trainPath = imagePath/ "train"
testPath = imagePath/ "test"


## Traversing Data using os.walk()
def traverse(path):
    '''This method will traverse the path parameter directory using os.walk()'''
    for dirPath, dirname, filename in os.walk(path):
        print(f"{len(dirname)} Folders and {len(filename)} Files in {dirPath} Directory")
traverse(imagePath)


## Getting allPaths, randomPath and multiple random paths
allImagePaths = list(imagePath.glob("*/*/*.jpg"))
random_path = random.choice(allImagePaths)
random_paths = random.sample(allImagePaths, 3)


## Visualizing Randomly using PIL(PIllow Library or Python Image Library)
# single random image
with Image.open(random_path) as image:
    Image.Image.show(image)
    print(f"Image Height - {image.height}")
    print(f"Image Width - {image.width}")
    print(f"Image Size - {image.size}")
    print(f"Image Class - {random_path.parent.stem}")
    print(f"Image Path - {random_path}")
# multiple random images
for i in random_paths:
    # Image.Image.show(Image.open(i))   # to open image in image viewer on device, show() is necessary
    Image.open(i)


## Visualizing Randomly using matplotlib
# single image
with Image.open(random_path) as image:
    plt.imshow(np.asarray(image))
    plt.title(random_path.parent.stem)
    plt.axis(False)
    #plt.show()


## Transforms and Augments
# simple tranformation
simple_transformation = transforms.Compose([
    transforms.Resize(size=(128,128)),
    transforms.ToTensor()
])
# with random filps 
simple_transformation_with_flips = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])
# with augmentation
augmentation_transformation = transforms.Compose([
    transforms.Resize(size=(128,128)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])
# with augmentation and filps
augmentation_transformation_with_flips = transforms.Compose([
    transforms.Resize(size=(128,128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

## Visualizing Images before and after
def visual_comapare(allPaths,
                    no_of_images: int,
                    transform : torchvision.transforms,
                    seed : any = None):
    '''This method will return images with and without transforms in comparision'''
    # setting random seed if available
    if seed:
        random.seed(seed)
    # take random samples out of paths
    random_img_paths = random.sample(allPaths, no_of_images)
    # looping
    for i, random_img in enumerate(random_img_paths):
        # opening img
        with Image.open(random_img) as img:
            # plotting
            fig , ax = plt.subplots(nrows=1,ncols=2)
            ax[0].imshow(img)
            ax[0].set_title(img.size)
            ax[0].axis(False)
            transformed_img = transform(img).permute(1,2,0)
            ax[1].imshow(transformed_img)
            ax[1].set_title(transformed_img.shape)
            ax[1].axis(False)
            fig.suptitle(random_img.parent.stem)
    #plt.show()
visual_comapare(allPaths=allImagePaths,
                no_of_images=3,
                transform=augmentation_transformation,
                seed=51)


## Create/Use Dataset creater class(converts data in files into a ML Dataset)
# torchvision.datasets.ImageFolder Class
train_dataset = datasets.ImageFolder(root=trainPath,
                                     transform=augmentation_transformation)
test_dataset = datasets.ImageFolder(root=testPath,
                                    transform=simple_transformation)
print(train_dataset,
      train_dataset.classes,
      train_dataset.class_to_idx,
      len(train_dataset), train_dataset[10]
      )


# function to return classes and class_to_idx -> custom dataset creater class
def returns_labels(path) -> Tuple[List[str], Dict[str,int]]:
    '''This method returns the classes and class_to_idx values for parameter path'''
    classes = sorted(folders.name for folders in os.scandir(path) if path.is_dir())
    if not classes:
        raise FileNotFoundError("Either the requested Directory doesn't have any folders to be considered as classes OR the files are not in Standard Image Classification Format")
    class_to_idx = {classname:i for i,classname in enumerate(classes)}
    return classes, class_to_idx
classes, class_to_idx = returns_labels(trainPath)
print(classes, class_to_idx)

class CustomImageFolder(dataset.Dataset):
    '''Purpose of this class is to be a Replica of ImageFolder, created Customly using torch.utils.data.dataset.Dataset'''
    def __init__(self,
                 tar_dir:str,
                 transform:torchvision.transforms):
        '''Parameterised Constructor, Purpose -> Create all Important Attributes'''
        self.allPaths = list(Path(tar_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = returns_labels(tar_dir)

    def load_images(self, index:int):
        '''Method to open an image using "index" parameter and returns loaded image'''
        image = self.allPaths[index]
        return Image.open(image)
    
    def __len__(self):
        '''Method to return the length of class parameter directory'''
        return len(self.allPaths)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        '''Method to forge the pair of (image, label) and return it. Image from load_images() method and label from index -> imgpath -> parent(classname) -> class_idx'''
        image = self.load_images(index)
        class_name = self.allPaths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        if self.transform:
            return self.transform(image) , class_idx
        else:
            return image, class_idx
        
train_cust_dataset = CustomImageFolder(tar_dir=trainPath,
                                       transform=augmentation_transformation)
test_cust_dataset = CustomImageFolder(tar_dir=testPath,
                                      transform=simple_transformation)
print(train_cust_dataset,
      train_cust_dataset.classes,
      train_cust_dataset.class_to_idx,
      len(train_cust_dataset),
      train_cust_dataset[10],
      Image.Image.show(train_cust_dataset.load_images(10))
      )


## Functionizing Visualization and visualizing the datasets from both dataset creater classes
def visualize(dataset: torch.utils.data.dataset,
              classnames: List[str],
              no_of_images: int,
              display_shape: bool=True,
              seed:int=None):
    if seed:
        random.seed(seed)
    if no_of_images > 10 :
        no_of_images = 10
        display_shape = False
        print("Since Images are more than 10, Shape will not be displayed.")
    random_idxs = random.sample(range(len(dataset)), no_of_images)
    plt.figure(figsize=(13,4))
    for i, random_idx in enumerate(random_idxs):
        image, label = dataset[random_idx]
        plt.subplot(1, no_of_images, i+1)
        plt.imshow(image.permute(1,2,0))
        if display_shape:
            title = f"Class {classnames[label]}\n Shape {image.shape}"
        else:
            title = f"Class {classnames[label]}"
        plt.title(title, fontsize=10)
        plt.axis(False)
    #plt.show()

visualize(dataset=train_dataset,
          classnames=train_dataset.classes,
          no_of_images=5)
visualize(dataset=train_cust_dataset,
          classnames=train_cust_dataset.classes,
          no_of_images=5)


## Creating Dataloaders for train and test of both kind, splitting into batches
train_dataloader = dataloader.DataLoader(dataset=train_dataset,
                                         batch_size=32,
                                         shuffle=True)
test_dataloader = dataloader.DataLoader(dataset=test_dataset,
                                        batch_size=32,
                                        shuffle=False)

train_cust_dataloader = dataloader.DataLoader(dataset=train_cust_dataset,
                                              batch_size=32,
                                              shuffle=True)
test_cust_dataloader = dataloader.DataLoader(dataset=test_cust_dataset,
                                             batch_size=32,
                                             shuffle=False)
print(f"The Train Datasets of Length {len(train_dataset)} has been splitted into {len(train_dataloader)} batches of Size {train_dataloader.batch_size} each.")
print(f"The Test Datasets of Length {len(test_dataset)} has been splitted into {len(test_dataloader)} batches of Size {test_dataloader.batch_size} each.")



## Creating a Baseline model, replicating the TinyVGG architecture from CNN Explainer website
class TinyVGG(nn.Module):
    def __init__(self,
                 inputs,
                 hidden_layers,
                 output):
        super().__init__()
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=inputs,
                      out_channels=hidden_layers,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_layers,
                      out_channels=hidden_layers,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_layers,
                      out_channels=hidden_layers,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_layers,
                      out_channels=hidden_layers,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        # self.convBlock3 = nn.Sequential(
        #     nn.Conv2d(in_channels=hidden_layers,
        #               out_channels=hidden_layers,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=hidden_layers,
        #               out_channels=hidden_layers,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2,
        #                  stride=2)
        # )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_layers *32*32,
                      out_features=output)
        )

    def forward(self, X:torch.Tensor) -> torch.Tensor:
        # print(f"Shape of X before TinyVGG {X.shape}")
        X = self.convBlock1(X)
        # print(f"Shape of X after ConvBlock1 {X.shape}")
        X = self.convBlock2(X)
        # print(f"Shape of X after ConvBlock2 {X.shape}")
        # X = self.convBlock3(X)
        # print(f"Shape of X after ConvBlock3 {X.shape}")
        X = self.classifier(X)
        # print(f"Shape of X after Classifier {X.shape}")
        return X
    
model_v1 = TinyVGG(inputs=3,
                   hidden_layers=5,
                   output=len(train_dataset.classes)).to(device)
print(model_v1)


## Make a forward pass and find out the value for in_features
X_train, y_train = train_dataset[70]
model_v1.eval()
with torch.inference_mode():
    y_logits = model_v1(X_train.unsqueeze(dim=0).to(device))
print(y_logits)   


## Model Summary using torchinfo
torchinfo.summary(model=model_v1, input_size=(32,3,128,128))


## Select Loss/Cost function, Optimizer and Create Accuracy
loss_fn = nn.CrossEntropyLoss() # since it's a multiclass classification problem
optimizer = torch.optim.Adam(params=model_v1.parameters(),
                             lr=0.001)
def accuracy(y_true, y_pred) -> float:
    correct = torch.eq(y_true, y_pred).sum().item()
    accuracy = correct / len(y_pred) * 100
    return accuracy


## Create Functionaized Training/Testing and Overall loop
def train_loop(model: nn.Module,
               dataloader: torch.utils.data.dataloader,
               loss_ftn: nn.Module,
               optimizer_ftn: torch.optim.Optimizer,
               accuracy_ftn):
    model.train()
    train_acc = 0
    train_loss = 0
    for batch, (X_train, y_train) in enumerate(dataloader):
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        y_logits_train = model(X_train)
        y_pred_train = torch.softmax(y_logits_train, dim=1).argmax(dim=1)
        loss = loss_ftn(y_logits_train, y_train)
        train_loss += loss
        acc = accuracy_ftn(y_train, y_pred_train)
        train_acc += acc
        optimizer_ftn.zero_grad()
        loss.backward()
        optimizer_ftn.step()
    #print(f"Number of batches trained : {batch+1}")
    train_loss_avg = train_loss / len(dataloader)
    train_acc_avg = train_acc / len(dataloader)
    return train_loss_avg, train_acc_avg

def test_loop(model: nn.Module,
              dataloader: torch.utils.data.dataloader,
              loss_ftn: nn.Module,
              accuracy_ftn):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.inference_mode():
        for batch, (X_test, y_test) in enumerate(dataloader):
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            y_logits_test = model(X_test)
            y_pred_test = torch.softmax(y_logits_test, dim=1).argmax(dim=1)
            loss = loss_ftn(y_logits_test, y_test)
            test_loss += loss
            acc = accuracy_ftn(y_test, y_pred_test)
            test_acc += acc
        #print(f"No of batches evaluated : {batch+1}")
        test_loss_avg = test_loss / len(dataloader)
        test_acc_avg = test_acc / len(dataloader)
    return test_loss_avg, test_acc_avg

def full_loop(epochs:int,
              loop_model: nn.Module,
              training_dataloader: torch.utils.data.dataloader,
              testing_dataloader: torch.utils.data.dataloader,
              loop_loss_fn: nn.Module,
              loop_optimizer: torch.optim.Optimizer,
              loop_accuracy):
    epochs = epochs
    start_time = timer()
    results = {"training_loss":[],
               "training_accuracy":[],
               "testing_loss":[],
               "testing_accuracy":[]}
    for epoch in tqdm(range(epochs)):
        print("-------------------------------------------------------------------------------------------------------------------------")
        train_loss, train_accuracy = train_loop(model=loop_model,
                                                dataloader=training_dataloader,
                                                loss_ftn=loop_loss_fn,
                                                optimizer_ftn=loop_optimizer,
                                                accuracy_ftn=loop_accuracy)
        test_loss, test_accuracy = test_loop(model=loop_model,
                                             dataloader=testing_dataloader,
                                             loss_ftn=loop_loss_fn,
                                             accuracy_ftn=loop_accuracy)
        print(f"Epoch - {epoch}")
        print(f"Training Loss {train_loss:.4f} , Training Accuracy {train_accuracy:.2f}")
        print(f"Testing Loss {test_loss:.4f} , Test Accuracy {test_accuracy:.2f}")
        results["training_loss"].append(train_loss.item())
        results["training_accuracy"].append(train_accuracy)
        results["testing_loss"].append(test_loss.item())
        results["testing_accuracy"].append(test_accuracy)
    print("-------------------------------------------------------------------------------------------------------------------------")
    end_time = timer()
    print(f"Time Taken by the Experiment is {(end_time - start_time):.2f} sec. on device {str(next(loop_model.parameters()).device)}")
    return results

results = full_loop(epochs=30,
                    loop_model=model_v1,
                    training_dataloader=train_dataloader,
                    testing_dataloader=test_dataloader,
                    loop_loss_fn=loss_fn,
                    loop_optimizer=optimizer,
                    loop_accuracy=accuracy)
print(results)


## Plotting Loss and Accuracy Curves
def plotting_curves(results: Dict[str,List[float]]):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(results['training_loss'], label="Training Loss")
    plt.plot(results['testing_loss'], label="Testing Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(results['training_accuracy'], label="Training Accuracy")
    plt.plot(results['testing_accuracy'], label="Testing Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.show()
plotting_curves(results=results)