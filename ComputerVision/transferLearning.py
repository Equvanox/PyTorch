### Transfer Learning for the ComputerVision-CustomDataset problem
if __name__ == "__main__":     # necessary for running next(iter()) and enumerate()
    ## Imports
    import torch
    import torchinfo
    import torchvision
    import random
    import requests
    import matplotlib.pyplot as plt
    from torch import nn
    from pathlib import Path
    from PIL import Image
    from zipfile import ZipFile
    from torchvision import datasets, transforms, models
    from Modularization.code import data_setup, engine
    from .SampleData import CustomFood101
    from torch.utils.data import dataloader
    from typing import Dict, List, Tuple


    ## Device Agnostic code and random seeds
    device= "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(50)
    torch.manual_seed(50)
    torch.cuda.manual_seed(50)


    # ## Loading Data from git (create folder -> download -> extract)
    # dirPath = Path("data/")
    # imagePath = dirPath / "images"
    # if dirPath.is_dir():
    #     print("dirPath already exists")
    #     if imagePath.is_dir():
    #         print("imagePath already exists.. moving to download")
    #     else:
    #         print("imagePath doesn't exists, creating now..")
    #         Path.mkdir(imagePath)
    # else:
    #     print("dirPath and imagePath doesn't exists, creating now..")
    #     imagePath.mkdir(parents=True, exist_ok=True)

    # with open(dirPath/"Samples.zip", "wb") as f:
    #     data = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip") # please note url should be ../raw/main/.. and not ../blob/main/..
    #     f.write(data.content)

    # with ZipFile(dirPath/"Sample.zip", "r") as zip_ref:
    #     zip_ref.extractall(imagePath)

    # trainPath = imagePath /"train"
    # testPath = imagePath /"test"


    ## Loading already downloaded data
    trainPath = Path(r"D:\VisualStudioCode\Python\ML\pyTorch\ComputerVision\SampleData\CustomFood101\Images\train")
    testPath = Path(r"D:\VisualStudioCode\Python\ML\pyTorch\ComputerVision\SampleData\CustomFood101\Images\test")


    ## Creating transforms in 2 different ways 
    # MANUALLY -> since we decided to use EfficientNet_B0 model, we have to take care of the data format that model accepts
    # size atleast 224, values b/w 0-1, normalization should be mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
    manual_transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # AUTOMATICALLY -> copying the pretrained model's transform
    weights = models.EfficientNet_B0_Weights.DEFAULT
    automatic_transform = weights.transforms()
    print(manual_transform, automatic_transform)


    ## Creating Dataloaders using data_setup.py for both transform
    train_dataloader_m, test_dataloader_m, class_names = data_setup.convert_data_to_datasets(trainPath=trainPath,
                                                                                            testPath=testPath,
                                                                                            batch_size=32,
                                                                                            transform=manual_transform)
    train_dataloader_a, test_dataloader_a, class_names = data_setup.convert_data_to_datasets(trainPath=trainPath,
                                                                                            testPath=testPath,
                                                                                            batch_size=32,
                                                                                            transform=automatic_transform)


    ## Initializing the model with pretrained model in 2 different ways
    # OLD -> using "pretrained" parameter -> soon to be deprecated
    model = torchvision.models.efficientnet_b0(pretrained=True).to(device)
    #print(model)

    # NEW -> using "weights" parameters
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)
    #print(model)


    ## Feature Extraction (freezing the features and manipulating the output layer)
    # freezing features/
    # freezing hidden layers/
    # making them untrainable/
    # disabling gradient tracking(stopping them from learning further)/
    # making the weights static
    for feature in model.features.parameters():
        feature.requires_grad = False

    # manipulating the output layer/ 
    # replaceing the output layer(classifier) with our own custom(but identical) layer
    # to match the output labels with our problem
    print(model.classifier)
    classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=len(class_names))
    )
    print(classifier)
    model.classifier = classifier
    # now the model is fitted for out problem


    ## Making a dummy forward pass
    # (its not gonna learn anything new in current state since there are no loss function and optimizer to guide it)
    X,y = next(iter(train_dataloader_m))
    model.eval()
    with torch.inference_mode():
        y_logits = model(X.to(device))
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
    print(y_pred, y)


    ## Selecting Loss/Cost function and Optimizer and creating accuracy(evaluation metric)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=0.001)
    def accuracy(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        accuracy = correct/ len(y_pred) * 100
        return accuracy
    

    ## Training the model using engine.py functions
    results_m = engine.full_loop(epochs=5,
                                 model=model,
                                 train_dataloader=train_dataloader_m,
                                 test_dataloader=test_dataloader_m,
                                 loss_fn=loss_fn,
                                 optimizer=optimizer,
                                 accuracy=accuracy,
                                 device=device)
    # results_a = engine.full_loop(epochs=5,
    #                              model=model,
    #                              train_dataloader=train_dataloader_a,
    #                              test_dataloader=test_dataloader_a,
    #                              loss_fn=loss_fn,
    #                              optimizer=optimizer,
    #                              accuracy=accuracy,
    #                              device=device)
    

    ## creating a function to print loss and accuracy curves
    def plot_curves(results: Dict[str,float]):
        training_loss = results["training_loss"]
        training_accuracy = results["training_accuracy"]
        testing_loss = results["testing_loss"]
        testing_accuracy = results["testing_accuracy"]

        plt.figure(figsize=(15,5))
        plt.subplot(1,2,1)
        plt.plot(training_loss, label="Training Loss")
        plt.plot(testing_loss, label="Testing Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(training_accuracy, label="Training Accuracy")
        plt.plot(testing_accuracy, label="Testing Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.legend()
        #plt.show()
    plot_curves(results=results_m)
    # plot_curves(results=results_a)


    ## Creating a function to make prediction on a single image
    def predict_a_image(path:str,
                        model:nn.Module,
                        class_names:List[str],
                        image_size:Tuple[int]=(224,224),
                        transform:torchvision.transforms=None,
                        device=device):
        img = Image.open(path)
        if not transform:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        transformed_img = transform(img).to(device)
        model.eval()
        with torch.inference_mode():
            y_logits = model(transformed_img.unsqueeze(dim=0))
            y_pred_probs = torch.softmax(y_logits, dim=1)
            y_pred_labels = y_pred_probs.argmax(dim=1)

        plt.figure()
        plt.imshow(img)
        plt.title(f"Pred {class_names[y_pred_labels]}, Prob {y_pred_probs.max():.3f}")
        plt.axis(False)
        plt.show()
        

    ## Creating a function to randomly pick desired number of images and predict on it
    def predictions(tar_dir:str,
                    model:nn.Module,
                    no_of_images:int,
                    class_names:List[str],
                    transform:torchvision.transforms=None):
        list_of_images = list(Path(tar_dir).glob("*/*.jpg"))
        random_images = random.sample(population=list_of_images, k=no_of_images)
        for random_img in random_images:
            predict_a_image(path=random_img,
                            model=model,
                            class_names=class_names,
                            transform=transform)
                

    ## Making Predictions on Images using above 2 methods
    predictions(tar_dir=testPath,
                model=model,
                no_of_images=3,
                class_names=class_names)