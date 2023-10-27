if __name__ == "__main__":
    import torch, torchvision, torchinfo, torchmetrics
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import PIL 
    import requests, zipfile, pathlib
    from torch import nn  
    from torchvision import datasets, transforms, models
    from torch.utils.data import dataset, dataloader
    import random
    from typing import List, Dict, Tuple
    import os
    from timeit import default_timer as timer
    from tqdm.auto import tqdm

    SEED = 50
    BATCH_SIZE = 32
    EPOCHS = 5
    DATA_URL_RAW = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"

    # random.seed(SEED)
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed(SEED)

    device="cuda" if torch.cuda.is_available() else "cpu"

    def data_unavailable():
        dirPath= pathlib.Path("data/")
        imagePath= dirPath/"images"
        if dirPath.is_dir():
            print("The DirPath Exist, Checking ImagePath..")
            if imagePath.is_dir():
                print("The ImagePath also exists.")
            else:
                print("DirPath exists but ImagePath doesn't, creating now..")
                pathlib.Path.mkdir(imagePath)
        else : 
            print("DirPath is missing, creating now..")
            imagePath.mkdir(parents=True, exist_ok=True)
        print("Directory Structure Ready, Moving to download now..")

        with open(dirPath/"Sample.zip", "wb") as f:
            file= requests.get(DATA_URL_RAW)
            f.write(file.content)
        print("Download Complete, Moving to Extract..")

        with zipfile.ZipFile(dirPath/"Sample.zip", "r") as zip_ref:
            zip_ref.extractall(imagePath)
        print("Extraction Complete, Data is Ready.")


    def data_available(imagePath:str):
        trainPath = pathlib.Path(imagePath)/"train"
        testPath = pathlib.Path(imagePath)/"test"
        return  trainPath, testPath


    def fetching_classes(path:str) -> Tuple[List[str], Dict[str, int]]:
        classes = list(classes.name for classes in os.scandir(path) if pathlib.Path(path).is_dir())
        if not classes:
            raise FileNotFoundError("Either there is no data or the data is not in standard image classification format.")
        class_to_idx = {classes:i for i,classes in enumerate(classes)}
        return classes, class_to_idx

    class ImageFolderReplica(dataset.Dataset):
        def __init__(self,
                    tar_dir:str,
                    transform:torchvision.transforms):
            self.all_paths = list(pathlib.Path(tar_dir).glob("*/*.jpg"))
            self.transform = transform
            self.classes, self.class_to_idx = fetching_classes(tar_dir)

        def __len__(self) -> int:
            return len(self.all_paths)
        
        def load_image(self, index) -> PIL.Image.Image:
            image = self.all_paths[index]
            return PIL.Image.open(image)
        
        def __getitem__(self, index) -> torch.tensor:
            image = self.load_image(index)
            class_name = self.all_paths[index].parent.name
            class_id = self.class_to_idx[class_name]
            if self.transform:
                transformed_image = self.transform(image)
            return  transformed_image, class_id
        
    def plotting_random_images(path:str,
                               number_of_imgs:int,
                               model:nn.Module,
                               class_names:List[str],
                               image_size:Tuple[int,int]=(224,224),
                               transform:torchvision.transforms=None,
                               device=device):
        list_of_all_paths = list(pathlib.Path(path).glob("*/*.jpg"))
        random_indexes = random.sample(list_of_all_paths, number_of_imgs)
        for random_index in random_indexes:
            img = PIL.Image.open(random_index)
            if not transform:
                transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
            transformed_img = transform(img)
            model.eval()
            with torch.inference_mode():
                y_logits = model(transformed_img.unsqueeze(dim=0).to(device))  
                y_pred_probs = torch.softmax(y_logits, dim=1)
                y_pred_labels = y_pred_probs.argmax(dim=1)   

            plt.figure()
            plt.imshow(img)
            plt.title(f"Pred {class_names[y_pred_labels]} , prob {y_pred_probs.max():.3f}") 
            plt.axis(False)
        plt.show()

    #----------------------------------------------------------------------------------------------------
    images = r"D:\VisualStudioCode\Python\ML\pyTorch\Modularization\data\images"
    trainPath, testPath = data_available(images)

    manual_transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    automatic_transform = weights.transforms()

    train_dataset_m = ImageFolderReplica(tar_dir=trainPath,
                                        transform=manual_transform)
    test_dataset_m = ImageFolderReplica(tar_dir=testPath,
                                        transform=manual_transform)
    train_dataset_a = ImageFolderReplica(tar_dir=trainPath,
                                        transform=automatic_transform)
    test_dataset_a = ImageFolderReplica(tar_dir=testPath,
                                        transform=automatic_transform)
    
    class_names = train_dataset_m.classes

    print(train_dataset_m,                    # <__main__.ImageFolderReplica object at 0x000001680C14E590>
        train_dataset_m[40],                  # Image and Label as tensor, output of __getitem__() method
        len(train_dataset_m),                 # 225, output of __len__() len
        train_dataset_m.load_image(40),       # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x1680CD31950>, A PIL Image, to view use PIL.Image.show(img)
        train_dataset_m.classes,              # ['pizza', 'steak', 'sushi']
        train_dataset_m.class_to_idx,         # {'pizza': 0, 'steak': 1, 'sushi': 2}
        train_dataset_m.all_paths,            # all images path in train directory
        train_dataset_m.transform)            # transform attribute populated with passed parameter value

    train_dataloader_m = dataloader.DataLoader(dataset=train_dataset_m,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)
    test_dataloader_m = dataloader.DataLoader(dataset=test_dataset_m,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False)
    train_dataloader_a = dataloader.DataLoader(dataset=train_dataset_a,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True)
    test_dataloader_a = dataloader.DataLoader(dataset=test_dataset_a,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False)
    print(f"The Train Dataset of len {len(train_dataset_m)} has been divided into {len(train_dataloader_m)} batches of size {train_dataloader_m.batch_size} each.")
    print(f"The Test Dataset of len {len(test_dataset_m)} has been divided into {len(test_dataloader_m)} batches of size {test_dataloader_m.batch_size} each.")

    X, y = next(iter(train_dataloader_m))
    print(X, y)

    model = torchvision.models.efficientnet_b0(pretrained=True)
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    torchinfo.summary(model=model,
                      input_size=[32,3,224,224],
                      col_names=["input_size", "num_params", "output_size", "trainable"],
                      col_width=20,
                      row_settings=['var_names'])
    
    print(model.features)
    print(model.avgpool)
    print(model.classifier)

    for i in model.features.parameters():
        i.requires_grad = False

    torchinfo.summary(model=model,
                      input_size=[32,3,224,224],
                      col_names=["input_size", "num_params", "output_size", "trainable"],
                      col_width=20,
                      row_settings=['var_names'])
    
    classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=1280, out_features=len(class_names))
    )
    model.classifier = classifier

    torchinfo.summary(model=model,
                      input_size=[32,3,224,224],
                      col_names=["input_size", "num_params", "output_size", "trainable"],
                      col_width=20,
                      row_settings=['var_names'])
    
    model.eval()
    with torch.inference_mode():
        y_logits = model(X.to(device))
        y_pred_prob = torch.softmax(y_logits, dim=1)
        y_pred_labels = y_pred_prob.argmax(dim=1)
    
    print(y_pred_labels, y)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=0.001)
    def accuracy(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item()
        accuracy = correct/ len(y_pred) * 100
        return accuracy
    
    start_time = timer()
    results = {"training_loss":list(),
               "testing_loss":list(),
               "training_accuracy":list(),
               "testing_accuracy":list()}
    for epoch in tqdm(range(EPOCHS)):
        train_loss = 0
        train_acc = 0
        for batch, (X_train, y_train) in enumerate(train_dataloader_m):
            X_train = X_train.to(device)
            y_train = y_train.to(device)
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

        train_loss_avg = train_loss/ len(train_dataloader_m)
        train_acc_avg = train_acc/ len(train_dataloader_m)

        model.eval()
        test_loss = 0
        test_acc = 0
        with torch.inference_mode():
            for batch, (X_test, y_test) in enumerate(test_dataloader_m):
                X_test = X_test.to(device)
                y_test = y_test.to(device)
                y_logits_test = model(X_test)
                y_pred_test = torch.softmax(y_logits_test, dim=1).argmax(dim=1)
                loss = loss_fn(y_logits_test, y_test)
                test_loss += loss
                acc = accuracy(y_test, y_pred_test)
                test_acc += acc

        test_loss_avg = test_loss/ len(test_dataloader_m)
        test_acc_avg = test_acc/ len(test_dataloader_m)

        if epoch % 1 == 0:
            print("\n--------------------------------------------------------------------------------------------------------------------------------------")
            print(f"Epoch : {epoch}")
            results['training_loss'].append(train_loss_avg.cpu().detach().numpy())
            results['testing_loss'].append(test_loss_avg.cpu().detach().numpy())
            results['training_accuracy'].append(train_acc_avg)
            results['testing_accuracy'].append(test_acc_avg)
            print(f"Training Loss {train_loss_avg:.4f} , Testing Loss {test_loss_avg:.4f}")
            print(f"Training Accuracy {train_acc_avg:.2f} , Testing Accuracy {test_acc_avg:.2f}")
    print("--------------------------------------------------------------------------------------------------------------------------------------")
    end_time= timer()
    print(f"Time Taken by Experiment {end_time - start_time:.2f} sec. on device {str(next(model.parameters()).device)}")

    training_loss = results['training_loss']
    testing_loss = results['testing_loss']
    training_acc = results['training_accuracy']
    testing_acc = results['testing_accuracy']
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(training_loss, label="Training Loss")
    plt.plot(testing_loss, label="Testing Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(training_acc, label="Training Accuracy")
    plt.plot(testing_acc, label="Testing Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plotting_random_images(path=testPath,
                           number_of_imgs=3,
                           model=model,
                           class_names=class_names)