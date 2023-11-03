### Doing Multiple Experiments, and Tracking them using TensorBoard to find thr best model for our, ComputerVision-CustomDataset problem
if __name__ == "__main__":
    ## Imports
    import os
    import torch
    import torchvision
    import torchinfo
    import requests
    import random
    from torch import nn 
    from zipfile import ZipFile
    from pathlib import Path
    from typing import Tuple, List
    from tqdm.auto import tqdm
    from torchvision import transforms
    from datetime import datetime
    from timeit import default_timer as timer
    from Modularization.code import data_setup, engine, utils, predictions
    from torch.utils.tensorboard import writer

    ## Constants
    BATCH_SIZE = 32
    SEED = 50

    ## Device Agnostic Code
    device="cuda" if torch.cuda.is_available() else "cpu"

    ## Random Seeds
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    ## Creating function to fetch data from mrdbrouke's Github repo
    def data_gatherer(dataSource:str,
                    destination:str,
                    delete_source:bool) -> Tuple[str,str]:
        
        dataPath = Path(destination)
        imagePath = dataPath/ "Images"
        download_flag = False
        if dataPath.is_dir():
            print("dataPath exists, checking imagePath..")
            if imagePath.is_dir():
                print("imagePath also exists, checking data within..")
                if Path(imagePath/"train").is_dir():
                    print("Data exists, skipping download")
                else:
                    print("imagePath is empty, moving to download..")
                    download_flag = True
            else:
                print("imagePath doesn't exist, creating..")
                Path.mkdir(imagePath)
                download_flag = True
        else:
            print("dataPath doesn't exist, creating..")
            imagePath.mkdir(parents=True, exist_ok=True)
            download_flag = True

        if download_flag:
            print("downloading..")
            with open(dataPath/"Sample.zip", "wb") as f: 
                data = requests.get(dataSource)
                f.write(data.content)
            print("download complete, extracting..")
            with ZipFile(dataPath/"Sample.zip", "r") as zip_ref:
                zip_ref.extractall(imagePath)
            print("extraction complete.")

            if delete_source:
                os.remove(dataPath/"Sample.zip")
                print("source file deleted.")
        trainPath = imagePath/"train"
        testPath = imagePath/"test"
        return trainPath, testPath

    ## Fetching 10% and 20% data from mrdbrouke's repo
    trainPath_10, testPath_10 = data_gatherer(dataSource="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                                            destination=r"D:\VisualStudioCode\Python\ML\pyTorch\ComputerVision\SampleData\CustomFood101\Data_10_percent",
                                            delete_source=True)
    trainPath_20, testPath_20 = data_gatherer(dataSource="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
                                            destination=r"D:\VisualStudioCode\Python\ML\pyTorch\ComputerVision\SampleData\CustomFood101\Data_20_percent",
                                            delete_source=True)


    ## Creating Transform
    transform = transforms.Compose([
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])


    ## Creating Dataloaders using predefined "data_setup.py" script
    train_10_dataloader, test_10_dataloader, class_names= data_setup.convert_data_to_datasets(trainPath= trainPath_10,
                                                                                            testPath= testPath_10,
                                                                                            batch_size= BATCH_SIZE,
                                                                                            transform= transform)
    train_20_dataloader, test_20_dataloader, class_names= data_setup.convert_data_to_datasets(trainPath= trainPath_20,
                                                                                            testPath= testPath_10,
                                                                                            batch_size= BATCH_SIZE,
                                                                                            transform= transform)
    print(f"The Train 10 percent data on {trainPath_10} is splitted into {len(train_10_dataloader)} batches of {train_10_dataloader.batch_size} each.")
    print(f"The Test 10 percent data on {testPath_10} is splitted into {len(test_10_dataloader)} batches of {test_10_dataloader.batch_size} each.")
    print(f"The Train 20 percent data on {trainPath_20} is splitted into {len(train_20_dataloader)} batches of {train_20_dataloader.batch_size} each.")


    ## Creating a function to choose different models(for now, effnet_b0 and effnet_b2)
    def creating_model(model_name:str,
                    class_names:List[str],
                    device=device) -> torch.nn.Module:
        if model_name == "effnet_b0":
            model_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
            model = torchvision.models.efficientnet_b0(weights=model_weights).to(device)
            input_features = 1280
        elif model_name == "effnet_b2":
            model_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
            model = torchvision.models.efficientnet_b2(weights=model_weights).to(device)
            input_features = 1408
        else:
            print("Unknown Model")
            raise Exception("[ERR] Unknown Model, Please Reconsider!")
            return 0
        ## Feature Extraction
        # making features untrainable/ freezing
        for feature in model.features.parameters():
            feature.requires_grad = False
        # replacing classifier layer
        classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=input_features, out_features=len(class_names))
        ).to(device)
        model.classifier = classifier
        # print(torchinfo.summary(model=model,
        #                         input_size=[32,3,224,224],
        #                         col_names=["input_size", "output_size", "num_params", "trainable"],
        #                         col_width=20,
        #                         row_settings=["var_names"]))
        return model


    ## Creating a test model
    test_model = creating_model(model_name="effnet_b0",
                                class_names=class_names,
                                device=device)


    ## Creating a SummaryWriter instance and function
    # torch.utils.tensorboard.writer.SummaryWriter() is a pytorch utility for tensorboard, it's purpose is to track experiment scalers/tensors like loss and accuracy
    # in form of folders and tensorflow events like(events.out.tfevents.1698931560.84bdaa7bbe56.733.0) so it can be easily read by Tensorflow's TensorBoard to show graphs
    # instead of manually tracking loss and acc of every experiment and plotting charts for it, we can use tensorboard, which is much more efficient
    tb_writer = writer.SummaryWriter(log_dir=r"D:\VisualStudioCode\Python\ML\pyTorch\ComputerVision\runs")  # this type of call will take log_dir parameter as default(/runs/<date>_<time>_<sysCode>) -> (/runs/Nov02_13-26-00_84bdaa7bbe56)
    def creating_tensorboard_writer(experiment_name: str,
                                    model_name: str,
                                    extra: str=None) -> torch.utils.tensorboard.writer:
        timestamp = datetime.now().strftime("%Y-%m-%d")
        log_dir = os.path.join(r"D:\VisualStudioCode\Python\ML\pyTorch\ComputerVision\runs",timestamp,experiment_name,model_name,extra) if extra else os.path.join(r"D:\VisualStudioCode\Python\ML\pyTorch\ComputerVision\runs",timestamp,experiment_name,model_name)
        tb_writer = writer.SummaryWriter(log_dir)
        return tb_writer


    ## Creating Accuracy Function and any other evaluation metric you want
    def accuracy(y_true, y_pred) -> float:
        correct = torch.eq(y_true, y_pred).sum().item()
        accuracy = correct/ len(y_pred) * 100
        return accuracy


    ## Creating a Writer() integrated Full_loop(train+test), by using existing train and test loops in "engine.py"
    def newFullLoop(epochs:int,
                    model:torch.nn.Module,
                    train_dataloader:torch.utils.data.dataloader.DataLoader,
                    test_dataloader:torch.utils.data.dataloader.DataLoader,
                    loss_fn:torch.nn.Module,
                    optimizer:torch.optim.Optimizer,
                    tb_writer:torch.utils.tensorboard.writer,
                    accuracy,
                    device=device):
        epochs = epochs
        results = {"train_loss":[],
                "test_loss":[],
                "train_acc":[],
                "test_acc":[]}
        start_time = timer()
        for epoch in tqdm(range(epochs)):
            print("\n"+("-"*162))
            train_loss, train_acc = engine.train_loop(model=model,
                                                    train_dataloader=train_dataloader,
                                                    loss_ftn=loss_fn,
                                                    optimizer_ftn=optimizer,
                                                    accuracy_ftn=accuracy,
                                                    device=device)
            test_loss, test_acc = engine.test_loop(model=model,
                                                test_dataloader=test_dataloader,
                                                loss_ftn=loss_fn,
                                                accuracy_ftn=accuracy,
                                                device=device)
            print(f"Epoch : {epoch}")
            print(f"Training Loss {train_loss:.4f} , Training Accuracy {train_acc:.2f}")
            print(f"Testing Loss {test_loss:.4f} , Testing Accuracy {test_acc:.2f}")
            results['train_loss'].append(train_loss.cpu().detach().numpy())
            results['test_loss'].append(test_loss.cpu().detach().numpy())
            results['train_acc'].append(train_acc)
            results['test_acc'].append(test_acc)

            ### New Code : Writer() ###
            if tb_writer:
                tb_writer.add_scalars(main_tag="Loss",
                                    tag_scalar_dict={"Train_Loss":train_loss,
                                                    "Test_Loss":test_loss},
                                    global_step=epoch)
                tb_writer.add_scalars(main_tag="Accuracy",
                                    tag_scalar_dict={"Train_Acc":train_acc,
                                                    "Test_Acc":test_acc},
                                    global_step=epoch)
                tb_writer.add_graph(model=model,
                                    input_to_model=torch.randn(size=(32,3,224,224), device=device))
                tb_writer.close()
            else:
                pass
            ### End ###
        print("-"*162+"\n")
        end_time = timer()
        print(f"Time Taken by the Model to train is {end_time - start_time:.2f} sec. on device {str(next(model.parameters()).device)}.")
        return results
    

    ## Creating Loop for Modelling Experiments and Save each model using prebuilt method in "utils.py"
    experiment_num = 0
    epoch_list = [5,10]
    dataloaders = {"train_10_dataloader":train_10_dataloader,
                   "train_20_dataloader":train_20_dataloader}
    models = ["effnet_b0", "effnet_b2"]
    # starter = timer()
    # for dataloader_name, dataloader in dataloaders.items():
    #     for model_name in models:
    #         for epoch in epoch_list:

    #             experiment_num += 1
    #             print(f"Experiment : {experiment_num}")
    #             print(f"Dataloader : {dataloader_name}")
    #             print(f"Model : {model_name}")
    #             print(f"Epochs : {epoch}_epochs")

    #             model = creating_model(model_name=model_name,
    #                                    class_names=class_names,
    #                                    device=device)
    #             loss_fn = nn.CrossEntropyLoss()
    #             optimizer = torch.optim.Adam(params=model.parameters(),
    #                                          lr=0.001)
    #             tb_writer = creating_tensorboard_writer(experiment_name=dataloader_name,
    #                                                     model_name=model_name,
    #                                                     extra=str(epoch)+"_epochs")
    #             newFullLoop(epochs=epoch,
    #                         model=model,
    #                         train_dataloader=dataloader,
    #                         test_dataloader=test_10_dataloader,
    #                         loss_fn=loss_fn,
    #                         optimizer=optimizer,
    #                         tb_writer=tb_writer,
    #                         accuracy=accuracy,
    #                         device=device)
                
    #             utils.saving_model(dirPath=r"D:\VisualStudioCode\Python\ML\pyTorch\ComputerVision\Saved_Models\ExperimentTracking",
    #                                model_name=f"Model-{dataloader_name}-{model_name}-{epoch}_epochs.pt",
    #                                model=model)

    #             print("+"*162+"\n")
    # ender = timer()
    # print(f"Time Taken by All the Experimental Trainings {(ender - starter)/60:.2f} mins.")

    
    ## Choosing the Best Performing Model after Analysing TensorBoard Scalers
    # the model EfficientNet_B2,
    # trained over 3 labels of 20% data each,
    # for 10 epochs have produced the better results then rest.


    ## Creating a New instance of _b2 and Loading the best model's state dict on it
    best_model_path = r"D:\VisualStudioCode\Python\ML\pyTorch\ComputerVision\Saved_Models\ExperimentTracking\Model-train_20_dataloader-effnet_b2-10_epochs.pt"
    best_model = creating_model(model_name="effnet_b2",
                                class_names=class_names)
    best_model.load_state_dict(torch.load(best_model_path).state_dict())


    ## Making Predictions on Random Images from 20% test dataset, using function in "predictions.py"
    all_paths = list(Path(testPath_20).glob("*/*.jpg"))
    random_paths = random.sample(population=all_paths, k=3)
    for random_path in random_paths:
        predictions.predict_a_image(model=best_model,
                                    path=random_path,
                                    class_names=class_names,
                                    image_size=(224,224))
        

    ## Making a prediction on a "out of testing data" image
    predictions.predict_a_image(path=r"D:\VisualStudioCode\Python\ML\pyTorch\ComputerVision\SampleData\CustomFood101\steak_from_internet.jpg",
                                model=best_model,
                                class_names=class_names,
                                image_size=(224,224))
