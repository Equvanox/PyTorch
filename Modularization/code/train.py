"""
This file is the master file for executing an entire model
"""
if __name__ == "__main__":
  # imports
  import torch
  import pathlib
  import torchinfo
  from torch import nn
  from torchvision import transforms
  import data_gatherer, data_setup, engine, model_builder, utils

  # random seed and device agnostic code
  torch.manual_seed(50)
  device ="cuda" if torch.cuda.is_available() else "cpu"

  # setting up directories
  PATH = pathlib.Path(data_gatherer.Path.root/"code/")
  if PATH.is_dir():
    print("Code Directory already exists, skipping creation")
  else:
    print("Code Directory missing, creating..")
    pathlib.Path.mkdir(data_gatherer.Path.root/"code/")
    print("Creation Complete.")

  # vars
  TRAIN_PATH = data_gatherer.Path.root/ "data/images/train"
  TEST_PATH = data_gatherer.Path.root/ "data/images/test"
  BATCH_SIZE = 32
  EPOCHS = 5

  # transforms (for this also we can create a separate file)
  simple_transformation = transforms.Compose([
      transforms.Resize(size=(64,64)),
      transforms.ToTensor()
  ])

  # dataloading
  train_dataloader, test_dataloader, class_names = data_setup.convert_data_to_datasets(trainPath= TRAIN_PATH,
                                                                                      testPath= TEST_PATH,
                                                                                      batch_size= 32,
                                                                                      transform= simple_transformation)
  X, y = train_dataloader.dataset[40]

  # model and dummy forward pass
  model = model_builder.TinyVGG(inputs=3,
                                hidden_layers=10,
                                output=len(class_names)).to(device)
  torchinfo.summary(model, (BATCH_SIZE,3,64,64))

  model.eval()
  with torch.inference_mode():
    y_logits= model(X.unsqueeze(dim=0).to(device))
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)
  print(y_pred, y)

  # loss/cost function, optimizer and accuracy
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(params=model.parameters(),lr=0.01)
  def accuracy(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    accuracy = correct/ len(y_pred) * 100
    return accuracy

  # training and testing Loops
  engine.full_loop(epochs=EPOCHS,
                  model=model,
                  train_dataloader=train_dataloader,
                  test_dataloader=test_dataloader,
                  loss_fn=loss_fn,
                  optimizer=optimizer,
                  accuracy=accuracy)

  # saving model
  utils.saving_model(model=model,
                    dirPath=data_gatherer.Path.root/"models",
                    model_name="model_v1.pt")
