"""
The Purpose of this file is to create the required directory structured and have the data downloaded in that from Github
"""
import requests
import zipfile
from pathlib import Path
from tqdm.auto import tqdm

Path.root = Path(r"D:\VisualStudioCode\Python\ML\pyTorch\Modularization")

dataPath = Path(Path.root/"data")
imagePath = Path(dataPath / "images")

if dataPath.is_dir():
  print("Data Directory already exists, Moving to download..")
else :
  print("Data Directory doesn't exist, creating now..")
  tqdm(imagePath.mkdir(parents=True, exist_ok=True))
  print("Data Directory created, Downloading now..")

with open(dataPath / "Sample.zip", "wb") as f:
  data = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
  print("Downloading..")
  f.write(data.content)

with zipfile.ZipFile(dataPath / "Sample.zip", "r") as zip_ref:
  print("Extracting..")
  zip_ref.extractall(imagePath)

print("Completed")