import torch
from torch.utils import data
from torchvision import transforms
from dataloader import MyDataset
from dataloader import UnZipFolder
import shutil
import os

# UnZip DataSet
if os.path.exists("data/images") or os.path.exists("data/mask"):
    print("Images or Mask folder exists")
else:
    UnZipFolder("D:/DataSet (NuScenes)/v1.0-trainval01_blobs.tgz", "samples/CAM_FRONT/", "data/images")
    #UnZipFolder("D:/DataSet (NuScenes)/v1.0-trainval02_blobs.tgz", "samples/CAM_FRONT/", "data/images")
    #UnZipFolder("D:/DataSet (NuScenes)/v1.0-trainval03_blobs.tgz", "samples/CAM_FRONT/", "data/images")
    #UnZipFolder("D:/DataSet (NuScenes)/v1.0-trainval04_blobs.tgz", "samples/CAM_FRONT/", "data/images")
    #UnZipFolder("D:/DataSet (NuScenes)/v1.0-trainval05_blobs.tgz", "samples/CAM_FRONT/", "data/images")
    #UnZipFolder("D:/DataSet (NuScenes)/v1.0-trainval06_blobs.tgz", "samples/CAM_FRONT/", "data/images")
    #UnZipFolder("D:/DataSet (NuScenes)/v1.0-trainval07_blobs.tgz", "samples/CAM_FRONT/", "data/images")
    #UnZipFolder("D:/DataSet (NuScenes)/v1.0-trainval08_blobs.tgz", "samples/CAM_FRONT/", "data/images")
    #UnZipFolder("D:/DataSet (NuScenes)/v1.0-trainval09_blobs.tgz", "samples/CAM_FRONT/", "data/images")
    #UnZipFolder("D:/DataSet (NuScenes)/v1.0-trainval10_blobs.tgz", "samples/CAM_FRONT/", "data/images")
    shutil.copytree("data/images", "data/mask")

#print(len(list_files_in_folder("data\samples")))
my_dataset = MyDataset(images_path="data\images",mask_path="data\mask",transform=transforms.ToTensor())
print("Found "+str(len(my_dataset))+" samples")

#Split in train, eval and test data
val_samples = int(0.2*len(my_dataset))
test_samples = int(0.2*len(my_dataset))
train_samples = len(my_dataset)-val_samples-test_samples

#Create train, eval and test dataset
train_dataset, val_dataset, test_dataset = data.random_split(dataset=my_dataset,lengths=[train_samples,val_samples,test_samples],generator=torch.Generator())
print("Samples split in: \n ",str(train_samples)," for train \n ",str(val_samples)," for validate \n ",str(test_samples)," for test \n")

#Parameters to use
config = {
    "batch_size": 64,
    }

#Create train, eval and test dataloader
train_loader = data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=config["batch_size"])
test_loader = data.DataLoader(test_dataset, batch_size=config["batch_size"])

