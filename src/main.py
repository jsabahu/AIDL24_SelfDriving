import torch
from torch.utils import data
from torchvision import transforms
from dataloader import MyDataset
from utils import read_yaml
from utils import UnZipFolder
from utils import list_files_in_folder

#Parameters to use
config = (read_yaml("configs\config.yaml"))

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

#Create train, eval and test dataloader
train_loader = data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=config["batch_size"])
test_loader = data.DataLoader(test_dataset, batch_size=config["batch_size"])
"""
