import torch
from torch.utils import data
from torchvision import transforms
from dataloader import MyDataset
from utils import read_yaml
from utils import UnZipFolder
from utils import list_files_in_folder
from logger import Logger

#Parameters to use
config = (read_yaml("configs\config.yaml"))

# Create a logger object
logger = Logger(log_file=config['dataloader']["log_filename"], level="debug")

#print(len(list_files_in_folder("data\samples")))
my_dataset = MyDataset(images_path=config['dataloader']["images_path"],mask_path=config['dataloader']["mask_path"],transform=transforms.ToTensor())
logger.log_debug("Found "+str(len(my_dataset))+" samples")

#Split in train, eval and test data
val_samples = int(0.2*len(my_dataset))
test_samples = int(0.2*len(my_dataset))
train_samples = len(my_dataset)-val_samples-test_samples

#Create train, eval and test dataset
train_dataset, val_dataset, test_dataset = data.random_split(dataset=my_dataset,lengths=[train_samples,val_samples,test_samples],generator=torch.Generator().manual_seed(42))
logger.log_debug("Samples split as follows: Train = "+str(train_samples)+" \ Validate = "+str(val_samples)+" \ Test = "+str(test_samples))

#Create train, eval and test dataloader
train_loader = data.DataLoader(train_dataset, batch_size=config['dataloader']["batch_size"], shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=config['dataloader']["batch_size"])
test_loader = data.DataLoader(test_dataset, batch_size=config['dataloader']["batch_size"])

