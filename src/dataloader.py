import os
import zipfile
import tarfile
import shutil
from pathlib import Path
from torch.utils import data
from PIL import Image

def UnZipFolder(file_path, folder_inside, destination_path):
    # Unzipp a folder from inside a ZIP file to a destination
    if os.path.exists(destination_path+folder_inside):
        print("Folder: "+destination_path+" exists")
    else:
        if file_path.lower().endswith('.zip'):
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    print("Unzipping folder..." + folder_inside)
                    all_members = zip_ref.getmembers()
                    folder_members = [member for member in all_members 
                                      if member.name.startswith(folder_inside)]
                    zip_ref.extractall(path=destination_path, members=folder_members)
                    #Relocate the images to the destination_path
                    for member in folder_members:
                        if os.path.dirname(member.name):
                            os.rename(os.path.join(destination_path,member.name),
                                      os.path.join(destination_path,os.path.basename(member.name)))
                    #Delete empty folder
                    DeleteFolder = os.path.split(Path(folder_inside))
                    shutil.rmtree(os.path.join("data/images",DeleteFolder[0]))
                    print("Unzipped in..." + destination_path)

            except Exception as e:
                print("Error extracting from ZIP file:", e)

        # Unzipp a folder from inside a TAR/TGZ file to a destination
        elif file_path.lower().endswith('.tar') or file_path.lower().endswith('.tgz'):
            try:
                with tarfile.open(file_path, 'r:gz') as tar_ref:
                    print("Unzipping folder..." + folder_inside)
                    all_members = tar_ref.getmembers()
                    folder_members = [member for member in all_members 
                                      if member.name.startswith(folder_inside)]
                    tar_ref.extractall(path=destination_path, members=folder_members)
                    #Relocate the images to the destination_path
                    for member in folder_members:
                        if os.path.dirname(member.name):
                            os.rename(os.path.join(destination_path,member.name),
                                      os.path.join(destination_path,os.path.basename(member.name)))
                    #Delete empty folder
                    DeleteFolder = os.path.split(Path(folder_inside))
                    shutil.rmtree(os.path.join("data/images",DeleteFolder[0]))
                    print("Unzipped in..." + destination_path)

            except Exception as e:
                print("Error extracting from TAR/TGZ file:", e)

def list_files_in_folder(folder_path):
    try:
        # Get a list of all files and directories in the folder
        files_in_folder = os.listdir(folder_path)
        # Show only jpg files
        files = [file for file in files_in_folder if file.lower().endswith('.jpg')]
        return files
    except Exception as e:
        print("Error listing files in folder:", e)
        return None
    
class MyDataset(data.Dataset):
    # Call the super().__init__() method in the __init__.
    def __init__(self, images_path, mask_path, transform=None):
        super().__init__()
        
        self.images_path = images_path
        self.mask_path = mask_path
        self.transform = transform
  
    #Implement a __len__ method with the length of the dataset.
    def __len__(self):
        # Get a list of all files and directories in the folder
        files_in_folder = os.listdir(self.images_path)
        files_in_folder = sorted(files_in_folder)
        # Show only jpg files
        files = [file for file in files_in_folder if file.lower().endswith('.jpg')]
        return len(files)

    #Implement a __getitem__ method that returns a transformed image and mask.
    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.images_path, image_name)
        mask_path = os.path.join(self.mask_path, image_name)  # Assuming mask has the same filename as image
       
        #Open image and mask for index element 
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        #Apply transform if need
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask
