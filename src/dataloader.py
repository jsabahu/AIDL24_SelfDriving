import os
from torch.utils import data
from PIL import Image
import numpy as np
import torch
   
class MyDataset(data.Dataset):
    # Call the super().__init__() method in the __init__.
    def __init__(self, images_path, mask_path, transform=None):
        super().__init__()
        
        self.images_path = images_path
        self.mask_path = mask_path
        self.transform = transform
     
        # List all image files and mask files
        self.image_files = sorted([file for file in os.listdir(images_path) if file.lower().endswith('.jpg')])
        self.mask_files = sorted([file for file in os.listdir(mask_path) if file.lower().endswith('.png')])
        
        # Ensure the number of images matches the number of masks
        assert len(self.image_files) == len(self.mask_files), "Mismatch between number of images and masks"

    #Implement a __len__ method with the length of the dataset.
    def __len__(self):
        return len(self.image_files)

    #Implement a __getitem__ method that returns a transformed image and mask.
    def __getitem__(self, index):
        # Construct full file paths
        image_path = os.path.join(self.images_path, self.image_files[index])
        mask_path = os.path.join(self.mask_path, self.mask_files[index])
        
        # Open image and mask
        image = np.array(Image.open(image_path).convert("RGB"))  # Convert image to RGB array (width x height x 3)          
        mask = np.array(Image.open(mask_path)) # Get mask

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # Convert mask in flatten array 1D (width x height)
        mask = mask.flatten() 
        
        # Convert tensor to binary tensor
        mask = torch.where(mask > 125, torch.tensor(1), torch.tensor(0)).float()
        return image, mask