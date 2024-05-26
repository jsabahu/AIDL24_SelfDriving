import os
from torch.utils.data import Dataset, DataLoader
import pathlib
from PIL import Image
from torchvision import transforms
from utils import read_yaml
import matplotlib.pyplot as plt
import numpy as np
import torch


class MyDataset(Dataset):
    def __init__(self, images_path, mask_path, transform=None):
        super().__init__()

        # Resolve the root folder
        self.root_folder = pathlib.Path(__file__).parent.parent
        
        # Set paths for images and masks
        self.images_path = os.path.join(self.root_folder, images_path)
        self.mask_path = os.path.join(self.root_folder, mask_path)
        self.transform = transform

        # List all files in the images directory and filter for jpg files
        self.image_files = sorted([f for f in os.listdir(self.images_path) if f.lower().endswith(".jpg")])


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        """
        Retrieves an image and its corresponding mask, applies transformations, and returns them.
        """
        # Get the image file name and construct the full path
        image_name = self.image_files[index]
        image_path = os.path.join(self.images_path, image_name)
        
        # Assume the mask has the same base name but with a .png extension
        mask_name = f"{os.path.splitext(image_name)[0]}.png"
        mask_path = os.path.join(self.mask_path, mask_name)

        # Open image and mask using PIL
        image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB mode
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask == 1).type(torch.int)

        return image, mask
    
if __name__ == "__main__":
    # Example usage:
    config_path = "configs/config.yaml"
    config = read_yaml(config_path)
    # Set print options to show the full tensor values
    torch.set_printoptions(profile="full")

    my_transforms = transforms.Compose(
        [
            #transforms.Resize((256, 256), antialias=True),
            transforms.ToTensor(),
        ]
    )
    # Initialize the dataset and dataloader
    dataset = MyDataset(images_path=config["train"]["train_images_path"],mask_path=config["train"]["train_labels_path"], transform=my_transforms)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Retrieve one batch of data
    data_iter = iter(dataloader)
    images, masks = next(data_iter)

    # Extract the first image and mask from the batch
    image = images[0]
    mask = masks[0]

    # Print the entire values and types of the tensors
    # print("Image Tensor Values:")
    # print(image)
    print("Image Tensor Type:", type(image))
    print("Image Tensor Shape:", image.shape)

    # print("\nMask Tensor Values:")
    # print(mask)
    print("Mask Tensor Type:", type(mask))
    print("Mask Tensor Shape:", mask.shape)

    # print("\nMask Tensor Values (not equal to 1):")
    # non_one_values_mask = mask[mask != 1]
    # print(non_one_values_mask)
    # print("Mask Tensor Type:", type(mask))
    # print("Mask Tensor Shape:", mask.shape)

    # Convert the tensors to numpy arrays for plotting
    image_np = image.numpy().transpose(1, 2, 0)  # Convert to HWC format for plotting
    mask_np = mask.numpy().squeeze()             # Remove the channel dimension for the mask

    # Plot the image and mask
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image_np)
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(mask_np, cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')

    plt.show()

