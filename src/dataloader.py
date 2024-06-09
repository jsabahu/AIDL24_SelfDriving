import os
from torch.utils.data import Dataset, DataLoader
import pathlib
from PIL import Image
from torchvision import transforms
from utils import read_yaml
import matplotlib.pyplot as plt
import numpy as np
import torch

import random

CONFIG_PATH = "configs/config.yaml"


class MaskDataset(Dataset):
    def __init__(self, images_path, mask_path, transform=None):
        super().__init__()
        self._root_folder = pathlib.Path(__file__).parent.parent
        self.images_path = os.path.join(self._root_folder, images_path)
        self.mask_path = os.path.join(self._root_folder, mask_path)
        self.transform = transform

        self.image_files = sorted(
            [f for f in os.listdir(self.images_path) if f.lower().endswith(".jpg")]
        )
        self.mask_files = sorted(
            [file for file in os.listdir(mask_path) if file.lower().endswith(".png")]
        )

        assert len(self.image_files) == len(self.mask_files)

        self._shuffle()

    def _shuffle(self):
        c = list(zip(self.image_files, self.mask_files))
        random.shuffle(c)
        self.image_files, self.mask_files = zip(*c)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        """
        Retrieves an image and its corresponding mask, applies transformations, and returns them.
        """
        image_name = self.image_files[index]
        image_path = os.path.join(self.images_path, image_name)

        mask_name = f"{os.path.splitext(image_name)[0]}.png"
        mask_path = os.path.join(self.mask_path, mask_name)

        image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB mode
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(
                mask
            )  # Es correcte aplicar totes les transformacions a la mask?????

        mask = (mask == 1).type(torch.int)

        return image, mask

class Dataset_Mask_R_CNN(Dataset):
    def __init__(self, images_path, mask_path, batch_size, transform=None, transform_mask=None):
        super().__init__()
        self._root_folder = pathlib.Path(__file__).parent.parent
        self.images_path = os.path.join(self._root_folder, images_path)
        self.mask_path = os.path.join(self._root_folder, mask_path)
        self.transform = transform
        self.transform_mask = transform_mask

        self.image_files = sorted(
            [f for f in os.listdir(self.images_path) if f.lower().endswith(".jpg")]
        )
        self.mask_files = sorted(
            [file for file in os.listdir(mask_path) if file.lower().endswith(".png")]
        )

        assert len(self.image_files) == len(self.mask_files)
        self._shuffle()
        self._epoch_multiple(batch_size)

    def _shuffle(self):
        c = list(zip(self.image_files, self.mask_files))
        random.shuffle(c)
        self.image_files, self.mask_files = zip(*c)

    def _epoch_multiple(self,batch_size):
        limit_samples = int(len(self.image_files) / batch_size) * batch_size
        self.image_files = self.image_files[:limit_samples] 
        self.mask_files = self.mask_files[:limit_samples]
        print(limit_samples)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        """
        Retrieves an image and its corresponding mask, applies transformations, and returns them.
        """
        image_name = self.image_files[index]
        image_path = os.path.join(self.images_path, image_name)

        mask_name = f"{os.path.splitext(image_name)[0]}.png"
        mask_path = os.path.join(self.mask_path, mask_name)

        image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB mode
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform_mask(mask)
        mask = (mask == 1).type(torch.int)

        return image, mask

if __name__ == "__main__":
    config = read_yaml(CONFIG_PATH)

    my_transforms = transforms.Compose(
        [
            transforms.Resize(tuple(config["resolution"]["360p"]), antialias=True),
            transforms.ToTensor(),
        ]
    )

    dataset = MaskDataset(
        images_path=config["train"]["train_images_path"],
        mask_path=config["train"]["train_labels_path"],
        transform=my_transforms,
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    data_iter = iter(dataloader)
    images, masks = next(data_iter)

    image = images[0]
    mask = masks[0]

    print("Image Tensor Type:", type(image))
    print("Image Tensor Shape:", image.shape)

    print("Mask Tensor Type:", type(mask))
    print("Mask Tensor Shape:", mask.shape)

    # Convert the tensors to numpy arrays for plotting
    image_np = image.numpy().transpose(1, 2, 0)  # Convert to HWC format for plotting
    mask_np = mask.numpy().squeeze()

    # Plot the image and mask
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")

    plt.show()
