import os
from torch.utils import data
import pathlib
from PIL import Image


class MyDataset(data.Dataset):
    # Call the super().__init__() method in the __init__.
    def __init__(self, images_path, mask_path, transform=None):
        super().__init__()

        self.root_folder = pathlib.Path(__file__).parent.parent
        self.images_path = os.path.join(self.root_folder, images_path)
        self.mask_path = os.path.join(self.root_folder, mask_path)
        self.transform = transform
        self.image_files = sorted(os.listdir(self.images_path))

    # Implement a __len__ method with the length of the dataset.
    def __len__(self):
        # Show only jpg files
        files = [file for file in self.image_files if file.lower().endswith(".jpg")]
        return len(files)

    # Implement a __getitem__ method that returns a transformed image and mask.
    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(self.images_path, image_name)
        mask_name = f"{image_name.split('.')[0]}.png"
        mask_path = os.path.join(
            self.mask_path, mask_name
        )  # Assuming mask has the same filename as image

        # Open image and mask for index element
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # Apply transform if need
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask
