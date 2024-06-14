import os
from torch.utils.data import Dataset, DataLoader
import pathlib
from PIL import Image
from torchvision import transforms
from utils import read_yaml
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from utils import Rescale
import random

CONFIG_PATH = "configs/config.yaml"


class MaskDataset(Dataset):
    def __init__(self, images_path, mask_path, transform=None, target_transform=None):
        super().__init__()
        self._root_folder = pathlib.Path(__file__).parent.parent
        self.images_path = os.path.join(self._root_folder, images_path)
        self.mask_path = os.path.join(self._root_folder, mask_path)
        self.transform = transform
        self.target_transform = target_transform

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
        if self.target_transform:
            mask = self.target_transform(
                mask
            )  # Es correcte aplicar totes les transformacions a la mask?????

        mask = (mask == 1).type(torch.int)

        return image, mask


class Dataset_Mask_R_CNN(Dataset):
    def __init__(
        self, images_path, mask_path, batch_size, transform=None, transform_mask=None
    ):
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

    def _epoch_multiple(self, batch_size):
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


class TusimpleSet(Dataset):
    def __init__(self, dataset, n_labels=3, transform=None, target_transform=None):
        self._gt_img_list = []
        self._gt_label_binary_list = []
        self._gt_label_instance_list = []
        self.transform = transform
        self.target_transform = target_transform
        self.n_labels = n_labels

        with open(dataset, "r") as file:
            for _info in file:
                info_tmp = _info.strip(" ").split()

                self._gt_img_list.append(info_tmp[0])
                self._gt_label_binary_list.append(info_tmp[1])
                self._gt_label_instance_list.append(info_tmp[2])

        assert (
            len(self._gt_img_list)
            == len(self._gt_label_binary_list)
            == len(self._gt_label_instance_list)
        )

        self._shuffle()

    def _shuffle(self):
        # randomly shuffle all list identically
        c = list(
            zip(
                self._gt_img_list,
                self._gt_label_binary_list,
                self._gt_label_instance_list,
            )
        )
        random.shuffle(c)
        (
            self._gt_img_list,
            self._gt_label_binary_list,
            self._gt_label_instance_list,
        ) = zip(*c)

    def __len__(self):
        return len(self._gt_img_list)

    def __getitem__(self, idx):
        assert (
            len(self._gt_label_binary_list)
            == len(self._gt_label_instance_list)
            == len(self._gt_img_list)
        )

        # load all
        img = Image.open(self._gt_img_list[idx])
        label_instance_img = cv2.imread(
            self._gt_label_instance_list[idx], cv2.IMREAD_UNCHANGED
        )
        label_img = cv2.imread(self._gt_label_binary_list[idx], cv2.IMREAD_COLOR)

        # optional transformations
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label_img = self.target_transform(label_img)
            label_instance_img = self.target_transform(label_instance_img)

        label_binary = np.zeros(
            [label_img.shape[0], label_img.shape[1]], dtype=np.uint8
        )
        mask = np.where((label_img[:, :, :] != [0, 0, 0]).all(axis=2))
        label_binary[mask] = 1

        # we could split the instance label here, each instance in one channel (basically a binary mask for each)
        return img, label_binary, label_instance_img


def plot_images(image, mask, mask_instance=None):
    image_np = image.numpy().transpose(1, 2, 0)
    mask_np = mask.numpy().squeeze()

    if mask_instance is not None:
        mask_instance_np = mask_instance.numpy().squeeze()
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[2].imshow(mask_instance_np, cmap="gray")
        axes[2].set_title("Mask Instance")
        axes[2].axis("off")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")

    plt.show()


if __name__ == "__main__":
    config = read_yaml(CONFIG_PATH)
    example_dataset = ["Tusimple", "Maskbdd100k"]
    dataset = example_dataset[0]

    resize_height = int(config["main"]["resize_height"])
    resize_width = int(config["main"]["resize_width"])

    if dataset == "Tusimple":

        data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.Resize((resize_height, resize_width)),
                    transforms.ColorJitter(
                        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize((resize_height, resize_width)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        }

        target_transforms = transforms.Compose(
            [
                Rescale((resize_width, resize_height)),
            ]
        )

        train_dataset_file = os.path.join(
            config["dataset"]["tusimple"]["train"]["dir"], "train.txt"
        )
        val_dataset_path = os.path.join(
            config["dataset"]["tusimple"]["train"]["dir"], "val.txt"
        )

        train_dataset = TusimpleSet(
            train_dataset_file,
            transform=data_transforms["train"],
            target_transform=target_transforms,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=config["dataloader"]["batch_size"], shuffle=True
        )

        data_iter = iter(train_loader)
        img, label_binary, label_instance_img = next(data_iter)

        image = img[0]
        mask = label_binary[0]
        mask_instance = label_instance_img[0]

        print("Image Tensor Type:", type(image))
        print("Image Tensor Shape:", image.shape)
        print("Mask Tensor Type:", type(label_binary))
        print("Mask Tensor Shape:", mask.shape)
        print("Mask Instance Tensor Type:", type(label_instance_img))
        print("Mask Instance Tensor Shape:", mask_instance.shape)

        plot_images(img[0], label_binary[0], label_instance_img[0])

    elif dataset == "Maskbdd100k":

        data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.Resize((resize_height, resize_width)),
                    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize((resize_height, resize_width)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            ),
        }

        target_transforms = transforms.Compose(
            [
                transforms.Resize((resize_height, resize_width)),
                transforms.ToTensor(),
            ]
        )

        dataset = MaskDataset(
            images_path=config["dataset"]["bdd100k"]["train"]["images_path"],
            mask_path=config["dataset"]["bdd100k"]["train"]["labels_path"],
            transform=data_transforms["train"],
            target_transform=target_transforms,
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

        plot_images(image, mask)
    else:
        print("Select good dataset name")
