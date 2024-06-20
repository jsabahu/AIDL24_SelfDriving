import torch
from torch.utils import data
from torchvision import transforms
from dataloader import MaskDataset, TusimpleSet, Dataset_Mask_R_CNN
from utils import read_yaml
from logger import Logger
from train import train_model, train_mask_rCNN
from hyperparameters import hparams
from torch.utils.data import DataLoader
from models.model_ENet import ENet
from models.model_mask_R_CNN import LaneDetectionModel
import pandas as pd
import os
from utils import Rescale
from models.LaneNet.LaneNet import LaneNet
import time
from utils import generate_full_image_rois, show_sample
import yaml
import matplotlib.pyplot as plt
from eval import eval_mask_rCNN

logger = Logger()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.log_debug(f"Using device: {DEVICE}")

# Read configuration
config_path = "configs/config.yaml"
try:
    config = read_yaml(config_path)
    logger.log_debug(f"Read configuration from {config_path}")
except Exception as e:
    logger.log_error(f"Failed to read configuration from {config_path}: {e}")
    raise


def main_jordi():
    # Define hyperparameters
    hparams = {
        "batch_size": 16,
        "lr": 0.05,
        "weight_decay": 0.01,
        "num_epochs": 5,
    }

    train_dataset_file = os.path.join(
        config["dataset"]["tusimple"]["train"]["dir"], "train.txt"
    )
    val_dataset_path = os.path.join(
        config["dataset"]["tusimple"]["train"]["dir"], "val.txt"
    )

    resize_height = int(config["main"]["resize_height"])
    resize_width = int(config["main"]["resize_width"])

    # Define transformation
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

    # Create training dataset and DataLoader
    train_dataset = TusimpleSet(
        train_dataset_file,
        transform=data_transforms["train"],
        target_transform=target_transforms,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=hparams["batch_size"], shuffle=True
    )

    # Create validation dataset and DataLoader
    val_dataset = TusimpleSet(
        val_dataset_path,
        transform=data_transforms["val"],
        target_transform=target_transforms,
    )
    val_loader = DataLoader(val_dataset, batch_size=hparams["batch_size"], shuffle=True)

    model = LaneNet()
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])
    logger.log_debug(
        f"{hparams['num_epochs']} epochs {len(train_dataset)} training samples\n"
    )

    model, log = train_model(
        model,
        hparams=hparams,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=DEVICE,
    )

    df = pd.DataFrame({"epoch": [], "training_loss": [], "val_loss": []})
    df["epoch"] = log["epoch"]
    df["training_loss"] = log["training_loss"]
    df["val_loss"] = log["val_loss"]

    train_log_save_filename = os.path.join(
        config["main"]["logs_dir"], "training_log.csv"
    )
    df.to_csv(
        train_log_save_filename,
        columns=["epoch", "training_loss", "val_loss"],
        header=True,
        index=False,
        encoding="utf-8",
    )
    logger.log_debug("training log is saved: {}".format(train_log_save_filename))

    model_save_filename = f"models/Lane_Model_ENet_{time.time()}.pth"
    torch.save(model.state_dict(), model_save_filename)
    logger.log_debug("model is saved: {}".format(model_save_filename))


def main_mask_R_CNN():
    # Load hyperparameters from config file
    with open("configs/config.yaml", "r") as file:
        CONFIG = yaml.safe_load(file)

    # Create logger
    logger = Logger()

    # Define device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log_debug(f"Using device: {DEVICE}")

    # Define hyperparameters
    hparams = {
        "batch_size": 32,
        "lr": 0.0001,
        # "weight_decay": 0.1,
        "num_epochs": 100,
        "target_size": (480, 854),
    }

    # Define Transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert to tensor
            transforms.Resize(
                hparams["target_size"], antialias=True
            ),  # Resize the image
        ]
    )

    # Create training dataset and DataLoader
    train_dataset = Dataset_Mask_R_CNN(
        images_path=CONFIG["train"]["images_path"],
        mask_path=CONFIG["train"]["labels_path"],
        batch_size=hparams["batch_size"],
        transform=transform,
        transform_mask=transform,
    )

    # Create training dataset and DataLoader
    eval_dataset = Dataset_Mask_R_CNN(
        images_path=CONFIG["val"]["images_path"],
        mask_path=CONFIG["val"]["labels_path"],
        batch_size=hparams["batch_size"],
        transform=transform,
        transform_mask=transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=hparams["batch_size"], shuffle=True
    )

    eval_loader = DataLoader(
        eval_dataset, batch_size=hparams["batch_size"], shuffle=True
    )

    logger.log_info("Found train " + str(len(train_dataset)) + " samples")
    logger.log_info("Found eval " + str(len(eval_dataset)) + " samples")

    # Create Model
    rois = generate_full_image_rois((hparams["batch_size"]), hparams["target_size"]).to(
        device=DEVICE
    )
    model = LaneDetectionModel()

    # Train Model
    tr_loss, tr_acc = train_mask_rCNN(model, hparams, train_loader, rois, DEVICE)
    save_model(model, "train_mask_rCNN_100k.pth")
    # Eval Model
    eval_loss, eval_acc = eval_mask_rCNN(model, hparams, eval_loader, rois, DEVICE)
    logger.log_info("Eval Loss: " + str(eval_loss) + " / Eval Acc:" + str(eval_acc))

    # Plot loast & Accuracy
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(tr_loss, label="loss train")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy [%]")
    plt.plot(tr_acc, label="acc train")
    plt.legend()
    plt.show()

    image_path = "data\\bdd100k\\images\\100k\\val\\fdc07c32-1af45031.jpg"
    mask_path = "data\\bdd100k\\labels\\lane\\masks\\val\\fdc07c32-1af45031.png"
    show_sample(
        model,
        image_path,
        mask_path,
        generate_full_image_rois(1, hparams["target_size"]),
        DEVICE,
    )


if __name__ == "__main__":
    check = False
    if check:
        main_jordi()
    else:
        main_mask_R_CNN()
