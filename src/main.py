import torch
from torch.utils import data
from torchvision import transforms
from dataloader import MaskDataset
from utils import read_yaml
from logger import Logger
from train import train_model, train_model2
from utils import save_model
from pathlib import Path
from hyperparameters import hparams
from torch.utils.data import DataLoader
from models.model_ENet import ENet
from models.model2 import LaneDetectionModel
import pandas as pd
import os

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

    # Define transformation
    transform = transforms.Compose(
        [
            transforms.Resize((360, 640), antialias=True),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Create training dataset and DataLoader
    train_dataset = MaskDataset(
        images_path=config["train"]["images_path"],
        mask_path=config["train"]["labels_path"],
        transform=transform,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=hparams["batch_size"], shuffle=True
    )

    # Create validation dataset and DataLoader
    val_dataset = MaskDataset(
        images_path=config["val"]["images_path"],
        mask_path=config["val"]["labels_path"],
        transform=transform,
    )
    val_loader = DataLoader(val_dataset, batch_size=hparams["batch_size"], shuffle=True)

    model = ENet(num_classes=1)
    model.to(device=DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"])
    logger.log_debug(
        f"{hparams['num_epochs']} epochs {len(train_dataset)} training samples\n"
    )

    model, log = train_model(
        model, hparams, train_loader, val_loader, optimizer, DEVICE
    )  # add loss types, or mor arguments

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

    model_save_filename = config["main"]["save_model_path"]
    torch.save(model.state_dict(), model_save_filename)
    logger.log_debug("model is saved: {}".format(model_save_filename))


def main_fermin():

    # Define hyperparameters
    hparams = {
        "batch_size": 16,
        "lr": 0.05,
        "weight_decay": 0.01,
        "num_epochs": 5,
    }
    
    # Define Transform
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert to tensor
            transforms.Resize((320, 180),antialias=True),  # Resize the image
        ]
    )

    # Create training dataset and DataLoader
    train_dataset = MaskDataset(
        images_path=config["train"]["images_path"],
        mask_path=config["train"]["labels_path"],
        transform=transform,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=hparams["batch_size"], shuffle=True
    )

    logger.log_info("Found train " + str(len(train_dataset)) + " samples")

    # Create validation dataset and DataLoader
    val_dataset = MaskDataset(
        images_path=config["val"]["images_path"],
        mask_path=config["val"]["labels_path"],
        transform=transform,
    )
    val_loader = DataLoader(val_dataset, batch_size=hparams["batch_size"], shuffle=True)

    logger.log_info("Found train " + str(len(val_dataset)) + " samples")

    # Create Model
    model = LaneDetectionModel()
    model.to(device=DEVICE)
    model = train_model2(hparams, train_loader, DEVICE)

if __name__ == "__main__":
    check = False
    if check:
        main_jordi()
    else:
        main_fermin()
