import torch
from torch.utils import data
from torchvision import transforms
from dataloader import MyDataset
from utils import read_yaml
from logger import Logger
from train import train_model
from utils import save_model
from pathlib import Path

logger = Logger()


def main():
    # Define hyperparameters
    hparams = {
        "batch_size": 64,
        "learning_rate": 0.05,
        "weight_decay": 0.1,
        "num_epochs": 5,
    }

    # Read configuration
    config_path = "configs/config.yaml"
    try:
        config = read_yaml(config_path)
        logger.log_debug(f"Read configuration from {config_path}")
    except Exception as e:
        logger.log_error(f"Failed to read configuration from {config_path}: {e}")
        raise

    my_model = train_model(hparams=hparams)


if __name__ == "__main__":
    main()
