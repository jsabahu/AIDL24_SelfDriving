import torch
from torch.utils import data
from torchvision import transforms
from dataloader import MyDataset
from utils import read_yaml
from logger import Logger
from train import train_model
from utils import save_model


def main():
    hparams = {
        "batch_size": 64,
        "learning_rate": 0.05,
        "weight_decay": 0.1,
        "num_epochs": 5,
    }
    # Parameters to use
    config_path = "configs/config.yaml"
    config = read_yaml(config_path)

    # Create a logger object
    logger = Logger(log_file=config["main"]["log_filename"], level="debug")

    my_model = train_model(hparams=hparams)
    model_name = "test"

    save_model(
        model=my_model, model_name=model_name, path=config["main"]["saving_path"]
    )


if __name__ == "__main__":
    main()
