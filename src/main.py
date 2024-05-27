import torch
from torch.utils import data
from torchvision import transforms
from dataloader import MyDataset, MyDataset2
from utils import read_yaml
from logger import Logger
from train import train_model, train_model2
from utils import save_model
from pathlib import Path
from hyperparameters import hparams

logger = Logger()

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
        "learning_rate": 0.05,
        "weight_decay": 0.01,
        "num_epochs": 5,
    }

    my_model = train_model(hparams=hparams)


def main_fermin():

    # Define Transform
    Transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert to tensor
            transforms.Resize(
                (
                    config["dataloader"]["resize_width"],
                    config["dataloader"]["resize_width"],
                ),
                antialias=True,
            ),  # Resize the image
        ]
    )

    # Create my_dataset
    my_dataset = MyDataset2(
        images_path=config["dataloader"]["images_path"],
        mask_path=config["dataloader"]["mask_path"],
        transform=Transform,
    )
    logger.log_info("Found " + str(len(my_dataset)) + " samples")

    # Split in train, eval and test data
    val_samples = int(0.40 * len(my_dataset))
    test_samples = int(0.40 * len(my_dataset))
    train_samples = len(my_dataset) - val_samples - test_samples

    # Create train, eval and test dataset
    train_dataset, val_dataset, test_dataset = data.random_split(
        dataset=my_dataset,
        lengths=[train_samples, val_samples, test_samples],
        generator=torch.Generator().manual_seed(42),
    )
    logger.log_info(
        "Samples split as follows: Train = "
        + str(train_samples)
        + " \ Validate = "
        + str(val_samples)
        + " \ Test = "
        + str(test_samples)
    )

    # Create train, eval and test dataloader
    train_loader = data.DataLoader(
        train_dataset, batch_size=config["dataloader"]["batch_size"], shuffle=True
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=config["dataloader"]["batch_size"]
    )
    test_loader = data.DataLoader(
        test_dataset, batch_size=config["dataloader"]["batch_size"]
    )

    logger.log_info("Start Trainning")
    my_model = train_model2(hparams, train_loader)
    logger.log_info("Train Finished")


if __name__ == "__main__":
    check = True
    if check:
        main_jordi()
    else:
        main_fermin()
