import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torcheval.metrics.functional import binary_accuracy
from utils import read_yaml
from dataloader import MyDataset
from hyperparameters import hparams
from utils import binary_accuracy_with_logits, save_model
from model_ENet import ENet

# from model_ENet_complete import ENet
from torchvision import transforms
from logger import Logger
from torch.utils.tensorboard import SummaryWriter
import os

# Initialize tensorboard writer and logger
writer = SummaryWriter()
logger = Logger()

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.log_debug(f"Using device: {device}")

# Read configuration
config_path = "configs/config.yaml"
try:
    config = read_yaml(config_path)
    logger.log_debug(f"Read configuration from {config_path}")
except Exception as e:
    logger.log_error(f"Failed to read configuration from {config_path}: {e}")
    raise


def train_single_epoch(model, train_loader, optimizer):
    """
    Train the model for a single epoch.
    """

    logger.log_debug("Starting training for a single epoch")
    model.train()  # Set model to training mode
    accs, losses = [], []  # Initialize accuracies and losses

    for step, (images, masks) in enumerate(train_loader):
        logger.log_debug(f"Processing step {step}")
        # Move data to device
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        detection_output, lane_output = model(images)
        masks = masks.unsqueeze(1).float()  # Adjust size masks

        # Compute loss
        loss = F.binary_cross_entropy_with_logits(
            detection_output, masks
        ) + F.binary_cross_entropy_with_logits(lane_output, masks)

        # Backward pass and update
        loss.backward()
        optimizer.step()

        # Compute metrics
        acc = binary_accuracy_with_logits(masks, detection_output)
        losses.append(loss.item())
        accs.append(acc.item())

    return np.mean(losses), np.mean(accs)


def train_single_epoch_lane_model(model, train_loader, optimizer):
    """
    Train the lane detection model for a single epoch.
    """
    logger.log_debug("Starting training single epoch (Lane Model)")

    model.train()  # Set model to training mode
    accs, losses = [], []  # Initialize accuracies and losses

    for step, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        lane_output = model(images.float())
        masks = masks.float()

        # Compute loss
        loss = F.binary_cross_entropy_with_logits(lane_output, masks)

        # Backward pass and update
        loss.backward()
        optimizer.step()

        # Flatten the output and target tensors
        lane_output_flat = lane_output.view(-1)
        masks_flat = masks.view(-1)
        acc = binary_accuracy(lane_output_flat, masks_flat, threshold=0.9)

        logger.log_debug(
            f"Step {step}/{len(train_loader)}: loss={loss.item()}, acc={acc.item()}"
        )
        losses.append(loss.item())
        accs.append(acc.item())

    mean_loss = np.mean(losses)
    mean_acc = np.mean(accs)
    logger.log_debug(
        f"Epoch completed with mean loss: {mean_loss}, mean accuracy: {mean_acc}"
    )

    return mean_loss, mean_acc


def train_model(hparams):
    """
    Train the model based on the provided hyperparameters.
    """
    logger.log_debug("Starting model training")

    # Define transformation
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256), antialias=True),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5],std=[0.5, 0.5]),
        ]
    )

    # Create training dataset and DataLoader
    train_dataset = MyDataset(
        images_path=config["train"]["train_images_path"],
        mask_path=config["train"]["train_labels_path"],
        transform=transform,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=hparams["batch_size"], shuffle=True
    )

    # Initialize model
    model = ENet(num_classes=1).to(device)

    # Initialize optimizer
    # Self-drive examples found use the Adam optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],
    )

    # Initialize best_loss
    best_loss = float("inf")  # Inicialitzar best_loss

    for epoch in range(hparams["num_epochs"]):
        logger.log_debug(f"Starting epoch {epoch}")

        # Train for one epoch
        train_loss, train_acc = train_single_epoch_lane_model(
            model, train_loader, optimizer
        )

        logger.log_debug(
            f"Train Epoch {epoch} loss={train_loss:.2f} acc={train_acc:.2f}"
        )
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)

        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            save_model(model, config["train"]["model_name"])
            logger.log_debug(f"Saved best model with loss {best_loss:.2f}")

    writer.flush()
    logger.log_debug("Training completed")
    return model
