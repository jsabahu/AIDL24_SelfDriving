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
from modelDebug import SimpleSegmentationModel  # Debug Model

# from model_ENet_complete import ENet
from torchvision import transforms
from logger import Logger
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt

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
    model.train()  # Model in training mode
    accs, losses = [], []  # Init accuracies and losses
    for Image_cnt, (images, masks) in enumerate(train_loader):
        logger.log_info(f"Processing image {Image_cnt+1}/{len(train_loader)}")
        # Move data to device
        images, masks = images.to(device), masks.to(device)
        # Set network gradients to 0.
        optimizer.zero_grad()  # Restart gradients
        # Forward batch of images through the network
        output = model(images)
        # Reshape output & masks
        output = output.reshape(-1)
        masks = masks.reshape(-1)
        # Compute loss
        loss = F.binary_cross_entropy_with_logits(output, masks)  # Calculate loss
        # Backward pass: compute gradients of the loss with respect to model parameters
        loss.backward()
        # Update parameters of the network
        optimizer.step()
        # Compute metrics
        acc = binary_accuracy_with_logits(
            masks, output
        )  # Not clear! Function using like "Lab 3" from projects
        # Add loss to list
        losses.append(loss.item())
        # Add accuracy to list
        accs.append(acc.item())
    # Add loss and accuracy mean
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

        acc = binary_accuracy(lane_output_flat, masks_flat, threshold=0.5)

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


def train_model(hparams: dict, train_loader: DataLoader):
    """
    Train the model based on the provided hyperparameters.
    """
    logger.log_debug("Starting model training")

    # Initialize model
    model = ENet(
        num_classes=1,
        binary_output=True,
    ).to(device)

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
            best_model = model
            best_loss = train_loss
            save_model(best_model, config["train"]["model_name"])
            logger.log_debug(f"Saved best model with loss {best_loss:.2f} ")

    writer.flush()
    logger.log_debug("Training completed")
    return best_model


def train_model2(hparams, train_loader):
    # First trials, just one class: Line road??
    model = SimpleSegmentationModel().to(device)  # Just one class: "LINE ROAD"?<-
    logger.log_info("Train Model Called")
    # Initialize optimizer (possibility to create specific optimizer call from utils)
    # Self-drive examples found use the Adam optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],
    )  # Note use weight_decay to prevent overfitting
    logger.log_info("Train Optimizer")
    # Initialize Parameters
    best_loss = float("inf")  # Inicialitzar best_loss
    tr_loss, tr_acc = [], []

    num_epoch = hparams["num_epochs"]
    for epoch in range(num_epoch):
        # Train model for 1 epoch
        logger.log_info(f"Train Epoch {epoch+1}/{num_epoch}")
        train_loss, train_acc = train_single_epoch(model, train_loader, optimizer)
        logger.log_info(
            f"Train Epoch {epoch} loss={train_loss:.2f} acc={train_acc:.2f}"
        )
        # Save best lost
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), "best_model.pth")
            logger.log_info(f"Saved best model with loss {best_loss:.2f}")
        tr_loss.append(train_loss)
        tr_acc.append(train_acc)

    # Plot loast & Accuracy
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(tr_loss, label="train")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy [%]")
    plt.plot(tr_acc, label="train")
    plt.legend()
    plt.show()

    return model
