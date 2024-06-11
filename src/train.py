import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torcheval.metrics.functional import binary_accuracy
from utils import read_yaml
from hyperparameters import hparams
from utils import binary_accuracy_with_logits, save_model
from models.model_ENet import ENet
from models.modelDebug import SimpleSegmentationModel  # Debug Model
from torchvision import transforms
from logger import Logger
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from time import time
import copy
from loss import FocalLoss, DiscriminativeLoss
import torch.nn as nn

# Initialize tensorboard writer and logger
writer = SummaryWriter()
logger = Logger()

# Read configuration
config_path = "configs/config.yaml"
try:
    config = read_yaml(config_path)
    logger.log_debug(f"Read configuration from {config_path}")
except Exception as e:
    logger.log_error(f"Failed to read configuration from {config_path}: {e}")
    raise


def train_single_epoch(model, train_loader, optimizer, device):
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
        # loss = F.binary_cross_entropy_with_logits(output, masks)  # Calculate loss
        loss = compute_loss(output, binary_label=masks)
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


def single_epoch_lane_model(model, dataloader, optimizer, device, phase):
    """
    Train or val the lane detection model for a single epoch.
    """
    logger.log_debug(f"Starting {phase} single epoch (Lane Model)")

    if phase == "train":
        model.train()
    elif phase == "val":
        model.eval()

    accs, losses = [], []  # Initialize accuracies and losses

    for step, (images, masks) in enumerate(dataloader):
        images = images.type(torch.FloatTensor).to(device)
        masks = masks.type(torch.LongTensor).to(device)

        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        # track history if only in train
        with torch.set_grad_enabled(phase == "train"):
            outputs = model(images)
            # loss = F.binary_cross_entropy_with_logits(outputs, masks)
            loss = compute_loss(outputs, binary_label=masks)

            # backward + optimize only if in training phase
            if phase == "train":
                loss.backward()
                optimizer.step()

        # Flatten the output and target tensors
        outputs_flat = outputs.view(-1)
        masks_flat = masks.view(-1)

        acc = binary_accuracy(outputs_flat, masks_flat, threshold=0.5)

        logger.log_debug(
            f"{phase} phase: Step {step}/{len(dataloader)}: loss={loss.item()}, acc={acc.item()}"
        )

        losses.append(loss.item())
        accs.append(acc.item())

    mean_loss = np.mean(losses)
    mean_acc = np.mean(accs)
    logger.log_debug(
        f"{phase} phase --> Epoch completed with mean loss: {mean_loss}, mean accuracy: {mean_acc}"
    )

    return mean_loss, mean_acc


def train_model(
    model: torch.nn.Module,
    hparams: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    device,
):
    t1 = time()
    logger.log_debug("Starting model training")
    training_log = {"epoch": [], "training_loss": [], "val_loss": []}
    best_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    num_epochs = hparams["num_epochs"]

    for epoch in range(num_epochs):
        training_log["epoch"].append(epoch)
        logger.log_debug("Epoch {}/{}".format(epoch, num_epochs - 1))
        logger.log_debug("-" * 10)

        for phase in ["train", "val"]:

            if phase == "train":
                loss, acc = single_epoch_lane_model(
                    model=model,
                    dataloader=train_loader,
                    optimizer=optimizer,
                    device=device,
                    phase=phase,
                )
            elif phase == "val":
                loss, acc = single_epoch_lane_model(
                    model=model,
                    dataloader=val_loader,
                    optimizer=optimizer,
                    device=device,
                    phase=phase,
                )

            logger.log_debug(f"{phase} Epoch {epoch} loss={loss:.2f} acc={acc:.2f}")

            writer.add_scalar(f"Loss/{phase}", loss, epoch)
            writer.add_scalar(f"Acc/{phase}", acc, epoch)

            # deep copy the model
            if phase == "train":
                training_log["training_loss"].append(loss)
            if phase == "val":
                training_log["val_loss"].append(loss)
                if loss < best_loss:
                    best_loss = loss
                    best_model_wts = copy.deepcopy(model.state_dict())

    t2 = time() - t1

    writer.flush()
    logger.log_debug("Training complete in {:.0f}m {:.0f}s".format(t2 // 60, t2 % 60))
    logger.log_debug("Best val_loss: {:4f}".format(best_loss))
    training_log["training_loss"] = np.array(training_log["training_loss"])
    training_log["val_loss"] = np.array(training_log["val_loss"])

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, training_log


def train_model2(hparams, train_loader, device):
    # First trials, just one class: Line road??
    model = SimpleSegmentationModel().to(device)  # Just one class: "LINE ROAD"?<-
    logger.log_info("Train Model Called")
    # Initialize optimizer (possibility to create specific optimizer call from utils)
    # Self-drive examples found use the Adam optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=hparams["lr"],
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
        train_loss, train_acc = train_single_epoch(
            model, train_loader, optimizer, device
        )
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


def compute_loss(output, binary_label, loss_type="FocalLoss"):
    k_binary = 10  # 1.7

    print(binary_label.shape)

    if loss_type == "FocalLoss":
        loss_fn = FocalLoss(gamma=2, alpha=0.25)
    elif loss_type == "CrossEntropyLoss":
        loss_fn = nn.CrossEntropyLoss()
    else:
        logger.log_warning("Wrong loss type, will use the default CrossEntropyLoss")
        loss_fn = nn.CrossEntropyLoss()

    binary_loss = loss_fn(output, binary_label)

    binary_loss = binary_loss * k_binary

    return binary_loss


def binary_accuracy(output, masks, threshold=0.5):
    preds = (output >= threshold).float()
    correct = (preds == masks).float()
    acc = correct.sum() / len(correct)
    return acc


def train_mask_rCNN(model, hparams, train_loader, rois, device):

    model.to(device)  # Move model to device
    # Model in train mode
    model.train()

    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=hparams["lr"],
        # weight_decay=hparams["weight_decay"],  # Uncomment if weight_decay is needed
    )
    logger.log_info("Train Optimizer")

    # Initialize loss function
    criterion = F.binary_cross_entropy_with_logits

    # Initialize Parameters
    num_epoch = hparams["num_epochs"]
    tr_loss, tr_acc = [], []

    for epoch in range(num_epoch):
        # Train model for 1 epoch
        logger.log_info(f"Train Epoch {epoch+1}/{num_epoch}")
        tr_loss_temp, tr_acc_temp = [], []
        for Image_cnt, (images, masks) in enumerate(train_loader):
            logger.log_info(f"Processing image {Image_cnt+1}/{len(train_loader)}")
            # Set network gradients to 0.
            optimizer.zero_grad()  # Restart gradients
            # Move data to device
            images, masks = images.to(device), masks.to(device)
            # Forward batch of images through the network
            output = model(images, rois)
            # Define the weights for the RGB to grayscale conversion
            weights = torch.tensor([0.2989, 0.5870, 0.1140], device=device).view(
                1, 3, 1, 1
            )
            # Apply the weights and sum across the channel dimension
            output = (output * weights).sum(dim=1, keepdim=True)
            # Reshape output & masks
            output = F.interpolate(
                output, size=hparams["target_size"], mode="bilinear", align_corners=False
            )
            output = output.reshape(-1).type(torch.float)
            masks = masks.reshape(-1).type(torch.float)  # Convert masks to float
            # Compute loss
            loss = criterion(output, masks)
            # Backward pass: compute gradients of the loss with respect to model parameters
            loss.backward()
            # Update parameters of the network
            optimizer.step()
            # Compute metrics
            acc = binary_accuracy(output, masks, threshold=0.5)
            # Add loss & acc to list
            tr_loss_temp.append(loss.item())
            tr_acc_temp.append(acc.item())
        tr_mean_loss = np.mean(tr_loss_temp)
        tr_mean_acc = np.mean(tr_acc_temp)
        logger.log_info(
            f"Train Epoch {epoch+1} loss={tr_mean_loss:.2f} acc={tr_mean_acc:.2f}"
        )
        tr_loss.append(tr_mean_loss)
        tr_acc.append(tr_mean_acc)

    return tr_loss, tr_acc
