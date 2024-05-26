import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F  # If required not use this black box, can create binary cross entropy in "utils"
import torch.optim as optim  # If required not use this black box, can create optimizer in "utils"
import numpy as np
from torcheval.metrics.functional import binary_accuracy
from utils import read_yaml
from dataloader import MyDataset
from hyperparameters import hparams
from utils import binary_accuracy_with_logits
from model_ENet import ENet

# from model_ENet_complete import ENet
from torchvision import transforms
from logger import Logger
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

logger = Logger()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Parameters to use
config_path = "configs/config.yaml"
config = read_yaml(config_path)

# Train model one epoch
def train_single_epoch(model, train_loader, optimizer):
    model.train()  # Model in training mode
    accs, losses = [], []  # Init accuracies and losses
    for step, (images, masks) in enumerate(train_loader):
        # Move data to device
        images, masks = images.to(device), masks.to(device)
        # Set network gradients to 0.
        optimizer.zero_grad()  # Restart gradients
        # Forward batch of images through the network
        detection_output, lane_output = model(images)
        # Adjust size masks
        masks = masks.unsqueeze(1).float()
        # Compute loss
        loss = F.binary_cross_entropy_with_logits(
            detection_output, masks
        ) + F.binary_cross_entropy_with_logits(
            lane_output, masks
        )  # Calculate loss
        # Backward pass: compute gradients of the loss with respect to model parameters
        loss.backward()
        # Update parameters of the network
        optimizer.step()
        # Compute metrics
        acc = binary_accuracy_with_logits(
            masks, detection_output
        )  # Not clear! Function using like "Lab 3" from projects
        # Add loss to list
        losses.append(loss.item())
        # Add accuracy to list
        accs.append(acc.item())
    # Add loss and accuracy mean
    return np.mean(losses), np.mean(accs)


def train_model(hparams):
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256), antialias=True),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5],std=[0.5, 0.5])  # Resize to [3, 256, 256]
        ]
    )
    # Generate Training dataset
    train_dataset = MyDataset(
        images_path=config["train"]["train_images_path"],
        mask_path=config["train"]["train_labels_path"],
        transform=transform,
    )  # ->REQUIRED UPDATE PATH<-
    # Generate DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=hparams["batch_size"], shuffle=True
    )
    # First trials, just one class: Line road??
    model = ENet(num_classes=1).to(device)  # Just one class: "LINE ROAD"?<-
    # Initialize optimizer (possibility to create specific optimizer call from utils)
    # Self-drive examples found use the Adam optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=hparams["learning_rate"],
        weight_decay=hparams["weight_decay"],
    )  # Note use weight_decay to prevent overfitting
    # Initialize best_loss
    best_loss = float("inf")  # Inicialitzar best_loss

    for epoch in range(hparams["num_epochs"]):
        # Train model for 1 epoch
        train_loss, train_acc = train_single_epoch_laneModel(
            model, train_loader, optimizer
        )
        logger.log_debug(
            f"Train Epoch {epoch} loss={train_loss:.2f} acc={train_acc:.2f}"
        )
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        # Save best lost
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), "best_model.pth")
            logger.log_debug(f"Saved best model with loss {best_loss:.2f}")
    return model


def train_single_epoch_laneModel(model, train_loader, optimizer):
    logger.log_debug("Starting training single epoch")
    model.train()
    accs, losses = [], []
    for step, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        lane_output = model(images.float())
        masks = masks.float()
        loss = F.cross_entropy(lane_output, masks)
        loss.backward()
        optimizer.step()
        acc = binary_accuracy(lane_output, masks, threshold=0.9)
        losses.append(loss.item())
        accs.append(acc.item())
    return np.mean(losses), np.mean(accs)


writer.flush()
