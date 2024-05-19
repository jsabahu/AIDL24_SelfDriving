import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F  # If required not use this black box, can create binary cross entropy in "utils"
import torch.optim as optim  # If required not use this black box, can create optimizer in "utils"
import numpy as np

from model import Model1  # Will define correct model name
from dataloader import MyDataset
from hyperparameters import hparams
from utils import binary_accuracy_with_logits

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
        loss = F.binary_cross_entropy_with_logits(detection_output, masks) + F.binary_cross_entropy_with_logits(lane_output, masks)  # Calculate loss
        # Backward pass: compute gradients of the loss with respect to model parameters
        loss.backward()
        # Update parameters of the network
        optimizer.step()
        # Compute metrics
        acc = binary_accuracy_with_logits(masks, detection_output)  # Not clear! Function using like "Lab 3" from projects
        # Add loss to list
        losses.append(loss.item())
        # Add accuracy to list
        accs.append(acc.item())
    # Add loss and accuracy mean
    return np.mean(losses), np.mean(accs)

def train_model(hparams):
    # Generate Training dataset
    train_dataset = MyDataset("train_images_path", "train_masks_path")  # ->REQUIRED UPDATE PATH<-
    # Generate DataLoader
    train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    # First trials, just one class: Line road??
    model = Model1(num_classes=1).to(device)  # Just one class: "LINE ROAD"?<-
    # Initialize optimizer (possibility to create specific optimizer call from utils)
    # Self-drive examples found use the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'],weight_decay=hparams['weight_decay']) #Note use weight_decay to prevent overfitting

    for epoch in range(hparams['num_epochs']):  
        # Train model for 1 epoch
        train_loss, train_acc = train_single_epoch(model, train_loader, optimizer)
        print(f"Train Epoch {epoch} loss={train_loss:.2f} acc={train_acc:.2f}")
    return model
