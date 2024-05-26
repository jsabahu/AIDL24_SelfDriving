import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F  # If required not use this black box, can create binary cross entropy in "utils"
import torch.optim as optim  # If required not use this black box, can create optimizer in "utils"
import numpy as np
from model2 import CustomBackbone  # Will define correct model name
from modelDebug import SimpleSegmentationModel  # Debug Model
from hyperparameters import hparams
from utils import binary_accuracy_with_logits
from logger import Logger
from utils import read_yaml
import matplotlib.pyplot as plt

# Parameters to use
config = (read_yaml("configs\config.yaml"))

# Create a logger object
logger = Logger(log_file=config['logger']["log_filename"], level="debug")

# Choose Device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Train model one epoch 
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
        loss = F.binary_cross_entropy_with_logits(output, masks) # Calculate loss
        # Backward pass: compute gradients of the loss with respect to model parameters
        loss.backward()
        # Update parameters of the network
        optimizer.step()
        # Compute metrics
        acc = binary_accuracy_with_logits(masks, output)  # Not clear! Function using like "Lab 3" from projects
        # Add loss to list
        losses.append(loss.item())
        # Add accuracy to list
        accs.append(acc.item())
    # Add loss and accuracy mean
    return np.mean(losses), np.mean(accs)

def train_model(hparams,train_loader):
    # First trials, just one class: Line road??
    model = SimpleSegmentationModel().to(device)  # Just one class: "LINE ROAD"?<-
    logger.log_info("Train Model Called")
    # Initialize optimizer (possibility to create specific optimizer call from utils)
    # Self-drive examples found use the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=hparams['learning_rate'],weight_decay=hparams['weight_decay']) #Note use weight_decay to prevent overfitting
    logger.log_info("Train Optimizer")
    #Initialize Parameters
    best_loss = float('inf')  # Inicialitzar best_loss
    tr_loss,tr_acc = [],[]

    num_epoch = hparams['num_epochs']
    for epoch in range(num_epoch):  
        # Train model for 1 epoch
        logger.log_info(f"Train Epoch {epoch+1}/{num_epoch}")
        train_loss, train_acc = train_single_epoch(model, train_loader, optimizer)
        logger.log_info(f"Train Epoch {epoch} loss={train_loss:.2f} acc={train_acc:.2f}")
        #Save best lost
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), 'best_model.pth')
            logger.log_info(f"Saved best model with loss {best_loss:.2f}")
        tr_loss.append(train_loss)
        tr_acc.append(train_acc)

    # Plot loast & Accuracy
    plt.figure(figsize=(10, 8))
    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(tr_loss, label='train')
    plt.legend()
    plt.subplot(2,1,2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy [%]')
    plt.plot(tr_acc, label='train')
    plt.legend()
    plt.show()

    return model

