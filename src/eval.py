import torch
import logging
from utils import read_yaml
from utils import accuracy, save_model
import torch.nn.functional as F  # If required not use this black box, can create binary cross entropy in "utils"
import numpy as np
from torcheval.metrics.functional import binary_accuracy

# Set up logging
logging.basicConfig(level=logging.INFO)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# Parameters to use
config_path = "configs/config.yaml"
config = read_yaml(config_path)


def eval_single_epoch(model, val_loader):
    accs, losses = [], []
    with torch.no_grad():
        model.eval()
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            y_ = model(x)
            loss = F.cross_entropy(y_, y)
            acc = accuracy(y, y_)
            losses.append(loss.item())
            accs.append(acc.item())
    return np.mean(losses), np.mean(accs)

def eval_mask_rCNN(model, hparams, eval_loader, rois, device):

    model.to(device)  # Move model to device
    # Model in train mode
    model.eval()

    # Initialize loss function
    criterion = F.binary_cross_entropy_with_logits

    # Initialize Parameters
    num_epoch = hparams["num_epochs"]
    ev_loss, ev_acc = [], []

    for epoch in range(num_epoch):
        # Train model for 1 epoch
        ev_loss_temp, ev_acc_temp = [], []
        for Image_cnt, (images, masks) in enumerate(eval_loader):
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
            # Compute metrics
            acc = binary_accuracy(output, masks, threshold=0.5)
            # Add loss & acc to list
            ev_loss_temp.append(loss.item())
            ev_acc_temp.append(acc.item())
        ev_mean_loss = np.mean(ev_loss_temp)
        ev_mean_acc = np.mean(ev_acc_temp)

        ev_loss.append(ev_mean_loss)
        ev_acc.append(ev_mean_acc)

    return np.mean(ev_loss), np.mean(ev_acc)