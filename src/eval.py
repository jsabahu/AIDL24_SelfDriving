import torch
import logging
from utils import read_yaml
from utils import accuracy, save_model
import torch.nn.functional as F  # If required not use this black box, can create binary cross entropy in "utils"
import numpy as np


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
