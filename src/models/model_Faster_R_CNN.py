#################### Load & Data Preprocess #####################################
import os
import torch
import torchvision
import json
import yaml
from logger import Logger
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Create logger
logger = Logger()

# Load hyperparameters from config file
with open("configs/config.yaml", "r") as file:
    CONFIG = yaml.safe_load(file)

# Paths from config
root = CONFIG["dataset"]["dataset_Faster_R_CNN"]["root"]
annotation_file = CONFIG["dataset"]["dataset_Faster_R_CNN"]["annotation_file"]

# Hyperparameters from config
batch_size = CONFIG["hyperparameters"]["batch_size"]
shuffle = CONFIG["hyperparameters"]["shuffle"]
num_workers = CONFIG["hyperparameters"]["num_workers"]
num_classes = CONFIG["hyperparameters"]["num_classes"]
learning_rate = CONFIG["hyperparameters"]["learning_rate"]
momentum = CONFIG["hyperparameters"]["momentum"]
weight_decay = CONFIG["hyperparameters"]["weight_decay"]
step_size = CONFIG["hyperparameters"]["step_size"]
gamma = CONFIG["hyperparameters"]["gamma"]
num_epochs = CONFIG["hyperparameters"]["num_epochs"]
accumulation_steps = CONFIG["hyperparameters"]["accumulation_steps"]

# Transformations from config
resize_height = CONFIG["transforms"]["resize_height"]
resize_width = CONFIG["transforms"]["resize_width"]
antialias = CONFIG["transforms"]["antialias"]


class BDDDataset(Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        with open(annotation_file) as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.root, ann["name"])

        if not os.path.exists(img_path):
            logger.log_debug(f"File not found: {img_path}. Skipping...")
            return None

        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        for obj in ann["labels"]:
            if obj["category"] == "car":
                x_min, y_min, x_max, y_max = (
                    obj["box2d"]["x1"],
                    obj["box2d"]["y1"],
                    obj["box2d"]["x2"],
                    obj["box2d"]["y2"],
                )
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(1)  # Car label

        if len(boxes) == 0:
            return None

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            img = self.transforms(img)

        return img, target


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(
            (resize_height, resize_width), antialias=antialias
        ),  # Explicitly set antialias from config
    ]
)

dataset = BDDDataset(root, annotation_file, transforms=transform)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return [], []
    return tuple(zip(*batch))


data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers,
    collate_fn=collate_fn,
)

from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    FasterRCNN_ResNet50_FPN_Weights,
)

# Load a pre-trained model for classification and return only the features
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
)

# Get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

import torch.optim as optim

# Move model to the right device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(
    params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=step_size, gamma=gamma
)

# Training loop with gradient accumulation
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    for i, (images, targets) in enumerate(data_loader):
        if len(images) == 0:
            continue

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if any(len(t["boxes"]) == 0 for t in targets):
            continue

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Normalize the loss by the accumulation steps
        losses = losses / accumulation_steps
        losses.backward()

        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    lr_scheduler.step()

    logger.log_info(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item()}")

torch.cuda.empty_cache()

model.eval()
with torch.no_grad():
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Perform the prediction
        predictions = model(images)

        # Here you can add code to calculate the evaluation metrics like mAP

torch.cuda.empty_cache()
