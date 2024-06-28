import os
import torch
import torchvision
import json
import yaml
from utils import save_model
from logger import Logger
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    FasterRCNN_ResNet50_FPN_Weights,
)
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Initialize tensorboard writer and logger
writer = SummaryWriter()
logger = Logger()


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def create_transforms(config):
    resize_height = config["transforms"]["resize_height"]
    resize_width = config["transforms"]["resize_width"]
    antialias = config["transforms"]["antialias"]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (resize_height, resize_width), antialias=antialias
            ),  # Explicitly set antialias from config
        ]
    )
    return transform


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


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return [], []
    return tuple(zip(*batch))


def create_data_loader(config, dataset):
    batch_size = config["hyperparameters"]["batch_size"]
    shuffle = config["hyperparameters"]["shuffle"]
    num_workers = config["hyperparameters"]["num_workers"]

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return data_loader


def create_model(num_classes):
    # Load a pre-trained model for classification and return only the features
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    )

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_model(config, model, data_loader, device):
    learning_rate = config["hyperparameters"]["learning_rate"]
    momentum = config["hyperparameters"]["momentum"]
    weight_decay = config["hyperparameters"]["weight_decay"]
    step_size = config["hyperparameters"]["step_size"]
    gamma = config["hyperparameters"]["gamma"]
    num_epochs = config["hyperparameters"]["num_epochs"]
    accumulation_steps = config["hyperparameters"]["accumulation_steps"]

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

            writer.add_scalar(f"Loss train", losses.item(), epoch)

        lr_scheduler.step()

        logger.log_info(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item()}")
    
    writer.flush()


def evaluate_model(model, data_loader, device):
    model.eval()

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Perform the prediction
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            writer.add_scalar(f"Loss eval", losses.item())

            # Here you can add code to calculate the evaluation metrics like mAP

    torch.cuda.empty_cache()

    logger.log_info(f"Eval Loss: {np.mean(losses)}")   

    writer.flush()
    torch.cuda.empty_cache()


def main():
    # Load configuration
    config = load_config()

    # Paths from config
    root = config["dataset"]["dataset_Faster_R_CNN"]["root"]
    annotation_file = config["dataset"]["dataset_Faster_R_CNN"]["annotation_file"]

    # Transformations
    transform = create_transforms(config)

    # Dataset and DataLoader
    dataset = BDDDataset(root, annotation_file, transforms=transform)
    data_loader = create_data_loader(config, dataset)

    # Model
    num_classes = config["hyperparameters"]["num_classes"]
    model = create_model(num_classes)

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Train
    train_model(config, model, data_loader, device)

    save_model(model, "Faster_R_CNN.pth")

    # Evaluate
    evaluate_model(model, data_loader, device)

if __name__ == "__main__":
    main()
