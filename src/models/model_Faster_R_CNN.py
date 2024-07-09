import os
import torch
from torchvision.ops import box_iou
import torchvision
import json
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor,
    FasterRCNN_ResNet50_FPN_Weights,
)
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from utils import save_model
from logger import Logger
from torchmetrics.detection.mean_ap import MeanAveragePrecision

"""
# Simple Logger class implementation
class Logger:
    def __init__(self, log_file="training.log"):
        self.log_file = log_file

    def log_info(self, message):
        self._log_message("[INFO]", message)

    def log_debug(self, message):
        self._log_message("[DEBUG]", message)

    def _log_message(self, level, message):
        log_message = f"{level}: {message}"
        print(log_message)  # Print to console
        with open(self.log_file, "a") as file:
            file.write(log_message + "\n")  # Write to file
"""

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
            transforms.Resize((resize_height, resize_width), antialias=antialias),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            ),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(
                (resize_height, resize_width), scale=(0.8, 1.0), antialias=antialias
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform


class BDDDataset(Dataset):
    def __init__(self, root, annotation_file, transforms=None, filter_annotations=True):
        self.root = root
        self.transforms = transforms
        self.filter_annotations = filter_annotations
        with open(annotation_file) as f:
            self.annotations = json.load(f)
        if self.filter_annotations:
            self.annotations = self._filter_annotations(self.annotations)
        logger.log_info(
            f"Filtered dataset to {len(self.annotations)} valid annotations."
        )

    def _filter_annotations(self, annotations):
        valid_annotations = []
        for ann in annotations:
            if "labels" in ann and any(
                obj["category"] == "car" for obj in ann["labels"]
            ):
                valid_annotations.append(ann)
            # else:
            # logger.log_debug(
            #    f"Skipping invalid annotation: {ann['name'] if 'name' in ann else 'unknown'}"
            # )
        return valid_annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if idx >= len(self.annotations):
            raise IndexError("Index out of range")

        ann = self.annotations[idx]
        img_path = os.path.join(self.root, ann["name"])

        if not os.path.exists(img_path):
            logger.log_debug(f"File not found: {img_path}. Skipping...")
            return None

        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        if "labels" in ann:
            for obj in ann["labels"]:
                if not self.filter_annotations or obj["category"] == "car":
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

        if torch.isnan(img).any():
            logger.log_debug(f"NaN detected in image: {img_path}. Skipping...")
            return None

        if torch.isnan(img).any():
            logger.log_debug(f"NaN detected in image: {img_path}. Skipping...")
            return None

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


def train_model(config, model, train_loader, val_loader, device):
    learning_rate = (
        config["hyperparameters"]["learning_rate"] * 0.1
    )  # Lower initial learning rate
    weight_decay = config["hyperparameters"]["weight_decay"]
    num_epochs = config["hyperparameters"]["num_epoch"]
    accumulation_steps = 8  # Increased from 4
    patience = 15  # Increased from 10
    warmup_epochs = 5

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    best_map_score = 0
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        # Warmup phase
        if epoch < warmup_epochs:
            lr_scale = min(1.0, float(epoch + 1) / warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * learning_rate

        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            losses = losses / accumulation_steps
            losses.backward()

            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            running_loss += losses.item()

        logger.log_info(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}"
        )

        # Evaluate after each epoch
        val_accuracy, map_score = evaluate_model(model, val_loader, device)
        logger.log_info(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Validation Accuracy: {val_accuracy:.4f}, Validation mAP: {map_score:.4f}"
        )

        # Adjust learning rate after warmup phase
        if epoch >= warmup_epochs:
            lr_scheduler.step(map_score)

        # Early stopping and best model saving
        if map_score > best_map_score:
            best_map_score = map_score
            early_stopping_counter = 0
            save_model(model, "best_Faster_R_CNN.pth")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                logger.log_info(f"Early stopping at epoch {epoch+1}")
                break

    # Save the final model
    save_model(model, "final_Faster_R_CNN.pth")

    # torch.cuda.empty_cache()


def compute_iou(box1, box2):
    """Compute the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    # Calculate the coordinates of the intersection rectangle
    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    # Calculate area of the intersection rectangle
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    # Calculate the area of both the prediction and ground truth rectangles
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)

    # Calculate the intersection over union
    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


def evaluate_model(model, data_loader, device, iou_threshold=0.5):
    model.eval()
    correct_detections = 0
    total_images = 0
    metric = MeanAveragePrecision(iou_type="bbox")

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get predictions
            predictions = model(images)

            # Update mAP metric
            metric.update(predictions, targets)

            # Simple accuracy calculation
            for pred, target in zip(predictions, targets):
                if len(pred["boxes"]) > 0 and len(target["boxes"]) > 0:
                    # Consider detection correct if any predicted box has IoU > threshold with any target box
                    ious = box_iou(pred["boxes"], target["boxes"])
                    if ious.max() > iou_threshold:
                        correct_detections += 1
                total_images += 1

    accuracy = correct_detections / total_images if total_images > 0 else 0
    map_score = metric.compute()["map"]

    return accuracy, map_score


def main():
    # Load configuration
    config = load_config()

    # Paths from config
    root = config["dataset"]["dataset_Faster_R_CNN"]["root"]
    annotation_file = config["dataset"]["dataset_Faster_R_CNN"]["annotation_file"]
    val_root = config["dataset"]["dataset_Faster_R_CNN"]["val_root"]
    val_annotation_file = config["dataset"]["dataset_Faster_R_CNN"][
        "val_annotation_file"
    ]
    val_root = config["dataset"]["dataset_Faster_R_CNN"]["val_root"]
    val_annotation_file = config["dataset"]["dataset_Faster_R_CNN"][
        "val_annotation_file"
    ]

    # Transformations
    transform = create_transforms(config)

    # Dataset and DataLoader for training
    train_dataset = BDDDataset(root, annotation_file, transforms=transform)
    train_dataset = Subset(
        train_dataset, range(500)
    )  # Limit training set to 500 images
    train_loader = create_data_loader(config, train_dataset)

    # Dataset and DataLoader for validation
    val_dataset = BDDDataset(
        val_root, val_annotation_file, transforms=transform, filter_annotations=False
    )
    val_valid_indices = range(
        min(1000, len(val_dataset))
    )  # Limit validation set to 500 images
    val_dataset = Subset(val_dataset, val_valid_indices)
    val_loader = create_data_loader(config, val_dataset)

    # Model
    num_classes = config["hyperparameters"]["num_classes"]
    model = create_model(num_classes)

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    # Train
    train_model(config, model, train_loader, val_loader, device)
    save_model(model, "Faster_R_CNN.pth")


if __name__ == "__main__":
    main()
