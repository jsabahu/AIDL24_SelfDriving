import os
import torch
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from utils import save_model
from logger import Logger

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
            transforms.Resize(
                (resize_height, resize_width), antialias=antialias
            ),  # Explicitly set antialias from config
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4
            ),
            transforms.RandomRotation(30),  # Increased rotation for augmentation
            transforms.RandomResizedCrop(
                (resize_height, resize_width), scale=(0.4, 1.0), antialias=antialias
            ),  # More aggressive random crop
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform


class BDDDataset(Dataset):
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.transforms = transforms
        with open(annotation_file) as f:
            self.annotations = json.load(f)
        self.annotations = self._filter_annotations(self.annotations)
        logger.log_info(
            f"Filtered dataset to {len(self.annotations)} valid annotations."
        )

    def _filter_annotations(self, annotations):
        valid_annotations = []
        image_files = sorted(
            [f for f in os.listdir(self.root) if f.lower().endswith(".jpg")]
        )
        for ann in annotations:
            if "labels" in ann and any(
                obj["category"] == "car" for obj in ann["labels"]
            ):
                valid_annotations.append(ann)
            else:
                logger.log_debug(
                    f"Skipping invalid annotation: {ann['name'] if 'name' in ann else 'unknown'}"
                )
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
        config["hyperparameters"]["learning_rate"] * 0.01
    )  # Lower initial learning rate
    weight_decay = config["hyperparameters"]["weight_decay"]
    num_epochs = config["hyperparameters"]["num_epoch"]
    accumulation_steps = config["hyperparameters"]["accumulation_steps"]
    patience = config["hyperparameters"]["patience"] + 5  # Increased patience

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

    # Learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(
        optimizer, "min", patience=3, factor=0.1, verbose=True
    )

    best_val_loss = float("inf")
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for i, (images, targets) in enumerate(train_loader):
            if len(images) == 0:  # Check if the list of images is empty
                logger.log_debug(
                    f"No valid images in batch at iteration {i}. Skipping..."
                )
                continue
            else:
                logger.log_debug(
                    f"{len(images)} valid images in batch at iteration {i}."
                )
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            if any(len(t["boxes"]) == 0 for t in targets):
                continue

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            if torch.isnan(losses):
                logger.log_debug(
                    f"NaN detected in losses at iteration {i} of epoch {epoch}"
                )
                for k, v in loss_dict.items():
                    logger.log_debug(f"{k} loss: {v}")
                logger.log_debug(f"Images: {images}")
                logger.log_debug(f"Targets: {targets}")
                continue

            losses = losses / accumulation_steps
            losses.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            running_loss += losses.item()
            
        logger.log_info(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}"
        )

        writer.add_scalar(f"Loss train", running_loss / len(train_loader), epoch)

        # Evaluate after each epoch
        accuracy, val_loss = evaluate_model(
            model, val_loader, device, iou_threshold=0.5
        )  # Adjusted IoU threshold for evaluation
        logger.log_info(
            f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}"
        )

        writer.add_scalar(f"Loss eval", val_loss)


        # Adjust learning rate
        lr_scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                logger.log_info(f"Early stopping at epoch {epoch+1}")
                break
    
    writer.flush()
    torch.cuda.empty_cache()


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
    correct = 0
    total = 0
    running_val_loss = 0.0
    with torch.no_grad():
        for images, targets in data_loader:
            if len(images) == 0:  # Check if the list of images is empty
                logger.log_debug("No valid images in batch. Skipping...")
                continue

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Perform the prediction
            predictions = model(images)

            for target, prediction in zip(targets, predictions):
                gt_boxes = target["boxes"].cpu()
                pred_boxes = prediction["boxes"].cpu()
                if len(pred_boxes) == 0:
                    continue

                val_loss = sum(
                    [
                        compute_iou(pred_box, gt_box)
                        for pred_box in pred_boxes
                        for gt_box in gt_boxes
                    ]
                ) / len(pred_boxes)
                running_val_loss += val_loss

                for pred_box in pred_boxes:
                    iou_scores = [compute_iou(pred_box, gt_box) for gt_box in gt_boxes]
                    if max(iou_scores) > iou_threshold:
                        correct += 1
                    total += 1

                # Log details for debugging
                # logger.log_debug(f"Ground Truth Boxes: {gt_boxes}")
                # logger.log_debug(f"Predicted Boxes: {pred_boxes}")
                # logger.log_debug(f"IOU Scores: {iou_scores}")

    accuracy = correct / total if total > 0 else 0
    val_loss = running_val_loss / len(data_loader) if len(data_loader) > 0 else 0
    return accuracy, val_loss


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

    # Transformations
    transform = create_transforms(config)

    # Dataset and DataLoader for training
    train_dataset = BDDDataset(root, annotation_file, transforms=transform)
    train_dataset = Subset(
        train_dataset, range(500)
    )  # Limit training set to 500 images
    train_loader = create_data_loader(config, train_dataset)

    # Dataset and DataLoader for validation
    val_dataset = BDDDataset(val_root, val_annotation_file, transforms=transform)
    val_valid_indices = range(
        min(500, len(val_dataset))
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