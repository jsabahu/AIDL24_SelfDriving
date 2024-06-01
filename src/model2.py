import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import RoIAlign
import os
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor, Compose, RandomHorizontalFlip

# Load hyperparameters from config file
with open("configs/config.yaml", "r") as file:
    CONFIG = yaml.safe_load(file)


class CustomBackbone(nn.Module):
    def __init__(self, config=CONFIG):
        super(CustomBackbone, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(
            config["model"]["input_channels"],
            config["backbone"]["conv1"]["out_channels"],
            kernel_size=config["backbone"]["conv1"]["kernel_size"],
            stride=config["backbone"]["conv1"]["stride"],
            padding=config["backbone"]["conv1"]["padding"],
        )
        self.bn1 = nn.BatchNorm2d(config["backbone"]["conv1"]["out_channels"])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Define blocks of layers
        self.layer1 = self._make_layer(
            config["backbone"]["layer1"]["in_channels"],
            config["backbone"]["layer1"]["out_channels"],
            config["backbone"]["layer1"]["blocks"],
        )
        self.layer2 = self._make_layer(
            config["backbone"]["layer2"]["in_channels"],
            config["backbone"]["layer2"]["out_channels"],
            config["backbone"]["layer2"]["blocks"],
            stride=config["backbone"]["layer2"]["stride"],
        )
        self.layer3 = self._make_layer(
            config["backbone"]["layer3"]["in_channels"],
            config["backbone"]["layer3"]["out_channels"],
            config["backbone"]["layer3"]["blocks"],
            stride=config["backbone"]["layer3"]["stride"],
        )
        self.layer4 = self._make_layer(
            config["backbone"]["layer4"]["in_channels"],
            config["backbone"]["layer4"]["out_channels"],
            config["backbone"]["layer4"]["blocks"],
            stride=config["backbone"]["layer4"]["stride"],
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            )
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(1, blocks):
            layers.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


# Feature Pyramid Network Definition
class FeaturePyramidNetwork(nn.Module):
    def __init__(self, config=CONFIG, backbone_out_channels=[]):
        super(FeaturePyramidNetwork, self).__init__()
        self.lateral4 = nn.Conv2d(
            backbone_out_channels[3],
            config["fpn"]["lateral"]["out_channels"],
            kernel_size=1,
        )
        self.lateral3 = nn.Conv2d(
            backbone_out_channels[2],
            config["fpn"]["lateral"]["out_channels"],
            kernel_size=1,
        )
        self.lateral2 = nn.Conv2d(
            backbone_out_channels[1],
            config["fpn"]["lateral"]["out_channels"],
            kernel_size=1,
        )
        self.lateral1 = nn.Conv2d(
            backbone_out_channels[0],
            config["fpn"]["lateral"]["out_channels"],
            kernel_size=1,
        )

        self.smooth4 = nn.Conv2d(
            config["fpn"]["lateral"]["out_channels"],
            config["fpn"]["smooth"]["out_channels"],
            kernel_size=config["fpn"]["smooth"]["kernel_size"],
            padding=config["fpn"]["smooth"]["padding"],
        )
        self.smooth3 = nn.Conv2d(
            config["fpn"]["lateral"]["out_channels"],
            config["fpn"]["smooth"]["out_channels"],
            kernel_size=config["fpn"]["smooth"]["kernel_size"],
            padding=config["fpn"]["smooth"]["padding"],
        )
        self.smooth2 = nn.Conv2d(
            config["fpn"]["lateral"]["out_channels"],
            config["fpn"]["smooth"]["out_channels"],
            kernel_size=config["fpn"]["smooth"]["kernel_size"],
            padding=config["fpn"]["smooth"]["padding"],
        )
        self.smooth1 = nn.Conv2d(
            config["fpn"]["lateral"]["out_channels"],
            config["fpn"]["smooth"]["out_channels"],
            kernel_size=config["fpn"]["smooth"]["kernel_size"],
            padding=config["fpn"]["smooth"]["padding"],
        )

    def forward(self, x):
        c1, c2, c3, c4 = x

        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, scale_factor=2, mode="nearest")
        p2 = self.lateral2(c2) + F.interpolate(p3, scale_factor=2, mode="nearest")
        p1 = self.lateral1(c1) + F.interpolate(p2, scale_factor=2, mode="nearest")

        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)

        return p1, p2, p3, p4


# RoI Align Layer
class RoIAlignLayer(nn.Module):
    def __init__(self, config):
        super(RoIAlignLayer, self).__init__()
        self.roi_align = RoIAlign(
            config["roi_align"]["output_size"],
            spatial_scale=config["roi_align"]["spatial_scale"],
            sampling_ratio=config["roi_align"]["sampling_ratio"],
        )

    def forward(self, features, rois):
        return self.roi_align(features, rois)


# Semantic Lane Head
class SemanticLaneHead(nn.Module):
    def __init__(self, config):
        super(SemanticLaneHead, self).__init__()
        in_channels = config["semantic_lane_head"]["in_channels"]
        num_classes = config["model"]["num_classes"]
        self.conv1 = nn.Conv2d(
            in_channels,
            config["semantic_lane_head"]["conv"]["out_channels"],
            kernel_size=config["semantic_lane_head"]["conv"]["kernel_size"],
            padding=config["semantic_lane_head"]["conv"]["padding"],
        )
        self.conv2 = nn.Conv2d(
            config["semantic_lane_head"]["conv"]["out_channels"],
            config["semantic_lane_head"]["conv"]["out_channels"],
            kernel_size=config["semantic_lane_head"]["conv"]["kernel_size"],
            padding=config["semantic_lane_head"]["conv"]["padding"],
        )
        self.conv3 = nn.Conv2d(
            config["semantic_lane_head"]["conv"]["out_channels"],
            config["semantic_lane_head"]["conv"]["out_channels"],
            kernel_size=config["semantic_lane_head"]["conv"]["kernel_size"],
            padding=config["semantic_lane_head"]["conv"]["padding"],
        )
        self.conv4 = nn.Conv2d(
            config["semantic_lane_head"]["conv"]["out_channels"],
            config["semantic_lane_head"]["conv"]["out_channels"],
            kernel_size=config["semantic_lane_head"]["conv"]["kernel_size"],
            padding=config["semantic_lane_head"]["conv"]["padding"],
        )
        self.deconv = nn.ConvTranspose2d(
            config["semantic_lane_head"]["deconv"]["out_channels"],
            config["semantic_lane_head"]["deconv"]["out_channels"],
            kernel_size=config["semantic_lane_head"]["deconv"]["kernel_size"],
            stride=config["semantic_lane_head"]["deconv"]["stride"],
        )
        self.mask_fcn_logits = nn.Conv2d(
            config["semantic_lane_head"]["deconv"]["out_channels"],
            num_classes,
            kernel_size=config["semantic_lane_head"]["mask_fcn_logits"]["kernel_size"],
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.deconv(x))
        x = self.mask_fcn_logits(x)
        return x


# Model Integration
class LaneDetectionModel(nn.Module):
    def __init__(self, config=CONFIG):
        super(LaneDetectionModel, self).__init__()
        self.backbone = CustomBackbone(config)
        backbone_out_channels = [
            config["backbone"]["layer1"]["out_channels"],
            config["backbone"]["layer2"]["out_channels"],
            config["backbone"]["layer3"]["out_channels"],
            config["backbone"]["layer4"]["out_channels"],
        ]
        self.fpn = FeaturePyramidNetwork(backbone_out_channels, config)
        self.roi_align = RoIAlignLayer(config)
        self.mask_head = SemanticLaneHead(config)

    def forward(self, images, rois):
        features = self.backbone(images)
        pyramid_features = self.fpn(features)
        aligned_features = self.roi_align(pyramid_features, rois)
        mask_logits = self.mask_head(aligned_features)
        return mask_logits


""" # Dataset and DataLoader Definitions
class BDD100KDataset(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        mask = np.array(mask)
        mask = mask.astype(np.uint8)
        masks = torch.as_tensor(mask, dtype=torch.uint8)

        target = {}
        target["masks"] = masks.unsqueeze(0)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)

# Load Configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Example of Usage
def main():
    config_path = 'config.yaml'
    config = load_config(config_path)

    dataset = BDD100KDataset(root='data/bdd100k', transforms=get_transform(train=True))
    dataset_test = BDD100KDataset(root='data/bdd100k', transforms=get_transform(train=False))

    data_loader = DataLoader(dataset, batch_size=config['dataloader']['batch_size'], shuffle=True, num_workers=config['dataloader']['num_workers'])
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=config['dataloader']['num_workers'])

    model = LaneDetectionModel(config)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=config['model']['learning_rate'], momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        lr_scheduler.step()

        # Add evaluation logic here

    torch.save(model.state_dict(), 'lane_detection_model.pth')
"""

if __name__ == "__main__":
    # Example usage
    # from torchviz import make_dot
    model = FeaturePyramidNetwork(backbone_out_channels=[1, 512, 8, 8])
    input_image = [1, 512, 8, 8]
    # input_image = torch.randn(1, 3, 255, 255)
    output = model(input_image)

    # make_dot(output, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
    print("Output Shape:", output.shape)
