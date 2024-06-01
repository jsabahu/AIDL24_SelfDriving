import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIAlign
from logger import Logger

# Load hyperparameters from config file
with open("configs/config.yaml", "r") as file:
    CONFIG = yaml.safe_load(file)


class CustomBackbone(nn.Module):
    def __init__(self, config=CONFIG):
        super(CustomBackbone, self).__init__()

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
        c1 = self.maxpool(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


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


class PyramidRoIAlign(nn.Module):
    def __init__(self, pool_size, image_shape):
        super(PyramidRoIAlign, self).__init__()
        self.pool_size = pool_size
        self.image_shape = image_shape

    def forward(self, boxes, feature_maps):
        if boxes.shape[1] != 5:
            raise ValueError(
                "RoIs should be in the format [batch_index, x1, y1, x2, y2]"
            )

        y1, x1, y2, x2 = boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4]
        h = y2 - y1
        w = x2 - x1

        image_area = self.image_shape[0] * self.image_shape[1]
        roi_level = 4 + torch.log2(
            torch.sqrt(h * w)
            / (224.0 / torch.sqrt(torch.tensor(image_area, dtype=torch.float32)))
        )
        roi_level = roi_level.round().int().clamp(2, 5)

        pooled = []
        box_to_level = []

        for i, level in enumerate(range(2, 6)):
            ix = roi_level == level
            if not ix.any():
                continue
            ix = torch.nonzero(ix)[:, 0]
            level_boxes = boxes[ix, :]

            box_to_level.append(ix)

            level_boxes = level_boxes.detach()
            indices = level_boxes[:, 0].int()
            level_boxes = level_boxes[:, 1:]
            pooled_features = RoIAlign(
                self.pool_size, spatial_scale=1 / (2**level), sampling_ratio=0
            )(
                feature_maps[i],
                torch.cat([indices[:, None].float(), level_boxes], dim=1),
            )
            pooled.append(pooled_features)

        pooled = torch.cat(pooled, dim=0)
        box_to_level = torch.cat(box_to_level, dim=0)
        _, box_to_level = torch.sort(box_to_level)
        pooled = pooled[box_to_level, :, :]

        return pooled


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
        image_shape = config.get("model", {}).get("image_shape", [256, 256, 3])
        self.fpn = FeaturePyramidNetwork(
            config=config, backbone_out_channels=backbone_out_channels
        )
        self.pyramid_roi_align = PyramidRoIAlign(
            pool_size=config["roi_align"]["output_size"], image_shape=image_shape
        )
        self.mask_head = SemanticLaneHead(config)

    def forward(self, images, rois):
        features = self.backbone(images)
        pyramid_features = self.fpn(features)
        aligned_features = self.pyramid_roi_align(rois, pyramid_features)
        mask_logits = self.mask_head(aligned_features)
        return mask_logits


if __name__ == "__main__":
    logger = Logger(log_file="logfile.log", level="debug")
    # Example usage
    config = CONFIG

    # Initialize the complete model
    model = LaneDetectionModel(config=config)

    # Generate an example input image and ROIs
    input_images = torch.randn(1, 3, 256, 256)
    rois = torch.tensor(
        [[0, 50, 50, 150, 150], [0, 30, 30, 100, 100]], dtype=torch.float32
    )

    # Pass the input image and ROIs through the model
    output = model(input_images, rois)

    # Print the output shape of the model
    logger.log_info(f"Lane Detection Model Output Shape: {output.shape}")
