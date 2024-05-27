import torch
import torch.nn as nn
import yaml

# Load hyperparameters from config file
with open("configs/config.yaml", "r") as file:
    config = yaml.safe_load(file)


class CustomBackbone(nn.Module):
    def __init__(self):
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


class LaneVehicleDetectionNetYOLO(nn.Module):
    def __init__(self):
        super(LaneVehicleDetectionNetYOLO, self).__init__()

        self.num_anchors = config["model"]["num_anchors"]
        self.num_classes = config["model"]["num_classes"]

        self.backbone = CustomBackbone()

        self.fpn = nn.ModuleList(
            [
                nn.Conv2d(
                    config["backbone"]["layer4"]["out_channels"],
                    config["fpn"]["channels"],
                    1,
                ),
                nn.Conv2d(
                    config["backbone"]["layer3"]["out_channels"],
                    config["fpn"]["channels"],
                    1,
                ),
                nn.Conv2d(
                    config["backbone"]["layer2"]["out_channels"],
                    config["fpn"]["channels"],
                    1,
                ),
                nn.Conv2d(
                    config["backbone"]["layer1"]["out_channels"],
                    config["fpn"]["channels"],
                    1,
                ),
            ]
        )

        self.detection_head = nn.Sequential(
            nn.Conv2d(
                config["fpn"]["channels"],
                config["detection_head"]["intermediate_channels"],
                config["detection_head"]["kernel_size"],
                padding=config["detection_head"]["padding"],
            ),
            nn.ReLU(),
            nn.Conv2d(
                config["detection_head"]["intermediate_channels"],
                self.num_anchors * (5 + self.num_classes),
                1,
            ),
        )

        self.lane_head = nn.Sequential(
            nn.Conv2d(
                config["fpn"]["channels"],
                config["lane_head"]["intermediate_channels1"],
                config["lane_head"]["kernel_size"],
                padding=config["lane_head"]["padding"],
            ),
            nn.ReLU(),
            nn.Conv2d(
                config["lane_head"]["intermediate_channels1"],
                config["lane_head"]["intermediate_channels2"],
                config["lane_head"]["kernel_size"],
                padding=config["lane_head"]["padding"],
            ),
            nn.ReLU(),
            nn.Conv2d(
                config["lane_head"]["intermediate_channels2"],
                config["lane_head"]["output_channels"],
                1,
            ),
        )

    def forward(self, images, targets=None):
        features = self.backbone(images)

        fpn_features = [fpn_layer(features) for fpn_layer in self.fpn]

        concatenated_features = torch.cat(fpn_features, dim=1)

        detection_output = self.detection_head(concatenated_features)

        N, _, H, W = detection_output.shape
        detection_output = detection_output.view(
            N, self.num_anchors, 5 + self.num_classes, H, W
        ).permute(0, 1, 3, 4, 2)

        lane_output = self.lane_head(concatenated_features)

        return detection_output, lane_output
