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
with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class CustomBackbone(nn.Module):
    def __init__(self):
        super(CustomBackbone, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(
            config['model']['input_channels'], 
            config['backbone']['conv1']['out_channels'], 
            kernel_size=config['backbone']['conv1']['kernel_size'], 
            stride=config['backbone']['conv1']['stride'], 
            padding=config['backbone']['conv1']['padding']
        )
        self.bn1 = nn.BatchNorm2d(config['backbone']['conv1']['out_channels'])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Define blocks of layers
        self.layer1 = self._make_layer(
            config['backbone']['layer1']['in_channels'], 
            config['backbone']['layer1']['out_channels'], 
            config['backbone']['layer1']['blocks']
        )
        self.layer2 = self._make_layer(
            config['backbone']['layer2']['in_channels'], 
            config['backbone']['layer2']['out_channels'], 
            config['backbone']['layer2']['blocks'], 
            stride=config['backbone']['layer2']['stride']
        )
        self.layer3 = self._make_layer(
            config['backbone']['layer3']['in_channels'], 
            config['backbone']['layer3']['out_channels'], 
            config['backbone']['layer3']['blocks'], 
            stride=config['backbone']['layer3']['stride']
        )
        self.layer4 = self._make_layer(
            config['backbone']['layer4']['in_channels'], 
            config['backbone']['layer4']['out_channels'], 
            config['backbone']['layer4']['blocks'], 
            stride=config['backbone']['layer4']['stride']
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
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
    def __init__(self, backbone_out_channels):
        super(FeaturePyramidNetwork, self).__init__()
        self.lateral4 = nn.Conv2d(backbone_out_channels[3], 256, kernel_size=1)
        self.lateral3 = nn.Conv2d(backbone_out_channels[2], 256, kernel_size=1)
        self.lateral2 = nn.Conv2d(backbone_out_channels[1], 256, kernel_size=1)
        self.lateral1 = nn.Conv2d(backbone_out_channels[0], 256, kernel_size=1)
        
        self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

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
    def __init__(self, output_size):
        super(RoIAlignLayer, self).__init__()
        self.roi_align = RoIAlign(output_size, spatial_scale=1.0, sampling_ratio=2)

    def forward(self, features, rois):
        return self.roi_align(features, rois)

# Semantic Lane Head
class SemanticLaneHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SemanticLaneHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.deconv = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.mask_fcn_logits = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.deconv(x))
        x = self.mask_fcn_logits(x)
        return x