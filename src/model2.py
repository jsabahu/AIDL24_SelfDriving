
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import MultiScaleRoIAlign, RoIPool

class CustomBackbone(nn.Module):
    def __init__(self):
        super(CustomBackbone, self).__init__()
        # ---------------------------------------------------------------------------------------------------------
        # Purpose: Initial processing of the input image to extract basic features and reduce spatial dimensions.

        # First convolutional layer: 3 input channels (RGB), 64 output channels, 7x7 kernel, stride 2, padding 3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        # self.conv1: A convolutional layer that processes the input image (with 3 channels for RGB) to produce 64 feature maps.
        # The 7x7 kernel size is relatively large, which helps in capturing more context from the input image.
        # The stride of 2 reduces the spatial dimensions by half.

        # Batch normalization for the first convolutional layer
        self.bn1 = nn.BatchNorm2d(64)
        # self.bn1: Batch normalization is applied to the output of the convolutional layer to normalize the activations.
        # It helps in stabilizing and speeding up the training process.
        
        # ReLU activation function applied in-place
        self.relu = nn.ReLU(inplace=True)
        # self.relu: The ReLU activation function introduces non-linearity, which is essential for the network to learn complex patterns.

        # Max pooling layer: 3x3 kernel, stride 2, padding 1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.maxpool: A max pooling layer that further reduces the spatial dimensions by half, with a 3x3 kernel and a stride of 2.
        # This helps in reducing the computational load and in capturing translation-invariant features.
        # -------------------------------------------------------------------------------------------------------
        # Purpose: Create a series of convolutional layers to extract more complex features at multiple scales.

        # Define the first block of layers with 64 input channels and 64 output channels, consisting of 2 layers
        self.layer1 = self._make_layer(64, 64, 2)
        # self.layer1: A block of layers with 64 input and 64 output channels, consisting of 2 layers.
        # This block maintains the spatial dimensions

        # Define the second block of layers with 64 input channels and 128 output channels, 
        # a stride of 2 (halving the spatial dimensions), and consisting of 2 layers
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        # self.layer2: A block with 64 input channels and 128 output channels, consisting of 2 layers.
        # The first layer in this block uses a stride of 2 to halve the spatial dimensions, increasing the depth of the features.

        # Define the third block of layers with 128 input channels and 256 output channels, 
        # a stride of 2 (halving the spatial dimensions), and consisting of 2 layers
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        # self.layer3: A block with 128 input channels and 256 output channels, consisting of 2 layers.
        # The first layer uses a stride of 2 to halve the spatial dimensions, further increasing the depth of the features.

        # Define the fourth block of layers with 256 input channels and 512 output channels, 
        # a stride of 2 (halving the spatial dimensions), and consisting of 2 layers
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        # self.layer4: A block with 256 input channels and 512 output channels, consisting of 2 layers.
        # The first layer uses a stride of 2 to halve the spatial dimensions, further increasing the depth of the features.

    # Purpose: Dynamically create a block of convolutional layers with batch normalization and ReLU activation.
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
    
        # First layer in the block: convolutional layer with in_channels to out_channels,
        # 3x3 kernel, specified stride, and padding of 1
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))

        # Batch normalization for the first layer in the block
        layers.append(nn.BatchNorm2d(out_channels))

        # ReLU activation function applied in-place
        layers.append(nn.ReLU(inplace=True))

        # Loop to add additional layers if blocks > 1
        for _ in range(1, blocks):
            # Convolutional layer with out_channels for both input and output,
            # 3x3 kernel, and padding of 1
            layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

            # Batch normalization for additional layers
            layers.append(nn.BatchNorm2d(out_channels))

            # ReLU activation function applied in-place for additional layers
            layers.append(nn.ReLU(inplace=True))

        # Combine all layers into a sequential module
        return nn.Sequential(*layers)
    

    
    def forward(self, x):
        # Apply the first convolutional layer
        x = self.conv1(x)

        # Apply batch normalization to the output of the first convolutional layer
        x = self.bn1(x)

        # Apply ReLU activation function to introduce non-linearity
        x = self.relu(x)

        # Apply max pooling to reduce the spatial dimensions
        x = self.maxpool(x)

        # Pass the output through the first block of layers
        x = self.layer1(x)

        # Pass the output through the second block of layers
        x = self.layer2(x)

        # Pass the output through the third block of layers
        x = self.layer3(x)

        # Pass the output through the fourth block of layers
        x = self.layer4(x)

        # Return the final output after all layers
        return x



class LaneVehicleDetectionNetYOLO(nn.Module):
    def __init__(self, num_classes, num_anchors=9):
        super(LaneVehicleDetectionNetYOLO, self).__init__()

        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        # Custom backbone network
        self.backbone = CustomBackbone()
        
        # Define the Feature Pyramid Network (FPN) to handle multi-scale feature extraction
        self.fpn = nn.ModuleList([
            nn.Conv2d(512, 256, 1), # 1x1 convolution to reduce channel dimensions for the topmost feature map
            nn.Conv2d(256, 256, 1), # 1x1 convolution for the second feature map
            nn.Conv2d(128, 256, 1), # 1x1 convolution for the third feature map
            nn.Conv2d(64, 256, 1), # 1x1 convolution for the fourth feature map
        ])
        # Explanation: The FPN helps in extracting features from different layers of the backbone.
        # The 1x1 convolutions reduce the number of channels to a uniform size (256) for easier processing.

        # Detection head for predicting class scores, bounding boxes, and lane segmentation
        self.detection_head = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, self.num_anchors * (5 + self.num_classes), 1)  # 5: (4 bounding box coordinates + 1 objectness score)
        )

        # Define the lane detection head for semantic segmentation of lanes
        self.lane_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), # 3x3 convolution for feature extraction
            nn.ReLU(),                         # Activation function
            nn.Conv2d(128, 64, 3, padding=1),  # 3x3 convolution for further feature extraction
            nn.ReLU(),                         # Activation function
            nn.Conv2d(64, 1, 1)                # 1x1 convolution for binary segmentation output
        )
        # Explanation: The lane detection head segments lanes in the image.
        # The convolutions extract and refine features, and the final 1x1 convolution outputs a binary mask indicating lane locations.

    def forward(self, images, targets=None):
        # Forward pass through the backbone to extract features
        features = self.backbone(images)
        
        # Pass features through the FPN layers to get multi-scale features
        fpn_features = [fpn_layer(features) for fpn_layer in self.fpn]
        # Explanation: The backbone features are passed through FPN layers to get multi-scale feature maps, which help in detecting objects of various sizes.
        
        # Flatten the pooled features and pass through the detection head
        detection_features = rois.view(rois.size(0), -1)
        class_logits = self.classifier(detection_features)
        bbox_preds = self.bbox_regressor(detection_features)
        # Explanation: The pooled features are flattened and passed through the detection head to produce class logits and bounding box predictions.
        
        # Forward pass through the lane detection head for semantic segmentation
        lane_preds = self.lane_head(fpn_features[0])
        # Explanation: The topmost FPN feature map is used for lane segmentation. The lane detection head processes it to produce a binary mask indicating lanes.

        
        # Return the outputs: class logits, bounding box predictions, and lane predictions
        return class_logits, bbox_preds, lane_preds