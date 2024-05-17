
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
        # 2048, 1024, 512, 256 channels in FPN: These correspond to feature maps from different layers of the ResNet50 backbone.
        # 1x1 convolutions in FPN: These reduce the number of channels to a uniform size (256) for consistency and easier processing.

        # Define the Region Proposal Network (RPN) to generate proposals for object detection
        self.rpn = nn.ModuleList([
            nn.Conv2d(256, 256, 3, padding=1), # 3x3 convolution for feature extraction
            nn.Conv2d(256, self.num_anchors * 2, 1),  # 1x1 convolution for anchor classification. 
            # self.num_anchors * 2:
            # Number of output channels, where self.num_anchors is the number of anchor boxes, and 2 corresponds to the two possible classes (object or background).
            nn.Conv2d(256, self.num_anchors * 4, 1), # 1x1 convolution for bounding box regression.
            #self.num_anchors * 4:
            #Number of output channels, where self.num_anchors is the number of anchor boxes, and 4 corresponds to the four coordinates (x, y, width, height) required to regress each bounding box.

        ])
        # Explanation: The RPN proposes regions where objects might be.
        # The 3x3 convolution extracts features, while the 1x1 convolutions output classification scores and bounding box deltas for each anchor.
        # 3x3 and 1x1 convolutions in RPN: The 3x3 convolutions are for feature extraction, while the 1x1 convolutions output classification scores and bounding box predictions for each anchor.

        # Define RoI Align for pooling features from the proposed regions 
        # MultiScaleRoIAlign is a PyTorch operation used to extract fixed-size feature maps from different levels of a feature pyramid network (FPN) for each region of interest (RoI).  
        self.roi_align = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], # Feature maps from the FPN layers
                                            output_size=7, # Output size for RoI align
                                            sampling_ratio=2) # Sampling ratio for RoI align.
        # A sampling_ratio of 2 means that for each output pixel, 2x2 sampling points are used to compute the value, enhancing the precision of the RoI alignment.
        # Explanation: RoI Align pools features from proposed regions into a fixed size (7x7), enabling subsequent classification and regression.
        # The sampling ratio helps determine the precision of pooling.
        # output_size=7 in RoI Align: This standard size allows the network to process RoIs consistently, regardless of their original dimensions.

        # Define the detection head for classifying objects and regressing bounding boxes
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024), # Fully connected layer for classification
            nn.ReLU(),                    # Activation function
            nn.Linear(1024, num_classes)  # Output layer for class logits
        )
        # Explanation: The detection head classifies objects within proposed regions.
        # The 7x7 feature map is flattened and processed through fully connected layers to produce class scores.
        # 256 * 7 * 7 in detection head:
        # The RoI Align output size (7x7) and the number of channels (256) determine the input size for the fully connected layers in the detection head.

        self.bbox_regressor = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024), # Fully connected layer for bounding box regression
            nn.ReLU(),                    # Activation function
            nn.Linear(1024, num_classes * 4) # Output layer for bounding box coordinates
        )
        # Explanation: The bounding box regressor predicts the coordinates of bounding boxes for each class.
        # The output size is num_classes * 4 because each box has 4 coordinates (x, y, width, height).
        # num_classes * 4 in bbox_regressor: Each bounding box is represented by four coordinates (x, y, width, height), so the output size is proportional to the number of classes.

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
        fpn_features = [fpn_layer(features[i]) for i, fpn_layer in enumerate(self.fpn)]
        # Explanation: The backbone features are passed through FPN layers to get multi-scale feature maps, which help in detecting objects of various sizes.
        
        # Initialize lists to store RPN classification logits and bounding box predictions
        # Loop through RPN layers and apply them to the feature maps from the FPN
        rpn_logits, rpn_bbox = [], []
        for rpn_layer in self.rpn:
            # Apply the current RPN layer to the feature maps from the FPN
            rpn_feat = rpn_layer(fpn_features)
                
            # Extract and append the classification scores (logits) for the anchors
            # rpn_feat[:, :self.num_anchors * 2, :, :] slices the output feature map to get the first self.num_anchors * 2 channels
            # These channels represent the classification scores for each anchor (object vs. background)
            rpn_logits.append(rpn_feat[:, :self.num_anchors * 2, :, :])
                
            # Extract and append the bounding box regression predictions for the anchors
            # rpn_feat[:, self.num_anchors * 2:self.num_anchors * 6, :, :] slices the output feature map to get the next self.num_anchors * 4 channels
            # These channels represent the bounding box regression predictions for each anchor (4 coordinates: x, y, width, height)
            rpn_bbox.append(rpn_feat[:, self.num_anchors * 2:self.num_anchors * 6, :, :])
            
        # Explanation: For each FPN feature map, apply RPN layers to get classification scores and bounding box predictions. These are collected in lists.
        
        # Use RoI Align to pool features from the proposed regions (proposals assumed to be generated)
        rois = self.roi_align(fpn_features, targets)
        # Explanation: RoI Align pools features from the regions proposed by the RPN, transforming them into fixed-size feature maps for further processing.
        
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