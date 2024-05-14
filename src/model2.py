
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import MultiScaleRoIAlign, RoIPool

class LaneVehicleDetectionNet(nn.Module):
    def __init__(self, num_classes, num_anchors=9):
        super(LaneVehicleDetectionNet, self).__init__()

        # Number of anchor boxes used in the RPN
        self.num_anchors = num_anchors
        
        # Initialize the backbone network using a pre-trained ResNet50 model, excluding the final classification layers
        backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Define the Feature Pyramid Network (FPN) to handle multi-scale feature extraction
        self.fpn = nn.ModuleList([
            nn.Conv2d(2048, 256, 1), # 1x1 convolution to reduce channel dimensions for the topmost feature map
            nn.Conv2d(1024, 256, 1), # 1x1 convolution for the second feature map
            nn.Conv2d(512, 256, 1), # 1x1 convolution for the third feature map
            nn.Conv2d(256, 256, 1), # 1x1 convolution for the fourth feature map
        ])
        
        # Define the Region Proposal Network (RPN) to generate proposals for object detection
        self.rpn = nn.ModuleList([
            nn.Conv2d(256, 256, 3, padding=1), # 3x3 convolution for feature extraction
            nn.Conv2d(256, self.num_anchors * 2, 1),  # 1x1 convolution for anchor classification
            nn.Conv2d(256, self.num_anchors * 4, 1), # 1x1 convolution for bounding box regression
        ])
        
        # Define RoI Align for pooling features from the proposed regions   
        self.roi_align = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], # Feature maps from the FPN layers
                                            output_size=7, # Output size for RoI align
                                            sampling_ratio=2) # Sampling ratio for RoI align
        
        # Define the detection head for classifying objects and regressing bounding boxes
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024), # Fully connected layer for classification
            nn.ReLU(),                    # Activation function
            nn.Linear(1024, num_classes)  # Output layer for class logits
        )
        
        self.bbox_regressor = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024), # Fully connected layer for bounding box regression
            nn.ReLU(),                    # Activation function
            nn.Linear(1024, num_classes * 4) # Output layer for bounding box coordinates
        )
        
        # Define the lane detection head for semantic segmentation of lanes
        self.lane_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), # 3x3 convolution for feature extraction
            nn.ReLU(),                         # Activation function
            nn.Conv2d(128, 64, 3, padding=1),  # 3x3 convolution for further feature extraction
            nn.ReLU(),                         # Activation function
            nn.Conv2d(64, 1, 1)                # 1x1 convolution for binary segmentation output
        )

    def forward(self, images, targets=None):
        # Forward pass through the backbone to extract features
        features = self.backbone(images)
        
        # Pass features through the FPN layers to get multi-scale features
        fpn_features = [fpn_layer(features[i]) for i, fpn_layer in enumerate(self.fpn)]
        
        # Initialize lists to store RPN classification logits and bounding box predictions
        rpn_logits, rpn_bbox = [], []
        for rpn_layer in self.rpn:
            rpn_feat = rpn_layer(fpn_features)
            rpn_logits.append(rpn_feat[:, :self.num_anchors * 2, :, :])
            rpn_bbox.append(rpn_feat[:, self.num_anchors * 4:, :, :])
        
        # Use RoI Align to pool features from the proposed regions (proposals assumed to be generated)
        rois = self.roi_align(fpn_features, targets)
        
        # Flatten the pooled features and pass through the detection head
        detection_features = rois.view(rois.size(0), -1)
        class_logits = self.classifier(detection_features)
        bbox_preds = self.bbox_regressor(detection_features)
        
        # Forward pass through the lane detection head for semantic segmentation
        lane_preds = self.lane_head(fpn_features[0])
        
        # Return the outputs: class logits, bounding box predictions, and lane predictions
        return class_logits, bbox_preds, lane_preds