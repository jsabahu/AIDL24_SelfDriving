
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import MultiScaleRoIAlign, RoIPool

class LaneVehicleDetectionNet(nn.Module):
    def __init__(self, num_classes, num_anchors=9 ):
        # num_anchors=9: This is a common choice, providing a sufficient variety of anchor box shapes and sizes to cover different object dimensions.
        super(LaneVehicleDetectionNet, self).__init__()

        # Number of anchor boxes used in the RPN
        self.num_anchors = num_anchors
        
        # Initialize the backbone network using a pre-trained ResNet50 model, excluding the final classification layers
        backbone = models.resnet50(pretrained=True) # or False if our DataSet is big enough, which I think so.
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        # Explanation: ResNet50 is a powerful CNN model pre-trained on ImageNet.
        # The last two layers are excluded as they are specific to ImageNet classification.
        
        # Define the Feature Pyramid Network (FPN) to handle multi-scale feature extraction
        self.fpn = nn.ModuleList([
            nn.Conv2d(2048, 256, 1), # 1x1 convolution to reduce channel dimensions for the topmost feature map
            nn.Conv2d(1024, 256, 1), # 1x1 convolution for the second feature map
            nn.Conv2d(512, 256, 1), # 1x1 convolution for the third feature map
            nn.Conv2d(256, 256, 1), # 1x1 convolution for the fourth feature map
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
        rpn_logits, rpn_bbox = [], []
        for rpn_layer in self.rpn:
            rpn_feat = rpn_layer(fpn_features)
            rpn_logits.append(rpn_feat[:, :self.num_anchors * 2, :, :])
            rpn_bbox.append(rpn_feat[:, self.num_anchors * 4:, :, :])
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