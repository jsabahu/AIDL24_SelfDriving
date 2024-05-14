
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import MultiScaleRoIAlign, RoIPool

class LaneVehicleDetectionNet(nn.Module):
    def __init__(self, num_classes, num_anchors=9):
        super(LaneVehicleDetectionNet, self).__init__()
        
        self.num_anchors = num_anchors
        
        # Backbone network
        backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Feature Pyramid Network
        self.fpn = nn.ModuleList([
            nn.Conv2d(2048, 256, 1),
            nn.Conv2d(1024, 256, 1),
            nn.Conv2d(512, 256, 1),
            nn.Conv2d(256, 256, 1),
        ])
        
        # Region Proposal Network (RPN)
        self.rpn = nn.ModuleList([
            nn.Conv2d(256, 256, 3, padding=1),
            nn.Conv2d(256, self.num_anchors * 2, 1),  # classification
            nn.Conv2d(256, self.num_anchors * 4, 1),  # bbox regression
        ])
        
        # RoI Align
        self.roi_align = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'],
                                            output_size=7,
                                            sampling_ratio=2)
        
        # Detection Head
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )
        
        self.bbox_regressor = nn.Sequential(
            nn.Linear(256 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes * 4)
        )
        
        # Lane Detection Head (segmentation)
        self.lane_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1)  # single channel output for binary segmentation
        )

    def forward(self, images, targets=None):
        # Backbone
        features = self.backbone(images)
        
        # FPN
        fpn_features = [fpn_layer(features[i]) for i, fpn_layer in enumerate(self.fpn)]
        
        # RPN
        rpn_logits, rpn_bbox = [], []
        for rpn_layer in self.rpn:
            rpn_feat = rpn_layer(fpn_features)
            rpn_logits.append(rpn_feat[:, :self.num_anchors * 2, :, :])
            rpn_bbox.append(rpn_feat[:, self.num_anchors * 4:, :, :])
        
        # RoI Align
        rois = self.roi_align(fpn_features, targets)
        
        # Detection Head
        detection_features = rois.view(rois.size(0), -1)
        class_logits = self.classifier(detection_features)
        bbox_preds = self.bbox_regressor(detection_features)
        
        # Lane Detection Head
        lane_preds = self.lane_head(fpn_features[0])
        
        return class_logits, bbox_preds, lane_preds