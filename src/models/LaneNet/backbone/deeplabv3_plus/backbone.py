# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
from .resnet_atrous import *
from .xception import *


def build_backbone(backbone_name, pretrained=True, os=16):
    if backbone_name == "res50_atrous":
        net = resnet50_atrous(pretrained=pretrained, os=os)
        return net
    elif backbone_name == "res101_atrous":
        net = resnet101_atrous(pretrained=pretrained, os=os)
        return net
    elif backbone_name == "res152_atrous":
        net = resnet152_atrous(pretrained=pretrained, os=os)
        return net
    elif backbone_name == "xception" or backbone_name == "Xception":
        net = xception(pretrained=pretrained, os=os)
        return net
    else:
        raise ValueError(
            "backbone.py: The backbone named %s is not supported yet." % backbone_name
        )
