"""ResNet18 model for chest X-ray classification."""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    """ResNet18 with ImageNet pretrained weights and custom classifier."""

    def __init__(self, class_count: int = 14, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = models.resnet18(weights=weights)
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, class_count)

    def forward(self, x):
        return self.net(x)