"""DenseNet121 model for chest X-ray classification."""

import torch
import torch.nn as nn
import torchvision.models as models


class DenseNet121(nn.Module):
    """DenseNet121 with ImageNet pretrained weights and custom classifier."""

    def __init__(self, class_count: int = 14, pretrained: bool = True):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = models.densenet121(weights=weights)
        in_features = self.net.classifier.in_features
        self.net.classifier = nn.Linear(in_features, class_count)

    def forward(self, x):
        return self.net(x)