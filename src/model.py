# model.py

import torch.nn as nn
from torchvision import models


class SoilClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super(SoilClassifier, self).__init__()

        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)

        # Replace final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def get_model(num_classes=5, pretrained=True):
    """
    Returns an instance of the SoilClassifier model.
    
    Args:
        num_classes (int): Number of soil classes.
        pretrained (bool): Whether to use ImageNet pretrained weights.
        
    Returns:
        model (nn.Module): Initialized model.
    """
    return SoilClassifier(num_classes=num_classes, pretrained=pretrained)
