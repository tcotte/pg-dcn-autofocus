import os

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import sys


class MobileNetV3_Regressor(nn.Module):
    """
    Read paper Real-Time Facial Affective Computing on Mobile Devices to understand why choose full connected layers
    compared to average pooling.

    """

    def __init__(self, pretrained=True, dropout: float = 0.2):
        super().__init__()
        if pretrained:
            base = torchvision.models.mobilenet_v3_small(weights='DEFAULT', dropout=dropout)
        else:
            base = torchvision.models.mobilenet_v3_small(dropout=dropout)

        self.features = base.features  # everything up to the last feature map
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc = nn.Linear(576 * 7 * 7, 1)  # single scalar output

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten all dims except batch
        x = self.fc(x)
        return x
