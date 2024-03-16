import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision import models


class ResNet18Model(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(ResNet18Model, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
