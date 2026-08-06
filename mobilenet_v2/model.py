"""Torchvision official MobileNet-v2 model."""

from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def create_model(pretrained: bool = False, num_classes: int = 1000) -> nn.Module:
    if pretrained:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        if num_classes != 1000:
            model.classifier[-1] = nn.Linear(model.last_channel, num_classes)
        return model
    return mobilenet_v2(weights=None, num_classes=num_classes)
