"""Torchvision official Inception-v3 model."""

from torch import nn
from torchvision.models import Inception_V3_Weights, inception_v3


def create_model(pretrained: bool = False, num_classes: int = 1000) -> nn.Module:
    if pretrained:
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        model.aux_logits = False
        model.AuxLogits = None
        if num_classes != 1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    return inception_v3(
        weights=None,
        aux_logits=False,
        init_weights=False,
        num_classes=num_classes,
    )
