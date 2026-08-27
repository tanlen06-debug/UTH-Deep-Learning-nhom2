# Các hàm xây dựng mô hình cho ResNet-18, VGG-16 và DenseNet-121 với tùy chọn tiền huấn luyện  và đóng băng các tầng .
from __future__ import annotations

from typing import Literal

import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights, ResNet18_Weights, VGG16_Weights


ResNetStrategy = Literal["fc_only", "layer4_fc"]


def build_resnet18(
    num_classes: int = 10,
    pretrained: bool = True,
    strategy: ResNetStrategy = "fc_only",
):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    if strategy not in {"fc_only", "layer4_fc"}:
        raise ValueError("strategy must be 'fc_only' or 'layer4_fc'")

    for parameter in model.parameters():
        parameter.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    if strategy == "layer4_fc":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True

    return model


def build_vgg16(
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_features: bool = True,
):
    weights = VGG16_Weights.DEFAULT if pretrained else None
    model = models.vgg16(weights=weights)

    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    if freeze_features:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    return model


def build_densenet121(
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = True,
):
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)

    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    return model


def enable_densenet_last_block(model) -> None:
    # bật các tham số của dếnblock4 và bộ phân loại để huấn luyện
    for parameter in model.features.denseblock4.parameters():
        parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def count_total_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(model) -> int:
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def count_frozen_parameters(model) -> int:
    return count_total_parameters(model) - count_trainable_parameters(model)
