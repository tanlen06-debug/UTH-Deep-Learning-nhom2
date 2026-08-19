from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_densenet121(
    num_classes: int = 10,
    pretrained: bool = True,
    freeze_backbone: bool = True,
):
    """
    Tạo DenseNet-121 cho bài toán CIFAR-10.

    Parameters
    ----------
    num_classes : int
        Số lớp đầu ra. CIFAR-10 có 10 lớp.

    pretrained : bool
        Nếu True, sử dụng weights pretrained.

    freeze_backbone : bool
        Nếu True, đóng băng backbone để thực hiện Transfer Learning.
    """

    # Tải weights pretrained
    if pretrained:
        weights = models.DenseNet121_Weights.DEFAULT
    else:
        weights = None

    model = models.densenet121(
        weights=weights
    )

    # Đóng băng backbone
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False

    # Lấy số feature đầu vào của classifier cũ
    in_features = model.classifier.in_features

    # Thay classifier cho CIFAR-10
    model.classifier = nn.Linear(
        in_features,
        num_classes
    )

    return model


def enable_finetuning_last_block(model):
    """
    Mở DenseBlock cuối và classifier
    để thực hiện Fine-tuning.
    """

    # Đóng băng toàn bộ trước
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Mở DenseBlock cuối
    for parameter in model.features.denseblock4.parameters():
        parameter.requires_grad = True

    # Classifier vẫn train
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def count_parameters(model):
    """
    Đếm tổng số parameters và số trainable parameters.
    """

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total_params, trainable_params