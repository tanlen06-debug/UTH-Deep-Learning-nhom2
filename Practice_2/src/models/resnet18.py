import torch.nn as nn
from torchvision import models


def build_resnet18_fc_only(num_classes=10):
    """
    ResNet-18 A:
    Freeze toàn bộ backbone và chỉ train lớp fc.
    """
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Freeze toàn bộ pretrained backbone
    for param in model.parameters():
        param.requires_grad = False

    # Thay classifier cuối cho CIFAR-10
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def build_resnet18_layer4_fc(num_classes=10):
    """
    ResNet-18 B:
    Freeze toàn bộ model, sau đó mở layer4 và fc để fine-tuning.
    """
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT
    )

    # Freeze toàn bộ model
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Thay classifier cuối
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def count_trainable_parameters(model):
    """
    Đếm số tham số được cập nhật trong quá trình training.
    """
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )