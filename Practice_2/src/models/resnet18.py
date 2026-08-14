from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


# ============================================================
# GHI CHU: File nay ban dau trong (chua duoc push len repo).
# Duoc viet lai theo dung cach notebooks/02_resnet18.ipynb dang
# su dung (build_resnet18_fc_only / build_resnet18_layer4_fc),
# de evaluate.py co the dung lai kien truc va load checkpoint.
# Neu ban phu trach ResNet-18 dung cau truc khac, can doi chieu lai.
# ============================================================


def _build_base_resnet18(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """Tai ResNet-18 pretrained tren ImageNet, thay lop fc cho CIFAR-10."""
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def build_resnet18_fc_only(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    Model A - FC only.
    Dong bang toan bo backbone (conv1, layer1-4), chi huan luyen lop fc cuoi.
    """
    model = _build_base_resnet18(num_classes=num_classes, pretrained=pretrained)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def build_resnet18_layer4_fc(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    Model B - Layer4 + FC.
    Dong bang conv1, layer1, layer2, layer3. Mo layer4 va fc de fine-tune.
    """
    model = _build_base_resnet18(num_classes=num_classes, pretrained=pretrained)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def build_resnet18_full_finetune(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    Model demo (notebooks/05_training_tensorboard.ipynb) - Full fine-tune.
    KHONG dong bang lop nao, toan bo backbone + fc deu duoc train.

    LUU Y: day la kien truc dung trong thi nghiem so sanh Adam vs SGD
    (chi 5 epoch, muc dich demo), KHONG PHAI checkpoint tot nhat chinh
    thuc cua ResNet-18 duoc chon cho ket luan Practice 2
    (checkpoint chinh thuc la build_resnet18_layer4_fc, xem notebook 02).
    """
    return _build_base_resnet18(num_classes=num_classes, pretrained=pretrained)


def count_total_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_frozen_parameters(model: nn.Module) -> int:
    return count_total_parameters(model) - count_trainable_parameters(model)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Giu API thong nhat voi src/models/vgg16.py va densenet121.py."""
    return {
        "total_params": count_total_parameters(model),
        "trainable_params": count_trainable_parameters(model),
        "frozen_params": count_frozen_parameters(model),
    }


def save_checkpoint(model: nn.Module, path: Path) -> None:
    """Luu trong so model (state_dict thuan)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(
    path: Path,
    strategy: str = "layer4_fc",
    num_classes: int = 10,
) -> nn.Module:
    """
    Dung lai dung kien truc theo `strategy` roi nap trong so tu checkpoint.

    strategy:
        "fc_only"    -> build_resnet18_fc_only
        "layer4_fc"  -> build_resnet18_layer4_fc

    Ho tro ca 2 dinh dang checkpoint co trong project:
    - dict co key "model_state_dict" (do src/trainer.py luu ra)
    - state_dict thuan
    """
    if strategy == "fc_only":
        model = build_resnet18_fc_only(num_classes=num_classes, pretrained=False)
    elif strategy == "layer4_fc":
        model = build_resnet18_layer4_fc(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"strategy khong hop le: {strategy!r}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )

    model.load_state_dict(state_dict)
    return model


def build_summary(model: nn.Module, strategy: str, num_classes: int = 10) -> dict[str, Any]:
    """Tong hop thong tin model de in ra notebook / ban giao cho buoc danh gia."""
    params = count_parameters(model)
    return {
        "strategy": strategy,
        "num_classes": num_classes,
        **params,
    }
