from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG16_Weights


@dataclass(frozen=True)
class VGG16Config:
    num_classes: int = 10
    freeze_features: bool = True
    learning_rate: float = 0.001


def build_vgg16(config: VGG16Config) -> nn.Module:
    """
    Tải VGG-16 pre-trained (ImageNet), thay classifier cuối để phù hợp
    với bài toán mới (mặc định 10 lớp, dùng cho CIFAR-10).

    Đóng băng phần features (trích xuất đặc trưng) nếu
    config.freeze_features = True, chỉ huấn luyện phần classifier.

    Returns
    -------
    nn.Module
        VGG-16 đã chỉnh sửa, sẵn sàng để train.
    """
    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

    if config.freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, config.num_classes)

    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Đếm tổng số tham số và số tham số trainable của model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


def model_size_mb(model: nn.Module) -> float:
    """Ước lượng dung lượng model (MB), dựa trên số tham số float32 (4 byte)."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params * 4 / (1024 ** 2)


def save_checkpoint(model: nn.Module, path: Path) -> None:
    """Lưu trọng số model tốt nhất."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_checkpoint(path: Path, config: VGG16Config) -> nn.Module:
    """Tải lại model đã lưu trọng số từ checkpoint."""
    model = build_vgg16(config)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model


def build_summary(model: nn.Module, config: VGG16Config) -> dict[str, Any]:
    """
    Tổng hợp thông tin model: số tham số, dung lượng, cấu hình —
    dùng để in ra notebook và bàn giao cho thành viên tổng hợp (Văn Sơn).
    """
    params = count_parameters(model)
    return {
        "num_classes": config.num_classes,
        "freeze_features": config.freeze_features,
        "learning_rate": config.learning_rate,
        "total_params": params["total_params"],
        "trainable_params": params["trainable_params"],
        "model_size_mb": round(model_size_mb(model), 2),
    }