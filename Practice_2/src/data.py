from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# Tên 10 lớp của CIFAR-10
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# Giá trị normalization của ImageNet
# Phù hợp với các mô hình pretrained trên ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path
    image_size: int = 224
    batch_size: int = 32
    val_ratio: float = 0.10
    num_workers: int = 2
    seed: int = 42


def build_transforms(
    image_size: int = 224,
) -> tuple[transforms.Compose, transforms.Compose]:
    """
    Tạo hai pipeline biến đổi ảnh:

    - train_transform:
      Có data augmentation để làm đa dạng dữ liệu huấn luyện.

    - eval_transform:
      Không có augmentation, sử dụng cho validation và test.
    """

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),

            # Lật ngang ảnh ngẫu nhiên
            transforms.RandomHorizontalFlip(p=0.5),

            # Xoay ảnh trong khoảng -10 đến +10 độ
            transforms.RandomRotation(degrees=10),

            # Thay đổi nhẹ độ sáng và độ tương phản
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
            ),

            transforms.ToTensor(),

            # Chuẩn hóa theo ImageNet
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )

    return train_transform, eval_transform


def build_dataloaders(
    config: DataConfig,
) -> dict[str, Any]:
    """
    Tải CIFAR-10, chia train/validation và tạo các DataLoader.

    Returns
    -------
    dict
        Chứa train_loader, val_loader, test_loader,
        class_names và các dataset.
    """

    if not 0 < config.val_ratio < 1:
        raise ValueError("val_ratio phải nằm trong khoảng từ 0 đến 1.")

    config.data_dir.mkdir(parents=True, exist_ok=True)

    train_transform, eval_transform = build_transforms(
        image_size=config.image_size
    )

    # Tạo hai đối tượng train khác nhau:
    # - Một đối tượng dùng augmentation cho train
    # - Một đối tượng không augmentation cho validation
    train_full_augmented = datasets.CIFAR10(
        root=str(config.data_dir),
        train=True,
        download=True,
        transform=train_transform,
    )

    train_full_evaluation = datasets.CIFAR10(
        root=str(config.data_dir),
        train=True,
        download=True,
        transform=eval_transform,
    )

    test_dataset = datasets.CIFAR10(
        root=str(config.data_dir),
        train=False,
        download=True,
        transform=eval_transform,
    )

    total_train_samples = len(train_full_augmented)
    val_size = int(total_train_samples * config.val_ratio)

    # Tạo thứ tự ngẫu nhiên nhưng có seed cố định
    generator = torch.Generator().manual_seed(config.seed)

    indices = torch.randperm(
        total_train_samples,
        generator=generator,
    ).tolist()

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Subset(
        train_full_augmented,
        train_indices,
    )

    val_dataset = Subset(
        train_full_evaluation,
        val_indices,
    )

    # Kiểm tra train và validation không trùng nhau
    overlap = set(train_indices).intersection(val_indices)

    if overlap:
        raise RuntimeError(
            "Phát hiện dữ liệu bị trùng giữa train và validation."
        )

    pin_memory = torch.cuda.is_available()

    common_loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": pin_memory,
    }

    # persistent_workers chỉ dùng khi num_workers > 0
    if config.num_workers > 0:
        common_loader_options["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        **common_loader_options,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_loader_options,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_loader_options,
    )

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "class_names": CIFAR10_CLASSES,
        "train_indices": train_indices,
        "val_indices": val_indices,
    }