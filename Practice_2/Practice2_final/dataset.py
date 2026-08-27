"""Pipeline dữ liệu CIFAR-10 dùng chung cho tất cả các thí nghiệm."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import CIFAR10_CLASSES, DataConfig, IMAGENET_MEAN, IMAGENET_STD


def build_transforms(image_size: int = 224, augmentation: bool = True):
    # Thiết lập các phép biến đổi dữ liệu cho tập huấn luyện và kiểm tra.
    train_ops = [transforms.Resize((image_size, image_size))]
    if augmentation:
        train_ops.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ]
        )
    train_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    test_ops = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(train_ops), transforms.Compose(test_ops)


def _split_indices(n_items: int, val_ratio: float, seed: int):
    val_size = int(n_items * val_ratio)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_items, generator=generator).tolist()
    val_indices = permutation[:val_size]
    train_indices = permutation[val_size:]
    return train_indices, val_indices


def build_dataloaders(config: DataConfig) -> dict[str, Any]:
# build_dataloaders nhận một cấu hình DataConfig và trả về một từ điển chứa các tập dữ liệu và Dataloaders cho huấn luyện,
# xác thực và kiểm tra. Nó cũng trả về danh sách
    config.data_dir.mkdir(parents=True, exist_ok=True)
    train_transform, test_transform = build_transforms(
        image_size=config.image_size,
        augmentation=config.augmentation,
    )

    train_augmented = datasets.CIFAR10(
        root=str(config.data_dir),
        train=True,
        download=True,
        transform=train_transform,
    )
    train_eval_transform = transforms.Compose(
        [
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_base_for_val = datasets.CIFAR10(
        root=str(config.data_dir),
        train=True,
        download=False,
        transform=train_eval_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=str(config.data_dir),
        train=False,
        download=True,
        transform=test_transform,
    )

    train_indices, val_indices = _split_indices(
        len(train_augmented), config.val_ratio, config.seed
    )
    train_dataset = Subset(train_augmented, train_indices)
    val_dataset = Subset(train_base_for_val, val_indices)

    pin_memory = torch.cuda.is_available()
    common_loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=config.num_workers > 0,
    )
    generator = torch.Generator().manual_seed(config.seed)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_kwargs)

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


def get_dataloader_bundle(
    batch_size: int = 32,
    augmentation: bool = True,
    data_dir: Path | None = None,
    seed: int = 42,
    num_workers: int = 2,
) -> dict[str, Any]:
    config = DataConfig(
        data_dir=data_dir or Path("data"),
        batch_size=batch_size,
        augmentation=augmentation,
        seed=seed,
        num_workers=num_workers,
    )
    return build_dataloaders(config)


def check_train_val_overlap(bundle: dict[str, Any]) -> int:
    """Return the number of shared indices between train and validation."""
    train_indices = set(bundle["train_indices"])
    val_indices = set(bundle["val_indices"])
    return len(train_indices.intersection(val_indices))
