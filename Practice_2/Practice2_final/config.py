"""Central configuration for Practice 2.

The notebook had the same project/data/device/seed setup repeated in several
places. This module keeps those values in one place while preserving the
experiment settings used in the source material.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

for _path in (CHECKPOINT_DIR, RUNS_DIR, FIGURES_DIR, TABLES_DIR):
    _path.mkdir(parents=True, exist_ok=True)

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

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DataConfig:
    data_dir: Path = DATA_DIR
    image_size: int = 224
    batch_size: int = 32
    val_ratio: float = 0.10
    num_workers: int = 2
    seed: int = 42
    augmentation: bool = True


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 5
    learning_rate: float = 0.001
    optimizer: str = "adam"
    momentum: float = 0.9
    seed: int = 42
    use_pretrained: bool = True


@dataclass(frozen=True)
class DenseNetConfig:
    transfer_epochs: int = 2
    finetune_epochs: int = 3
    learning_rate: float = 0.001
    optimizer: str = "adam"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
