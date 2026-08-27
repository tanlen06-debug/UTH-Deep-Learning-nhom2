"""Command-line training engine for Practice 2.

Examples
--------
python train.py --model resnet18 --strategy fc_only
python train.py --model resnet18 --strategy layer4_fc
python train.py --model vgg16 --learning-rate 0.0001
python train.py --model densenet121 --batch-size 64
python train.py --model resnet18 --optimizer sgd
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    CHECKPOINT_DIR,
    FIGURES_DIR,
    PROJECT_ROOT,
    RUNS_DIR,
    TABLES_DIR,
    CIFAR10_CLASSES,
    DataConfig,
    get_device,
)
from dataset import build_dataloaders, check_train_val_overlap
from model import (
    build_densenet121,
    build_resnet18,
    build_vgg16,
    count_frozen_parameters,
    count_total_parameters,
    count_trainable_parameters,
    enable_densenet_last_block,
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_optimizer(name: str, model, learning_rate: float, momentum: float = 0.9):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    name = name.lower()
    if name == "adam":
        return optim.Adam(parameters, lr=learning_rate)
    if name == "sgd":
        return optim.SGD(parameters, lr=learning_rate, momentum=momentum)
    raise ValueError("optimizer must be 'adam' or 'sgd'")


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    val_loss: float,
    val_acc: float,
    history: dict[str, list[float]],
    metadata: dict[str, Any] | None = None,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "history": history,
        "metadata": metadata or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, model, optimizer=None, map_location=None):
    kwargs = {"map_location": map_location}
    try:
        checkpoint = torch.load(path, weights_only=False, **kwargs)
    except TypeError:
        checkpoint = torch.load(path, **kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def _accuracy_from_logits(outputs, labels) -> int:
    return (outputs.argmax(dim=1) == labels).sum().item()


def train_one_epoch(model, loader, criterion, optimizer, device, writer=None, epoch=0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += _accuracy_from_logits(outputs, labels)
        total += batch_size

    loss_value = running_loss / max(total, 1)
    acc_value = 100.0 * correct / max(total, 1)
    if writer:
        writer.add_scalar("Loss/Train", loss_value, epoch)
        writer.add_scalar("Accuracy/Train", acc_value, epoch)
    return loss_value, acc_value


def validate_one_epoch(model, loader, criterion, device, writer=None, epoch=0):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct += _accuracy_from_logits(outputs, labels)
            total += batch_size

    loss_value = running_loss / max(total, 1)
    acc_value = 100.0 * correct / max(total, 1)
    if writer:
        writer.add_scalar("Loss/Validation", loss_value, epoch)
        writer.add_scalar("Accuracy/Validation", acc_value, epoch)
    return loss_value, acc_value


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs: int,
    device,
    log_dir: str | Path,
    checkpoint_path: str | Path,
):
    """Shared training/validation/checkpoint/TensorBoard engine."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorBoard is required for training. Run `pip install -r requirements.txt`."
        ) from exc
    writer = SummaryWriter(log_dir=str(log_dir))
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }
    best_val_acc = float("-inf")
    best_epoch = 0
    start_time = time.time()

    model.to(device)
    sample_batch = next(iter(train_loader))
    sample_images = sample_batch[0].to(device)
    try:
        writer.add_graph(model, sample_images[:1])
    except Exception as exc:  # graph is optional in the source requirement
        print(f"TensorBoard model graph skipped: {exc}")
    writer.add_images("Images/Train", sample_batch[0][:8], 0, dataformats="NCHW")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, writer, epoch
        )
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device, writer, epoch
        )
        current_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("LearningRate", current_lr, epoch)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_checkpoint(
                Path(checkpoint_path),
                model,
                optimizer,
                epoch,
                val_loss,
                val_acc,
                history,
            )

        print(
            f"Epoch {epoch:>2}/{num_epochs} | "
            f"train loss {train_loss:.4f} | train acc {train_acc:.2f}% | "
            f"val loss {val_loss:.4f} | val acc {val_acc:.2f}% | "
            f"{time.time() - epoch_start:.1f}s"
        )

    total_time = time.time() - start_time
    writer.close()

    # Keep the timing in the best checkpoint so predict.py can build the
    # same final comparison table without duplicating handwritten values.
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.exists():
        try:
            final_checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        except TypeError:
            final_checkpoint = torch.load(checkpoint_file, map_location="cpu")
        final_checkpoint["training_time_sec"] = total_time
        torch.save(final_checkpoint, checkpoint_file)

    return history, best_val_acc, best_epoch, total_time


def _plot_history(history, title: str, filename: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
    axes[0].set_title(f"{title} - Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], marker="o", label="Train Accuracy")
    axes[1].plot(epochs, history["val_acc"], marker="o", label="Validation Accuracy")
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True)
    axes[1].legend()
    fig.tight_layout()
    output = FIGURES_DIR / filename
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output}")


def run_single_experiment(args, data_bundle, device):
    num_classes = len(CIFAR10_CLASSES)
    criterion = nn.CrossEntropyLoss()

    if args.model == "resnet18":
        strategy = args.strategy
        model = build_resnet18(
            num_classes=num_classes,
            pretrained=not args.no_pretrained,
            strategy=strategy,
        )
        checkpoint_name = (
            "resnet18_fc_only_best.pth"
            if strategy == "fc_only"
            else "resnet18_layer4_fc_best.pth"
        )
        run_name = f"resnet18_{strategy}"
        epochs = 5 if args.epochs is None else args.epochs
        learning_rate = args.learning_rate
        optimizer_name = args.optimizer
    elif args.model == "vgg16":
        model = build_vgg16(
            num_classes=num_classes,
            pretrained=not args.no_pretrained,
            freeze_features=True,
        )
        checkpoint_name = f"vgg16_lr_{args.learning_rate:g}_best.pth"
        run_name = f"vgg16_lr_{args.learning_rate:g}"
        epochs = 10 if args.epochs is None else args.epochs
        learning_rate = args.learning_rate
        optimizer_name = "adam"
    elif args.model == "densenet121":
        return run_densenet_experiment(args, data_bundle, device)
    else:
        raise ValueError(args.model)

    optimizer = get_optimizer(optimizer_name, model, learning_rate, args.momentum)
    checkpoint_path = CHECKPOINT_DIR / checkpoint_name
    log_dir = RUNS_DIR / run_name

    print(f"\n===== {args.model.upper()} | {run_name} =====")
    print(f"Device: {device}")
    print(f"Total parameters: {count_total_parameters(model):,}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    print(f"Frozen parameters: {count_frozen_parameters(model):,}")
    print(f"Learning rate: {learning_rate}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Checkpoint: {checkpoint_path}")

    history, best_acc, best_epoch, training_time = train_model(
        model=model,
        train_loader=data_bundle["train_loader"],
        val_loader=data_bundle["val_loader"],
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=epochs,
        device=device,
        log_dir=log_dir,
        checkpoint_path=checkpoint_path,
    )
    _plot_history(history, run_name, f"{run_name}_curves.png")

    summary = pd.DataFrame(
        [
            {
                "Model": args.model,
                "Strategy": getattr(args, "strategy", "transfer learning"),
                "Learning Rate": learning_rate,
                "Batch Size": args.batch_size,
                "Optimizer": optimizer_name,
                "Best Val Accuracy (%)": best_acc,
                "Best Epoch": best_epoch,
                "Training Time (s)": training_time,
                "Checkpoint": checkpoint_path.name,
            }
        ]
    )
    summary_path = TABLES_DIR / f"{run_name}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Saved summary: {summary_path}")
    return summary


def run_densenet_experiment(args, data_bundle, device):
    num_classes = len(CIFAR10_CLASSES)
    criterion = nn.CrossEntropyLoss()
    batch_size = args.batch_size
    name = f"densenet121_bs{batch_size}"
    transfer_ckpt = CHECKPOINT_DIR / f"{name}_transfer.pth"
    finetune_ckpt = CHECKPOINT_DIR / f"{name}_finetune.pth"

    model = build_densenet121(
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        freeze_backbone=True,
    )
    optimizer = get_optimizer(
        "adam", model, learning_rate=0.001, momentum=args.momentum
    )

    print(f"\n===== {name} | TRANSFER LEARNING =====")
    transfer_history, transfer_best_acc, transfer_best_epoch, transfer_time = train_model(
        model=model,
        train_loader=data_bundle["train_loader"],
        val_loader=data_bundle["val_loader"],
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=2,
        device=device,
        log_dir=RUNS_DIR / name / "transfer",
        checkpoint_path=transfer_ckpt,
    )

    load_checkpoint(transfer_ckpt, model, map_location=device)
    enable_densenet_last_block(model)
    optimizer = get_optimizer("adam", model, learning_rate=0.001, momentum=args.momentum)

    print(f"\n===== {name} | FINE-TUNING =====")
    finetune_history, finetune_best_acc, finetune_best_epoch, finetune_time = train_model(
        model=model,
        train_loader=data_bundle["train_loader"],
        val_loader=data_bundle["val_loader"],
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=3,
        device=device,
        log_dir=RUNS_DIR / name / "finetune",
        checkpoint_path=finetune_ckpt,
    )

    history = {
        key: transfer_history[key] + finetune_history[key]
        for key in transfer_history
    }
    if finetune_best_acc >= transfer_best_acc:
        best_acc = finetune_best_acc
        best_stage = "Fine-tuning"
        best_checkpoint = finetune_ckpt
        best_epoch = 2 + finetune_best_epoch
    else:
        best_acc = transfer_best_acc
        best_stage = "Transfer Learning"
        best_checkpoint = transfer_ckpt
        best_epoch = transfer_best_epoch

    _plot_history(history, name, f"{name}_curves.png")
    summary = pd.DataFrame(
        [
            {
                "Model": "DenseNet-121",
                "Strategy": "Transfer Learning + Fine-tuning denseblock4",
                "Learning Rate": 0.001,
                "Batch Size": batch_size,
                "Optimizer": "Adam",
                "Transfer Best Val Accuracy (%)": transfer_best_acc,
                "Fine-tune Best Val Accuracy (%)": finetune_best_acc,
                "Best Val Accuracy (%)": best_acc,
                "Best Stage": best_stage,
                "Best Epoch": best_epoch,
                "Training Time (s)": transfer_time + finetune_time,
                "Checkpoint": best_checkpoint.name,
            }
        ]
    )
    summary_path = TABLES_DIR / f"{name}_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Saved summary: {summary_path}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Practice 2 script runner for pretrained CNNs")
    parser.add_argument("--model", choices=["resnet18", "vgg16", "densenet121"], required=True)
    parser.add_argument("--strategy", choices=["fc_only", "layer4_fc"], default="fc_only")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    data_config = DataConfig(
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        augmentation=not args.no_augmentation,
    )
    data_bundle = build_dataloaders(data_config)

    print("===== PRACTICE 2 SETUP =====")
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Train samples:", len(data_bundle["train_dataset"]))
    print("Validation samples:", len(data_bundle["val_dataset"]))
    print("Test samples:", len(data_bundle["test_dataset"]))
    print("Classes:", data_bundle["class_names"])
    print("Train/validation index overlap:", check_train_val_overlap(data_bundle))
    images, labels = next(iter(data_bundle["train_loader"]))
    print("Images shape:", tuple(images.shape))
    print("Labels shape:", tuple(labels.shape))

    run_single_experiment(args, data_bundle, device)


if __name__ == "__main__":
    main()
