"""Evaluate the best Practice 2 checkpoints on the common CIFAR-10 test set."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

from config import CHECKPOINT_DIR, FIGURES_DIR, TABLES_DIR, CIFAR10_CLASSES, DataConfig, get_device
from dataset import build_dataloaders
from model import build_densenet121, build_resnet18, build_vgg16
from train import load_checkpoint, set_seed


def _build_model(name: str):
    if name == "ResNet-18":
        return build_resnet18(10, pretrained=False, strategy="layer4_fc")
    if name == "VGG-16":
        return build_vgg16(10, pretrained=False, freeze_features=True)
    if name == "DenseNet-121":
        return build_densenet121(10, pretrained=False, freeze_backbone=False)
    raise ValueError(name)


def _checkpoint_specs():
    return [
        {
            "Model": "ResNet-18",
            "Strategy": "Layer4 + FC",
            "checkpoint": CHECKPOINT_DIR / "resnet18_layer4_fc_best.pth",
        },
        {
            "Model": "VGG-16",
            "Strategy": "Learning Rate = 0.0001",
            "checkpoint": CHECKPOINT_DIR / "vgg16_lr_0.0001_best.pth",
        },
        {
            "Model": "DenseNet-121",
            "Strategy": "Batch Size 64 + Fine-tuning",
            "checkpoint": CHECKPOINT_DIR / "densenet121_bs64_finetune.pth",
        },
    ]


def _denormalize(images):
    mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def evaluate_model(model, loader, device, class_names, model_name):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    correct_examples = []
    wrong_examples = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)
            bs = labels.size(0)
            loss_sum += loss.item() * bs
            correct += (predictions == labels).sum().item()
            total += bs
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

            if len(correct_examples) < 8 or len(wrong_examples) < 8:
                for image, label, pred in zip(images.cpu(), labels.cpu(), predictions.cpu()):
                    if label.item() == pred.item() and len(correct_examples) < 8:
                        correct_examples.append((image, label.item(), pred.item()))
                    elif label.item() != pred.item() and len(wrong_examples) < 8:
                        wrong_examples.append((image, label.item(), pred.item()))

    avg_loss = loss_sum / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)
    precision = 100.0 * precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = 100.0 * recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = 100.0 * f1_score(y_true, y_pred, average="macro", zero_division=0)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print(f"\n===== {model_name} TEST REPORT =====")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} - Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    cm_path = FIGURES_DIR / f"{model_name.lower().replace('-', '').replace(' ', '_')}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    for kind, examples in (("correct", correct_examples), ("wrong", wrong_examples)):
        if not examples:
            continue
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = np.asarray(axes).reshape(-1)
        for ax in axes:
            ax.axis("off")
        for ax, (image, label, pred) in zip(axes, examples):
            image = _denormalize(image.unsqueeze(0)).squeeze(0).permute(1, 2, 0).numpy()
            ax.imshow(image)
            ax.set_title(f"true: {class_names[label]}\npred: {class_names[pred]}")
        fig.suptitle(f"{model_name} - {kind.title()} Predictions")
        fig.tight_layout()
        out = FIGURES_DIR / f"{model_name.lower().replace('-', '').replace(' ', '_')}_{kind}_predictions.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return {
        "Model": model_name,
        "Test loss": avg_loss,
        "Test accuracy (%)": accuracy,
        "Precision macro (%)": precision,
        "Recall macro (%)": recall,
        "F1-score macro (%)": f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["all", "resnet18", "vgg16", "densenet121"], default="all")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    data = build_dataloaders(
        DataConfig(
            batch_size=args.batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
            augmentation=False,
        )
    )
    test_loader = data["test_loader"]

    name_map = {
        "resnet18": "ResNet-18",
        "vgg16": "VGG-16",
        "densenet121": "DenseNet-121",
    }
    specs = _checkpoint_specs()
    if args.model != "all":
        wanted = name_map[args.model]
        specs = [spec for spec in specs if spec["Model"] == wanted]

    rows = []
    for spec in specs:
        checkpoint_path = spec["checkpoint"]
        if not checkpoint_path.exists():
            print(f"WARNING: missing checkpoint, skipped: {checkpoint_path}")
            continue

        model = _build_model(spec["Model"]).to(device)
        load_checkpoint(checkpoint_path, model, map_location=device)
        result = evaluate_model(
            model,
            test_loader,
            device,
            data["class_names"],
            spec["Model"],
        )
        result["Strategy"] = spec["Strategy"]
        result["Checkpoint"] = checkpoint_path.name
        try:
            checkpoint_meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint_meta = torch.load(checkpoint_path, map_location="cpu")
        result["Training time (s)"] = checkpoint_meta.get("training_time_sec", np.nan)
        rows.append(result)

    comparison = pd.DataFrame(rows)
    comparison_path = TABLES_DIR / "metrics.csv"
    if not comparison.empty:
        comparison.to_csv(comparison_path, index=False)
        print("\n===== MODEL COMPARISON =====")
        print(comparison.to_string(index=False))
        print(f"Saved metrics: {comparison_path}")

        for column, filename, title in [
            ("Test accuracy (%)", "model_accuracy_comparison.png", "Model Accuracy Comparison"),
            ("Test loss", "model_loss_comparison.png", "Model Loss Comparison"),
            ("Training time (s)", "model_training_time_comparison.png", "Training Time Comparison"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(comparison["Model"], comparison[column])
            ax.set_title(title)
            ax.set_ylabel(column)
            ax.tick_params(axis="x", rotation=20)
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
            plt.close(fig)


if __name__ == "__main__":
    main()
