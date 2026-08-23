"""
Practice_2/src/evaluate.py

Danh gia mo hinh & tong hop ket qua - Van Son
(ResNet-18, VGG-16, DenseNet-121 tren CIFAR-10)

Module nay CHI chua cac ham dung chung, khong tu chay gi ca.
Orchestration (goi cac ham theo dung thu tu, luu file, in bao cao)
duoc thuc hien trong notebooks/06_evaluation.ipynb.

Cach dung nhanh:

    from src.evaluate import (
        ModelSpec, get_device, evaluate_model,
        plot_confusion_matrix, show_prediction_samples,
        build_comparison_table, plot_metric_comparison,
        best_model_summary,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .data import CIFAR10_CLASSES, IMAGENET_MEAN, IMAGENET_STD


# ============================================================
# 1. TIEN ICH CHUNG
# ============================================================

def get_device() -> torch.device:
    """Tu dong dung GPU neu co, khong thi dung CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ModelSpec:
    """
    Mo ta mot checkpoint can danh gia.

    display_name       : ten hien thi trong bang so sanh, vd "ResNet-18"
    build_fn            : ham dung kien truc model CHUA load weight,
                           vd lambda: build_resnet18_layer4_fc(num_classes=10, pretrained=False)
    checkpoint_path     : duong dan file .pth (thuong bi gitignore, phai xin tu ban train)
    training_time_sec   : thoi gian train (giay) - lay tu log/README cua nguoi train model
    """

    display_name: str
    build_fn: Callable[[], nn.Module]
    checkpoint_path: Path
    training_time_sec: Optional[float] = None


def load_model_weights(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    """
    Nap trong so vao model tu checkpoint.

    Ho tro ca 2 dinh dang checkpoint dang ton tai trong project:
    - dict co key 'model_state_dict' (do src/trainer.py luu ra, vd ResNet-18, DenseNet-121)
    - state_dict thuan (do vgg16.save_checkpoint luu ra)
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay checkpoint: {checkpoint_path}\n"
            "File .pth bi gitignore nen phai xin truc tiep tu ban da train model nay "
            "(copy vao thu muc Practice_2/checkpoints/ tren may cua ban)."
        )

    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
    else:
        state_dict = raw

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_and_load(spec: ModelSpec, device: torch.device) -> nn.Module:
    """Dung kien truc theo spec.build_fn roi nap trong so tu spec.checkpoint_path."""
    model = spec.build_fn()
    return load_model_weights(model, spec.checkpoint_path, device)


# ============================================================
# 2. DANH GIA TREN TEST SET
# ============================================================

@dataclass
class EvaluationResult:
    model_name: str
    test_loss: float
    test_accuracy: float           # %
    precision_macro: float         # %
    recall_macro: float            # %
    f1_macro: float                # %
    y_true: np.ndarray
    y_pred: np.ndarray
    training_time_sec: Optional[float]
    report_text: str


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader,
    device: torch.device,
    class_names: list[str] = CIFAR10_CLASSES,
    model_name: str = "model",
    training_time_sec: Optional[float] = None,
) -> EvaluationResult:
    """
    Chay model tren toan bo test set, tinh:
    - Test loss (CrossEntropy trung binh)
    - Test accuracy (%)
    - Precision / Recall / F1-score (macro, %)
    - Classification report day du (per-class)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    total = 0
    correct = 0

    y_true_batches: list[np.ndarray] = []
    y_pred_batches: list[np.ndarray] = []

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        y_true_batches.append(labels.cpu().numpy())
        y_pred_batches.append(predicted.cpu().numpy())

    y_true = np.concatenate(y_true_batches)
    y_pred = np.concatenate(y_pred_batches)

    test_loss = running_loss / total
    test_accuracy = 100.0 * correct / total

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    report_text = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, zero_division=0
    )

    return EvaluationResult(
        model_name=model_name,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        precision_macro=precision * 100,
        recall_macro=recall * 100,
        f1_macro=f1 * 100,
        y_true=y_true,
        y_pred=y_pred,
        training_time_sec=training_time_sec,
        report_text=report_text,
    )


# ============================================================
# 3. CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    result: EvaluationResult,
    class_names: list[str] = CIFAR10_CLASSES,
    save_path: Optional[Path] = None,
    normalize: bool = False,
):
    """Ve confusion matrix cho mot EvaluationResult, luu PNG neu co save_path."""
    cm = confusion_matrix(result.y_true, result.y_pred)

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix - {result.model_name}")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


# ============================================================
# 4. ANH DU DOAN DUNG / SAI
# ============================================================

def _unnormalize(img_tensor: torch.Tensor) -> np.ndarray:
    """Dao nguoc normalize ImageNet de hien thi anh dung mau."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


@torch.no_grad()
def show_prediction_samples(
    model: nn.Module,
    test_loader,
    device: torch.device,
    class_names: list[str] = CIFAR10_CLASSES,
    model_name: str = "model",
    n_correct: int = 8,
    n_wrong: int = 8,
    save_path_correct: Optional[Path] = None,
    save_path_wrong: Optional[Path] = None,
):
    """
    Duyet test set, thu thap toi da n_correct anh du doan DUNG
    va n_wrong anh du doan SAI, roi ve thanh luoi anh.
    """
    model.eval()

    correct_samples = []
    wrong_samples = []

    for images, labels in test_loader:
        images_dev = images.to(device)
        outputs = model(images_dev)
        _, preds = outputs.max(1)
        preds = preds.cpu()

        for img, true_label, pred_label in zip(images, labels, preds):
            true_label = int(true_label)
            pred_label = int(pred_label)

            if true_label == pred_label and len(correct_samples) < n_correct:
                correct_samples.append((img, true_label, pred_label))
            elif true_label != pred_label and len(wrong_samples) < n_wrong:
                wrong_samples.append((img, true_label, pred_label))

        if len(correct_samples) >= n_correct and len(wrong_samples) >= n_wrong:
            break

    def _plot_grid(samples, title, save_path):
        if not samples:
            print(f"[{model_name}] Khong tim thay mau cho: {title}")
            return None

        n = len(samples)
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = np.atleast_1d(axes).reshape(-1)

        for ax, (img, true_l, pred_l) in zip(axes, samples):
            ax.imshow(_unnormalize(img))
            ax.set_title(
                f"True: {class_names[true_l]}\nPred: {class_names[pred_l]}",
                fontsize=9,
            )
            ax.axis("off")

        for ax in axes[len(samples):]:
            ax.axis("off")

        fig.suptitle(f"{title} - {model_name}")
        fig.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")

        return fig

    fig_correct = _plot_grid(correct_samples, "Du doan DUNG", save_path_correct)
    fig_wrong = _plot_grid(wrong_samples, "Du doan SAI", save_path_wrong)
    return fig_correct, fig_wrong


# ============================================================
# 5. BANG SO SANH & BIEU DO SO SANH 3 MODEL
# ============================================================

def build_comparison_table(results: list[EvaluationResult]) -> pd.DataFrame:
    """
    Gop cac EvaluationResult thanh 1 DataFrame - dung de:
    - in ra notebook
    - luu results/tables/metrics.csv
    """
    rows = []
    for r in results:
        rows.append({
            "Model": r.model_name,
            "Test loss": round(r.test_loss, 4),
            "Test accuracy (%)": round(r.test_accuracy, 2),
            "Precision macro (%)": round(r.precision_macro, 2),
            "Recall macro (%)": round(r.recall_macro, 2),
            "F1-score macro (%)": round(r.f1_macro, 2),
            "Training time (s)": (
                round(r.training_time_sec, 2)
                if r.training_time_sec is not None else None
            ),
        })
    return pd.DataFrame(rows)


def plot_metric_comparison(
    df: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    save_path: Optional[Path] = None,
    value_fmt: str = "{:.2f}",
    colors: tuple[str, ...] = ("#4C72B0", "#DD8452", "#55A868", "#C44E52"),
):
    """Ve bieu do cot so sanh 1 chi so (accuracy / loss / training time / ...) giua cac model."""
    fig, ax = plt.subplots(figsize=(7, 5))
    bar_colors = [colors[i % len(colors)] for i in range(len(df))]
    bars = ax.bar(df["Model"], df[column], color=bar_colors)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, df[column]):
        if pd.notna(val):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                value_fmt.format(val),
                ha="center", va="bottom", fontsize=9,
            )

    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def best_model_summary(df: pd.DataFrame) -> str:
    """Sinh vai dong nhan xet mo hinh tot nhat, dung de dua vao phan Ket luan."""
    best_acc_row = df.loc[df["Test accuracy (%)"].idxmax()]
    best_f1_row = df.loc[df["F1-score macro (%)"].idxmax()]

    lines = [
        f"- Mo hinh co Test Accuracy cao nhat: **{best_acc_row['Model']}** "
        f"({best_acc_row['Test accuracy (%)']:.2f}%).",
        f"- Mo hinh co F1-score macro cao nhat: **{best_f1_row['Model']}** "
        f"({best_f1_row['F1-score macro (%)']:.2f}%).",
    ]

    if df["Training time (s)"].notna().any():
        fastest_row = df.loc[df["Training time (s)"].idxmin()]
        lines.append(
            f"- Mo hinh huan luyen nhanh nhat: **{fastest_row['Model']}** "
            f"({fastest_row['Training time (s)']:.1f} giay)."
        )

    return "\n".join(lines)


# ============================================================
# 6. ORCHESTRATOR - DANH GIA CA 3 MODEL TRONG 1 LAN GOI
# ============================================================

def run_full_evaluation(
    specs: list[ModelSpec],
    test_loader,
    device: torch.device,
    class_names: list[str] = CIFAR10_CLASSES,
    figures_dir: Optional[Path] = None,
    tables_dir: Optional[Path] = None,
    n_correct_samples: int = 8,
    n_wrong_samples: int = 8,
) -> tuple[list[EvaluationResult], pd.DataFrame]:
    """
    Chay toan bo pipeline danh gia cho danh sach ModelSpec:
    1. Dung + load checkpoint tung model
    2. evaluate_model() tren test set
    3. Ve + luu confusion matrix, anh dung/sai
    4. Gop bang so sanh + luu metrics.csv (neu co tables_dir)
    5. Ve + luu bieu do so sanh accuracy / loss / training time (neu co figures_dir)

    Bat ky model nao thieu checkpoint se bi BO QUA (in canh bao),
    khong lam sap ca pipeline.
    """
    results: list[EvaluationResult] = []

    for spec in specs:
        print(f"\n===== DANH GIA: {spec.display_name} =====")
        try:
            model = build_and_load(spec, device)
        except FileNotFoundError as e:
            print(f"[BO QUA] {e}")
            continue

        result = evaluate_model(
            model, test_loader, device,
            class_names=class_names,
            model_name=spec.display_name,
            training_time_sec=spec.training_time_sec,
        )

        print(f"Test loss     : {result.test_loss:.4f}")
        print(f"Test accuracy : {result.test_accuracy:.2f}%")
        print(f"Precision (macro) : {result.precision_macro:.2f}%")
        print(f"Recall (macro)    : {result.recall_macro:.2f}%")
        print(f"F1-score (macro)  : {result.f1_macro:.2f}%")
        print("\n" + result.report_text)

        slug = spec.display_name.lower().replace(" ", "_").replace("-", "")

        cm_path = (figures_dir / f"{slug}_confusion_matrix.png") if figures_dir else None
        plot_confusion_matrix(result, class_names=class_names, save_path=cm_path)

        correct_path = (figures_dir / f"{slug}_correct_predictions.png") if figures_dir else None
        wrong_path = (figures_dir / f"{slug}_wrong_predictions.png") if figures_dir else None
        show_prediction_samples(
            model, test_loader, device,
            class_names=class_names, model_name=spec.display_name,
            n_correct=n_correct_samples, n_wrong=n_wrong_samples,
            save_path_correct=correct_path, save_path_wrong=wrong_path,
        )

        results.append(result)

    comparison_df = build_comparison_table(results)

    if tables_dir and not comparison_df.empty:
        tables_dir = Path(tables_dir)
        tables_dir.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(tables_dir / "metrics.csv", index=False)
        print(f"\nDa luu bang so sanh: {tables_dir / 'metrics.csv'}")

    if figures_dir and not comparison_df.empty:
        plot_metric_comparison(
            comparison_df, "Test accuracy (%)", "Test Accuracy (%)",
            "So sanh Test Accuracy giua 3 mo hinh",
            save_path=figures_dir / "model_comparison_accuracy.png",
        )
        plot_metric_comparison(
            comparison_df, "Test loss", "Test Loss",
            "So sanh Test Loss giua 3 mo hinh",
            save_path=figures_dir / "model_comparison_loss.png",
            value_fmt="{:.4f}",
        )
        if comparison_df["Training time (s)"].notna().any():
            plot_metric_comparison(
                comparison_df, "Training time (s)", "Training time (giay)",
                "So sanh thoi gian huan luyen giua 3 mo hinh",
                save_path=figures_dir / "model_comparison_training_time.png",
                value_fmt="{:.0f}",
            )

    return results, comparison_df
