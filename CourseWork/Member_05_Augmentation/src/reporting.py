"""Generate real-image examples and paired validation reports."""
import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


def save_examples(train_csv, zip_path, output_path, seed=42, n_images=4, n_variants=3):
    """Read training images only. Save baseline and mild affine variants."""
    import torch
    from .transforms import build_transform

    if n_images < 1 or n_variants < 1:
        raise ValueError("n_images and n_variants must be positive")
    frame = pd.read_csv(train_csv)
    if frame.empty:
        raise ValueError("Training split is empty")
    frame = frame.sample(n=min(n_images, len(frame)), random_state=seed)
    baseline, augmented = build_transform("train"), build_transform("train", True)
    fig, axes = plt.subplots(
        len(frame), n_variants + 1, figsize=(3 * (n_variants + 1), 3 * len(frame)),
        squeeze=False, constrained_layout=True,
    )
    try:
        # Preserve RNG state so visualization does not alter later experiments.
        with torch.random.fork_rng(), zipfile.ZipFile(zip_path) as archive:
            torch.manual_seed(seed)
            for row_index, (_, row) in enumerate(frame.iterrows()):
                with archive.open(str(row["zip_member"])) as handle:
                    original = Image.open(io.BytesIO(handle.read())).convert("L")
                versions = [baseline(original)] + [
                    augmented(original) for _ in range(n_variants)
                ]
                for column, tensor in enumerate(versions):
                    axes[row_index, column].imshow(
                        (tensor[0].numpy() * 0.5 + 0.5).clip(0, 1),
                        cmap="gray", vmin=0, vmax=1,
                    )
                    label = "Baseline" if column == 0 else f"Augmentation {column}"
                    axes[row_index, column].set_title(
                        f"{label} | age={row['Patient Age']:g}"
                    )
                    axes[row_index, column].axis("off")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
    finally:
        plt.close(fig)
    return path


def write_reports(run_dir):
    run_dir = Path(run_dir)
    results = pd.read_csv(run_dir / "validation_results.csv").set_index("experiment")
    if set(results.index) != {"E1", "E2", "E3", "E4"} or len(results) != 4:
        raise ValueError("A complete, unique E1–E4 matrix is required")
    if not np.isfinite(results[["val_mae", "val_mse", "val_rmse"]].to_numpy()).all():
        raise ValueError("Metrics must be finite")
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    paired = []
    for baseline, augmented in (("E1", "E3"), ("E2", "E4")):
        base, aug = results.loc[baseline], results.loc[augmented]
        delta = float(aug.val_mae - base.val_mae)
        paired.append({
            "baseline": baseline, "augmented": augmented, "loss": base.loss,
            "baseline_val_mae": base.val_mae, "augmented_val_mae": aug.val_mae,
            "delta_val_mae": delta,
            "relative_improvement_percent": (
                100 * -delta / base.val_mae if base.val_mae else None
            ),
        })
    comparison = pd.DataFrame(paired)
    comparison.to_csv(run_dir / "augmentation_comparison.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, experiment in zip(axes.flat, ("E1", "E2", "E3", "E4")):
        history = pd.read_csv(run_dir / experiment / "history.csv")
        axis.plot(history.epoch, history.train_loss, label="Train")
        axis.plot(history.epoch, history.val_loss, label="Validation")
        loss_name = results.loc[experiment, "loss"]
        axis.set(
            title=f"{experiment}: {loss_name}", xlabel="Epoch",
            ylabel="MSE (years²)" if loss_name == "MSE" else "MAE (years)",
        )
        axis.legend()
        axis.grid(alpha=0.2)
    fig.savefig(figures / "train_validation_loss.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for experiment in ("E1", "E2", "E3", "E4"):
        history = pd.read_csv(run_dir / experiment / "history.csv")
        axis.plot(history.epoch, history.val_mae, label=experiment)
    axis.set(
        xlabel="Epoch", ylabel="Validation MAE (years)",
        title="E1–E4: same validation set",
    )
    axis.legend()
    axis.grid(alpha=0.2)
    fig.savefig(figures / "validation_mae.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.bar(results.index, results.val_mae)
    axis.set(
        ylabel="Validation MAE (years)", title="Best validation checkpoint per experiment"
    )
    fig.savefig(figures / "validation_comparison.png", dpi=160)
    plt.close(fig)
    lines = [
        "# Nhận xét augmentation trên validation", "",
        "ΔMAE = MAE có augmentation − MAE baseline; giá trị âm là cải thiện.", "",
    ]
    for row in paired:
        delta = row["delta_val_mae"]
        direction = "giảm" if delta < 0 else "tăng" if delta > 0 else "không đổi"
        lines.append(
            f"- {row['augmented']} so với {row['baseline']}: "
            f"validation MAE {direction}, Δ = {delta:.4f} năm."
        )
    lines.extend([
        "",
        "Đây là một seed trên validation; chưa đủ kết luận ý nghĩa thống kê.",
        "Train MAE của E3/E4 đo trên ảnh đã biến đổi; không so trực tiếp độ chênh "
        "train–val giữa các pipeline.",
        "Member 06 đánh giá checkpoint đã chốt trên test; không chỉnh augmentation theo test.",
        "Kiểm tra hình trước/sau và các trường hợp bị cắt vùng phổi trước khi diễn giải.",
    ])
    (run_dir / "discussion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return comparison
