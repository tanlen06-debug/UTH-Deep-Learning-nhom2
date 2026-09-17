"""Controlled E1–E4 runner. Checkpoint selection uses validation MAE only."""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from .contracts import sha256_file, validate_splits
from .integration import (
    DEFAULT_PROCESSED_DIR, DEFAULT_ZIP_PATH, MEMBER_DIR, dataset_class, model_class,
)
from .transforms import AUGMENTATION, build_transform

EXPERIMENTS = {
    "E1": ("MSE", False), "E2": ("MAE", False),
    "E3": ("MSE", True), "E4": ("MAE", True),
}


@dataclass(frozen=True)
class Config:
    seed: int = 42
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "auto"

    def __post_init__(self):
        if self.image_size != 224:
            raise ValueError("Coursework input is [B,1,224,224]")
        if self.batch_size < 1 or self.epochs < 1 or self.num_workers < 0:
            raise ValueError("Invalid batch_size, epochs or num_workers")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative and finite")


def set_seed(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(_worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def make_loader(csv_path, zip_path, split, augment, config):
    dataset = dataset_class()(
        csv_path, zip_path,
        transform=build_transform(split, augment, config.image_size),
    )
    # A separate generator keeps affine random draws independent of batch order.
    return DataLoader(
        dataset, batch_size=config.batch_size, shuffle=(split == "train"),
        num_workers=config.num_workers, worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(config.seed),
        pin_memory=torch.cuda.is_available(), drop_last=False,
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    count, loss_sum, absolute_sum, square_sum = 0, 0.0, 0.0, 0.0
    with torch.set_grad_enabled(training):
        for images, ages in loader:
            images, ages = images.to(device), ages.to(device)
            if images.ndim != 4 or tuple(images.shape[1:]) != (1, 224, 224):
                raise ValueError(f"Expected [B,1,224,224], got {tuple(images.shape)}")
            predictions = model(images)
            if predictions.shape != ages.shape or tuple(ages.shape[1:]) != (1,):
                raise ValueError("Predictions and ages must both be [B,1]")
            if not torch.isfinite(predictions).all() or not torch.isfinite(ages).all():
                raise ValueError("Non-finite prediction or target")
            loss = criterion(predictions, ages)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            errors = (predictions.detach() - ages).double()
            count += ages.numel()
            loss_sum += loss.item() * ages.numel()
            absolute_sum += errors.abs().sum().item()
            square_sum += errors.square().sum().item()
    if count == 0:
        raise ValueError("Cannot evaluate an empty loader")
    return {
        "loss": loss_sum / count, "mae": absolute_sum / count,
        "mse": square_sum / count, "rmse": (square_sum / count) ** 0.5,
    }


def state_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def check_archive(zip_path, processed_dir):
    if not Path(zip_path).is_file():
        raise FileNotFoundError(
            f"NIH archive.zip not found: {zip_path}. Set --zip-path to Member 02's archive."
        )
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        for split in ("train", "val"):
            path = Path(processed_dir) / f"{split}.csv"
            with path.open(encoding="utf-8-sig", newline="") as handle:
                missing = [
                    r["zip_member"] for r in csv.DictReader(handle)
                    if r["zip_member"] not in names
                ]
            if missing:
                raise FileNotFoundError(
                    f"{split}: {len(missing)} missing ZIP members; examples: {missing[:3]}"
                )


def run_matrix(zip_path, processed_dir=DEFAULT_PROCESSED_DIR, output_dir=None, config=None):
    """Run E1–E4 from scratch; never read test images or import unrelated baselines."""
    config = config or Config()
    zip_path, processed_dir = Path(zip_path), Path(processed_dir)
    output_dir = Path(output_dir or MEMBER_DIR / "results/run_seed42")
    audit = validate_splits(processed_dir)
    check_archive(zip_path, processed_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {output_dir}")
    model_type, model_info = model_class()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.device == "auto" else torch.device(config.device)
    )
    set_seed(config.seed)
    initial_model = model_type()
    initial_state = {
        k: v.detach().cpu().clone() for k, v in initial_model.state_dict().items()
    }
    initial_model.eval()
    with torch.no_grad():
        if tuple(initial_model(torch.zeros(2, 1, 224, 224)).shape) != (2, 1):
            raise ValueError("Model violates the shared output contract")
    manifest = {
        "status": "running", "dataset": "NIH ChestX-ray14", "target": "Patient Age",
        "config": asdict(config), "optimizer": "Adam", "scheduler": None,
        "selection_metric": "val_mae", "selection_ties": "earliest epoch",
        "model": model_info, "initial_state_sha256": state_digest(initial_state),
        "trainable_parameters": sum(
            p.numel() for p in initial_model.parameters() if p.requires_grad
        ),
        "splits": audit, "augmentation": AUGMENTATION,
        "zip_path": str(zip_path.resolve()), "processed_dir": str(processed_dir.resolve()),
        "environment": {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "device": str(device),
        },
        "test_evaluated": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "manifest.json", manifest)
    from .reporting import save_examples
    save_examples(
        processed_dir / "train.csv", zip_path,
        output_dir / "figures/augmentation_examples.png", seed=config.seed,
    )
    rows = []
    for experiment, (loss_name, augment) in EXPERIMENTS.items():
        set_seed(config.seed)
        model = model_type().to(device)
        model.load_state_dict(initial_state)
        criterion = nn.MSELoss() if loss_name == "MSE" else nn.L1Loss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        train = make_loader(processed_dir / "train.csv", zip_path, "train", augment, config)
        val = make_loader(processed_dir / "val.csv", zip_path, "val", augment, config)
        directory = output_dir / experiment
        directory.mkdir()
        history, best, best_epoch, best_metrics = [], float("inf"), None, None
        started = time.perf_counter()
        try:
            for epoch in range(1, config.epochs + 1):
                train_metrics = run_epoch(model, train, criterion, device, optimizer)
                val_metrics = run_epoch(model, val, criterion, device)
                history.append({
                    "epoch": epoch,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                })
                pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
                if val_metrics["mae"] < best:
                    best, best_epoch, best_metrics = val_metrics["mae"], epoch, val_metrics
                    torch.save({
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }, directory / "best_model.pt")
                print(
                    f"{experiment} epoch {epoch}/{config.epochs}: "
                    f"train_loss={train_metrics['loss']:.4f}, val_mae={val_metrics['mae']:.4f}",
                    flush=True,
                )
        finally:
            train.dataset.close()
            val.dataset.close()
        row = {
            "experiment": experiment, "loss": loss_name, "augmentation": augment,
            "best_epoch": best_epoch, "val_mae": best_metrics["mae"],
            "val_mse": best_metrics["mse"], "val_rmse": best_metrics["rmse"],
            "training_seconds": time.perf_counter() - started,
            "checkpoint": f"{experiment}/best_model.pt",
            "checkpoint_sha256": sha256_file(directory / "best_model.pt"),
        }
        rows.append(row)
        write_json(directory / "result.json", row)
        pd.DataFrame(rows).to_csv(output_dir / "validation_results.csv", index=False)
    from .reporting import write_reports
    write_reports(output_dir)
    manifest["status"] = "completed"
    write_json(output_dir / "manifest.json", manifest)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", type=Path, default=DEFAULT_ZIP_PATH)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = Config(
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        num_workers=args.num_workers, device=args.device,
    )
    run_matrix(args.zip_path, args.processed_dir, args.output_dir, config)


if __name__ == "__main__":
    main()
