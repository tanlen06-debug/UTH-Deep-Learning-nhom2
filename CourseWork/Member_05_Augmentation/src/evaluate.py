"""Explicit Member 06 handoff after the recipe/checkpoints have been frozen."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from .contracts import sha256_file, validate_splits
from .experiments import Config, make_loader, run_epoch, write_json
from .integration import model_class


def evaluate_frozen_run(run_dir, zip_path, processed_dir, device="cpu"):
    """No training/tuning here. Refuse changed model, splits or checkpoints."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "completed":
        raise ValueError("Finish the complete E1–E4 run before final evaluation")
    destination = run_dir / "test_evaluation"
    if destination.exists():
        raise FileExistsError("Test evaluation already exists; do not silently rerun/tune")
    audit = validate_splits(processed_dir)
    if audit != manifest["splits"]:
        raise ValueError("CSV split fingerprints changed since training")
    model_type, model_info = model_class()
    if model_info["sha256"] != manifest["model"]["sha256"]:
        raise ValueError("Model source changed since training; restore the frozen source")
    results = pd.read_csv(run_dir / "validation_results.csv")
    if len(results) != 4 or set(results.experiment) != {"E1", "E2", "E3", "E4"}:
        raise ValueError("Missing or duplicated experiment")
    # Check all checkpoint fingerprints before opening any test image.
    for row in results.itertuples(index=False):
        if sha256_file(run_dir / row.checkpoint) != row.checkpoint_sha256:
            raise ValueError(f"Checkpoint changed: {row.experiment}")
    config = replace(Config(**manifest["config"]), device=device)
    destination.mkdir()
    write_json(destination / "status.json", {"status": "running", "selection": "validation only"})
    loader = make_loader(Path(processed_dir) / "test.csv", zip_path, "test", False, config)
    rows = []
    try:
        for row in results.itertuples(index=False):
            model = model_type().to(device)
            state = torch.load(run_dir / row.checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state)
            criterion = nn.MSELoss() if row.loss == "MSE" else nn.L1Loss()
            metrics = run_epoch(model, loader, criterion, torch.device(device))
            rows.append({
                "experiment": row.experiment, "best_epoch": row.best_epoch,
                **{f"test_{key}": metrics[key] for key in ("mae", "mse", "rmse")},
            })
    finally:
        loader.dataset.close()
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "test_results.csv", index=False)
    write_json(destination / "status.json", {"status": "completed", "selection": "validation only"})
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    evaluate_frozen_run(args.run_dir, args.zip_path, args.processed_dir, args.device)
