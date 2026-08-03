from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    project_dir: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = project_dir / "data"
    checkpoint_dir: Path = project_dir / "checkpoints"
    run_dir: Path = project_dir / "runs"
    result_dir: Path = project_dir / "results"

    seed: int = 42
    num_classes: int = 10
    image_size: int = 224
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 10
    num_workers: int = 2
