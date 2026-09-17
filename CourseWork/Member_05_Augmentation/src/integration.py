"""Explicit member imports avoid collisions between generic src packages."""
import importlib
import importlib.util
import warnings
from pathlib import Path

from .contracts import sha256_file

MEMBER_DIR = Path(__file__).resolve().parents[1]
COURSEWORK_DIR = MEMBER_DIR.parent
DEFAULT_PROCESSED_DIR = COURSEWORK_DIR / "Member_02_Data/data/processed"
DEFAULT_ZIP_PATH = COURSEWORK_DIR / "Member_02_Data/data/data NIH xray14/archive.zip"


def dataset_class():
    # Normal namespace import remains picklable with spawned DataLoader workers.
    return importlib.import_module(
        "CourseWork.Member_02_Data.src.dataset"
    ).ZipChestXrayAgeDataset


def model_class():
    shared = COURSEWORK_DIR / "Member_03_CNN_Model/src/model.py"
    snapshot = MEMBER_DIR / "src/reference_model.py"
    path = shared if shared.is_file() else snapshot
    if path == snapshot:
        warnings.warn(
            "Member 03 src/model.py is absent on main; using its exact snapshot "
            "from linh-brand commit 736cbbc34de4287c8e44dea14731a6834f715ad2. "
            "Review/freeze the shared model before final training.",
            stacklevel=2,
        )
    spec = importlib.util.spec_from_file_location("member03_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load model: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AgeRegressionCNN, {
        "path": str(path.relative_to(COURSEWORK_DIR)),
        "sha256": sha256_file(path),
        "source": "member03" if path == shared else "member03_pinned_snapshot",
    }
