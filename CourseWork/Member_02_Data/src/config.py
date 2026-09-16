from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = PROJECT_ROOT / "data" / "data NIH xray14" / "archive.zip"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "val.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"
IMAGE_INDEX_CSV = PROCESSED_DIR / "zip_image_index.csv"

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
IMAGE_SIZE = 224
