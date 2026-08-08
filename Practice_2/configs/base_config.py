from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # Thư mục gốc của Practice_2
    project_dir: Path = Path(__file__).resolve().parents[1]

    # Thư mục lưu dữ liệu
    data_dir: Path = project_dir / "data"

    # Cấu hình tái lập kết quả
    seed: int = 42

    # CIFAR-10 có 10 lớp
    num_classes: int = 10

    # Các mô hình pretrained thường nhận ảnh 224 × 224
    image_size: int = 224

    # Cấu hình DataLoader
    batch_size: int = 32
    num_workers: int = 2

    # Tỷ lệ validation lấy từ tập train gốc
    val_ratio: float = 0.10