from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def find_metadata_member(z: zipfile.ZipFile) -> str:
    """
    Tìm Data_Entry_2017.csv hoặc Data_Entry_2017_v2020.csv
    ở bất kỳ thư mục nào bên trong ZIP.
    """
    candidates = [
        name
        for name in z.namelist()
        if Path(name).name.lower()
        in {
            "data_entry_2017.csv",
            "data_entry_2017_v2020.csv",
        }
    ]

    if not candidates:
        csv_files = [
            name
            for name in z.namelist()
            if name.lower().endswith(".csv")
        ]

        raise FileNotFoundError(
            "Không tìm thấy Data_Entry_2017.csv trong ZIP.\n"
            f"Các file CSV tìm thấy: {csv_files[:20]}"
        )

    # Ưu tiên Data_Entry_2017.csv
    candidates.sort(
        key=lambda x: (
            0
            if Path(x).name.lower() == "data_entry_2017.csv"
            else 1
        )
    )

    return candidates[0]


def load_metadata_from_zip(
    zip_path: str | Path,
) -> pd.DataFrame:
    """
    Đọc metadata trực tiếp từ file ZIP.
    KHÔNG giải nén toàn bộ ZIP.
    """
    zip_path = Path(zip_path)

    if not zip_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy ZIP: {zip_path}"
        )

    with zipfile.ZipFile(zip_path, "r") as z:
        metadata_member = find_metadata_member(z)

        print(
            f"Reading metadata from ZIP: "
            f"{metadata_member}"
        )

        with z.open(metadata_member) as f:
            df = pd.read_csv(f)

    return df


def clean_age(
    df: pd.DataFrame,
    min_age: float = 1,
    max_age: float = 100,
) -> pd.DataFrame:
    """
    Giữ lại các cột cần thiết và làm sạch Patient Age.
    """

    required_columns = [
        "Image Index",
        "Patient ID",
        "Patient Age",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Thiếu các cột: {missing}"
        )

    result = df[
        required_columns
    ].copy()

    # Chuyển Patient Age sang numeric
    result["Patient Age"] = pd.to_numeric(
        result["Patient Age"],
        errors="coerce",
    )

    # Xóa NaN
    result = result.dropna(
        subset=["Patient Age"]
    ).copy()

    # Giữ tuổi hợp lệ
    result = result[
        result["Patient Age"].between(
            min_age,
            max_age,
        )
    ].copy()

    result["Image Index"] = (
        result["Image Index"].astype(str)
    )

    result["Patient ID"] = (
        result["Patient ID"].astype(str)
    )

    return result


def build_image_index(
    zip_path: str | Path,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Đọc danh sách ảnh nằm bên trong ZIP.

    Không giải nén ảnh.
    Chỉ lấy:
        Image Index -> zip_member
    """

    zip_path = Path(zip_path)

    records = []
    seen = set()
    duplicates = []

    with zipfile.ZipFile(zip_path, "r") as z:

        for member in z.namelist():

            suffix = Path(member).suffix.lower()

            if suffix not in {
                ".png",
                ".jpg",
                ".jpeg",
            }:
                continue

            filename = Path(member).name

            if filename in seen:
                duplicates.append(filename)
            else:
                seen.add(filename)

                records.append(
                    {
                        "Image Index": filename,
                        "zip_member": member,
                    }
                )

    if duplicates:
        raise ValueError(
            "Phát hiện duplicate image filename.\n"
            f"Ví dụ: {duplicates[:10]}"
        )

    index_df = pd.DataFrame(records)

    if index_df.empty:
        raise FileNotFoundError(
            "Không tìm thấy file ảnh trong ZIP."
        )

    if output_csv is not None:

        output_csv = Path(output_csv)

        output_csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        index_df.to_csv(
            output_csv,
            index=False,
        )

    return index_df


def attach_zip_paths(
    df: pd.DataFrame,
    image_index: pd.DataFrame,
) -> Tuple[pd.DataFrame, int]:
    """
    Ghép metadata với path của ảnh bên trong ZIP.
    """

    result = df.merge(
        image_index,
        on="Image Index",
        how="left",
        validate="many_to_one",
    )

    missing = int(
        result["zip_member"].isna().sum()
    )

    result = result.dropna(
        subset=["zip_member"]
    ).copy()

    return result, missing


def patient_level_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    """
    Split theo Patient ID, KHÔNG split theo image.
    """

    total = (
        train_ratio
        + val_ratio
        + test_ratio
    )

    if not np.isclose(
        total,
        1.0,
    ):
        raise ValueError(
            "Train/Val/Test ratios phải tổng bằng 1."
        )

    patients = (
        df["Patient ID"]
        .drop_duplicates()
        .to_numpy()
    )

    train_patients, temp_patients = (
        train_test_split(
            patients,
            test_size=1 - train_ratio,
            random_state=seed,
        )
    )

    test_fraction = (
        test_ratio
        / (val_ratio + test_ratio)
    )

    val_patients, test_patients = (
        train_test_split(
            temp_patients,
            test_size=test_fraction,
            random_state=seed,
        )
    )

    train_set = set(train_patients)
    val_set = set(val_patients)
    test_set = set(test_patients)

    # Kiểm tra leakage
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)

    train_df = df[
        df["Patient ID"].isin(train_set)
    ].copy()

    val_df = df[
        df["Patient ID"].isin(val_set)
    ].copy()

    test_df = df[
        df["Patient ID"].isin(test_set)
    ].copy()

    return (
        train_df,
        val_df,
        test_df,
    )


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Lưu train.csv / val.csv / test.csv
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "Image Index",
        "Patient ID",
        "Patient Age",
        "zip_member",
    ]

    train_df[columns].to_csv(
        output_dir / "train.csv",
        index=False,
    )

    val_df[columns].to_csv(
        output_dir / "val.csv",
        index=False,
    )

    test_df[columns].to_csv(
        output_dir / "test.csv",
        index=False,
    )


def split_summary(
    train_df,
    val_df,
    test_df,
):
    """
    Tổng hợp thông tin train/val/test.
    """

    train_patients = set(
        train_df["Patient ID"]
    )

    val_patients = set(
        val_df["Patient ID"]
    )

    test_patients = set(
        test_df["Patient ID"]
    )

    return {
        "train_images": len(train_df),
        "val_images": len(val_df),
        "test_images": len(test_df),

        "train_patients": len(
            train_patients
        ),

        "val_patients": len(
            val_patients
        ),

        "test_patients": len(
            test_patients
        ),

        "train_val_overlap": len(
            train_patients
            & val_patients
        ),

        "train_test_overlap": len(
            train_patients
            & test_patients
        ),

        "val_test_overlap": len(
            val_patients
            & test_patients
        ),
    }