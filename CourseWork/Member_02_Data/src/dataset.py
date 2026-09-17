from __future__ import annotations
import io
import zipfile
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def default_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

class ZipChestXrayAgeDataset(Dataset):
    """Read X-ray images directly from archive.zip without extracting the archive."""
    def __init__(self, csv_file, zip_path, transform=None):
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        required = {"Image Index", "Patient ID", "Patient Age", "zip_member"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        self.zip_path = Path(zip_path)
        if not self.zip_path.exists():
            raise FileNotFoundError(f"ZIP not found: {self.zip_path}")
        self.transform = transform or default_transform()
        self._zip = None

    def _get_zip(self):
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path, "r")
        return self._zip

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        member = str(row["zip_member"])
        with self._get_zip().open(member, "r") as f:
            image = Image.open(io.BytesIO(f.read())).convert("L")
        if self.transform:
            image = self.transform(image)
        age = torch.tensor([float(row["Patient Age"])], dtype=torch.float32)
        return image, age

    def close(self):
        if self._zip is not None:
            self._zip.close()
            self._zip = None

def make_dataloaders(train_csv, val_csv, test_csv, zip_path, batch_size=32, num_workers=0):
    train_ds = ZipChestXrayAgeDataset(train_csv, zip_path)
    val_ds = ZipChestXrayAgeDataset(val_csv, zip_path)
    test_ds = ZipChestXrayAgeDataset(test_csv, zip_path)
    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train_ds, shuffle=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )
