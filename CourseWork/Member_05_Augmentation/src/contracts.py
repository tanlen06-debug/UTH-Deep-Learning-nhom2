"""Validate frozen Member 02 metadata without opening test images."""
import csv
import hashlib
import math
from itertools import combinations
from pathlib import Path

REQUIRED = {"Image Index", "Patient ID", "Patient Age", "zip_member"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_splits(processed_dir):
    """Reject invalid metadata and leakage; never resplit, filter or reorder."""
    groups, summary = {}, {}
    for split in ("train", "val", "test"):
        path = Path(processed_dir) / f"{split}.csv"
        patients, images, members = set(), set(), set()
        count = 0
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path}: missing columns {sorted(missing)}")
            for row in reader:
                count += 1
                if any(not str(row.get(key) or "").strip() for key in REQUIRED):
                    raise ValueError(f"{path}: empty value at row {count + 1}")
                age = float(row["Patient Age"])
                if not math.isfinite(age) or not 1 <= age <= 100:
                    raise ValueError(f"{path}: invalid age at row {count + 1}")
                patient = int(row["Patient ID"])
                if patient < 0:
                    raise ValueError(f"{path}: invalid Patient ID")
                image, member = row["Image Index"].strip(), row["zip_member"].strip()
                if image in images or member in members:
                    raise ValueError(f"{path}: duplicate image/ZIP member: {image}")
                if Path(member).name != image:
                    raise ValueError(f"{path}: image/ZIP member mismatch: {image}")
                patients.add(patient)
                images.add(image)
                members.add(member)
        if not count:
            raise ValueError(f"{path}: empty split")
        groups[split] = (patients, images, members)
        summary[split] = {
            "images": count, "patients": len(patients), "sha256": sha256_file(path)
        }
    overlaps = {}
    for left, right in combinations(groups, 2):
        counts = [len(a & b) for a, b in zip(groups[left], groups[right])]
        if any(counts):
            raise ValueError(f"Leakage {left}/{right}: patients/images/members={counts}")
        overlaps[f"{left}_{right}"] = 0
    return {"splits": summary, "overlap": overlaps, "test_images_read": False}
