"""Run from repository root: python -m unittest discover -s CourseWork/Member_05_Augmentation/tests -v"""
import csv
import importlib.util
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from CourseWork.Member_05_Augmentation.src.contracts import validate_splits

FIELDS = ["Image Index", "Patient ID", "Patient Age", "zip_member"]
HAS_TORCH = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("torchvision") is not None
)


def write_split(root, split, rows):
    with (root / f"{split}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for index, split in enumerate(("train", "val", "test"), 1):
            write_split(self.root, split, [
                [f"{index}.png", index, 20 + index, f"images/{index}.png"]
            ])

    def test_valid_disjoint_splits(self):
        audit = validate_splits(self.root)
        self.assertEqual(audit["splits"]["train"]["images"], 1)
        self.assertEqual(audit["overlap"]["train_val"], 0)
        self.assertFalse(audit["test_images_read"])

    def test_patient_leakage_with_leading_zeros(self):
        write_split(self.root, "val", [["2.png", "001", 22, "images/2.png"]])
        with self.assertRaisesRegex(ValueError, "Leakage"):
            validate_splits(self.root)

    def test_cross_split_image_leakage(self):
        write_split(self.root, "val", [["1.png", 2, 22, "other/1.png"]])
        with self.assertRaisesRegex(ValueError, "Leakage"):
            validate_splits(self.root)

    def test_duplicate_image(self):
        row = ["1.png", 1, 21, "images/1.png"]
        write_split(self.root, "train", [row, row])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_splits(self.root)

    def test_invalid_age(self):
        for age in ("nan", "inf", -1, 101):
            with self.subTest(age=age):
                write_split(self.root, "train", [["1.png", 1, age, "images/1.png"]])
                with self.assertRaisesRegex(ValueError, "invalid age"):
                    validate_splits(self.root)

    def test_empty_split(self):
        write_split(self.root, "val", [])
        with self.assertRaisesRegex(ValueError, "empty split"):
            validate_splits(self.root)

    def test_missing_column(self):
        (self.root / "train.csv").write_text("Image Index\n1.png\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing columns"):
            validate_splits(self.root)

    def test_changed_metadata_changes_fingerprint(self):
        before = validate_splits(self.root)
        write_split(self.root, "train", [["1.png", 1, 22, "images/1.png"]])
        after = validate_splits(self.root)
        self.assertNotEqual(
            before["splits"]["train"]["sha256"], after["splits"]["train"]["sha256"]
        )


@unittest.skipUnless(HAS_TORCH, "Install PyTorch/torchvision to execute runtime tests")
class RuntimeTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch
        from PIL import Image
        self.torch = torch
        self.image = Image.fromarray(
            np.arange(256 * 256, dtype=np.uint8).reshape(256, 256)
        )

    def test_baseline_exactly_matches_member02(self):
        from CourseWork.Member_02_Data.src.dataset import default_transform
        from CourseWork.Member_05_Augmentation.src.transforms import build_transform
        self.assertTrue(self.torch.equal(
            build_transform("train")(self.image), default_transform()(self.image)
        ))

    def test_validation_and_test_are_deterministic_even_when_augment_true(self):
        from CourseWork.Member_05_Augmentation.src.transforms import build_transform
        for split in ("val", "test"):
            transform = build_transform(split, True)
            self.assertTrue(self.torch.equal(transform(self.image), transform(self.image)))
            self.assertTrue(self.torch.equal(
                transform(self.image), build_transform(split, False)(self.image)
            ))
            self.assertEqual(tuple(transform(self.image).shape), (1, 224, 224))

    def test_training_is_random_but_seed_reproducible(self):
        from CourseWork.Member_05_Augmentation.src.transforms import build_transform
        transform = build_transform("train", True)
        self.torch.manual_seed(42)
        first, second = transform(self.image), transform(self.image)
        self.torch.manual_seed(42)
        self.assertTrue(self.torch.equal(first, transform(self.image)))
        self.assertFalse(self.torch.equal(first, second))
        self.assertGreaterEqual(first.min().item(), -1)
        self.assertLessEqual(first.max().item(), 1)

    def test_augmentation_does_not_change_sampler_order(self):
        from torch.utils.data import DataLoader, Dataset
        from CourseWork.Member_05_Augmentation.src.transforms import build_transform
        image = self.image
        class Indexed(Dataset):
            def __init__(self, augment):
                self.transform = build_transform("train", augment)
            def __len__(self):
                return 9
            def __getitem__(self, index):
                return self.transform(image), index
        orders = []
        for augment in (False, True):
            loader = DataLoader(
                Indexed(augment), batch_size=4, shuffle=True,
                generator=self.torch.Generator().manual_seed(42),
            )
            orders.append([index for _, ids in loader for index in ids.tolist()])
        self.assertEqual(*orders)

    def test_reference_model_contract(self):
        from CourseWork.Member_05_Augmentation.src.reference_model import AgeRegressionCNN
        model = AgeRegressionCNN().eval()
        with self.torch.no_grad():
            result = model(self.torch.zeros(2, 1, 224, 224))
        self.assertEqual(tuple(result.shape), (2, 1))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 389057)

    def test_metrics_weight_last_batch_by_samples(self):
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
        from CourseWork.Member_05_Augmentation.src.experiments import run_epoch
        torch = self.torch
        class ZeroModel(nn.Module):
            def forward(self, x):
                return torch.zeros(x.shape[0], 1, device=x.device)
        loader = DataLoader(
            TensorDataset(torch.zeros(3, 1, 224, 224), torch.tensor([[1.], [2.], [9.]])),
            batch_size=2,
        )
        metrics = run_epoch(ZeroModel(), loader, nn.MSELoss(), torch.device("cpu"))
        self.assertAlmostEqual(metrics["mae"], 4)
        self.assertAlmostEqual(metrics["mse"], 86 / 3)
        self.assertAlmostEqual(metrics["rmse"], (86 / 3) ** 0.5)

    def test_complete_matrix_on_tiny_synthetic_fixture(self):
        """Synthetic pixels check plumbing only; never report them as NIH results."""
        import numpy as np
        from PIL import Image
        from CourseWork.Member_05_Augmentation.src.experiments import Config, run_matrix
        from CourseWork.Member_05_Augmentation.src.reporting import save_examples
        torch = self.torch
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                archive_path = root / "fixture.zip"
                with zipfile.ZipFile(archive_path, "w") as archive:
                    for offset, split in enumerate(("train", "val", "test")):
                        rows = []
                        for index in range(2):
                            pid = offset * 2 + index + 1
                            name = f"{pid}.png"
                            rows.append([name, pid, 20 + pid, f"images/{name}"])
                            # Omit test images deliberately: training must never read them.
                            if split != "test":
                                buffer = io.BytesIO()
                                image = Image.fromarray(
                                    np.random.default_rng(pid).integers(0, 256, (64, 64), dtype=np.uint8)
                                )
                                image.save(buffer, format="PNG")
                                archive.writestr(f"images/{name}", buffer.getvalue())
                        write_split(root, split, rows)
                results = run_matrix(
                    archive_path, root, root / "run",
                    Config(epochs=1, batch_size=2, device="cpu"),
                )
                self.assertEqual(results.experiment.tolist(), ["E1", "E2", "E3", "E4"])
                manifest = json.loads((root / "run/manifest.json").read_text())
                self.assertEqual(manifest["status"], "completed")
                self.assertFalse(manifest["test_evaluated"])
                for experiment in results.experiment:
                    self.assertTrue((root / "run" / experiment / "best_model.pt").is_file())
                self.assertTrue((root / "run/augmentation_comparison.csv").is_file())
                path = save_examples(root / "train.csv", archive_path, root / "examples.png")
                self.assertTrue(path.is_file())
                with self.assertRaises(FileExistsError):
                    run_matrix(archive_path, root, root / "run", Config(epochs=1))
        finally:
            torch.set_num_threads(old_threads)


if __name__ == "__main__":
    unittest.main()
