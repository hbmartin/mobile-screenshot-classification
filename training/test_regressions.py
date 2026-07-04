import importlib
import inspect
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_DIR.parent
sys.path.insert(0, str(TRAINING_DIR))

import dedupe
import split


class FakeDataset:
    def __init__(self, tensors):
        self.tensors = tensors

    def shuffle(self, *args, **kwargs):
        return self

    def map(self, *args, **kwargs):
        return self

    def batch(self, *args, **kwargs):
        return self

    def prefetch(self, *args, **kwargs):
        return self


def load_data_with_fake_tensorflow():
    def image_dataset_from_directory(*args, **kwargs):
        raise AssertionError("directory mode must not call Keras directory loading")

    fake_tf = types.SimpleNamespace(
        data=types.SimpleNamespace(
            AUTOTUNE=object(),
            Dataset=types.SimpleNamespace(
                from_tensor_slices=lambda tensors: FakeDataset(tensors)
            ),
        ),
        keras=types.SimpleNamespace(
            utils=types.SimpleNamespace(
                image_dataset_from_directory=image_dataset_from_directory
            )
        ),
    )
    sys.modules.pop("data", None)
    with mock.patch.dict(sys.modules, {"tensorflow": fake_tf}):
        return importlib.import_module("data")


def version_tuple(version):
    return tuple(int(part) for part in version.split("."))


class ReviewFeedbackRegressionTests(unittest.TestCase):
    def test_keras_pin_is_patched(self):
        requirements = (TRAINING_DIR / "requirements.txt").read_text(encoding="utf-8")
        match = re.search(r"^keras==([0-9.]+)$", requirements, re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertGreaterEqual(version_tuple(match.group(1)), (3, 13, 2))

    def test_semgrep_blocks_vulnerable_keras_pins(self):
        semgrep_config = (REPO_ROOT / ".semgrep.yml").read_text(encoding="utf-8")
        self.assertIn("blocked-vulnerable-training-pins", semgrep_config)
        self.assertIn("- training/requirements.txt", semgrep_config)
        self.assertNotIn("- /training/requirements.txt", semgrep_config)
        self.assertIn("keras==3.11.3", semgrep_config)
        self.assertIn("keras==3\\.13\\.[01]", semgrep_config)

    def test_directory_mode_avoids_keras_validation_split(self):
        data = load_data_with_fake_tensorflow()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for class_name in ("alpha", "beta"):
                class_dir = root / class_name
                class_dir.mkdir()
                for idx in range(4):
                    (class_dir / f"{idx}.png").write_bytes(b"image")

            cfg = {
                "seed": 123,
                "data": {
                    "dir": str(root),
                    "image_height": 16,
                    "image_width": 9,
                    "batch_size": 2,
                    "shuffle_buffer_size": 4,
                    "validation_split": 0.4,
                    "test_fraction": 0.5,
                },
            }

            train_ds, val_ds, test_ds, class_names = data._load_from_directory(cfg)

        self.assertIsInstance(train_ds, FakeDataset)
        self.assertIsInstance(val_ds, FakeDataset)
        self.assertIsInstance(test_ds, FakeDataset)
        self.assertEqual(class_names, ["alpha", "beta"])

    def test_directory_rows_skip_empty_classes(self):
        data = load_data_with_fake_tensorflow()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "empty").mkdir()
            (root / "text_only").mkdir()
            (root / "text_only" / "notes.txt").write_text("ignore", encoding="utf-8")
            (root / "with_images").mkdir()
            (root / "with_images" / "screen.png").write_bytes(b"image")

            rows, class_names = data._list_directory_rows(str(root))

        self.assertEqual(class_names, ["with_images"])
        self.assertEqual(
            [(Path(path).name, label) for path, label in rows],
            [("screen.png", 0)],
        )

    def test_manifest_train_split_must_not_be_empty(self):
        data = load_data_with_fake_tensorflow()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.csv"
            manifest.write_text(
                "path,class,split\nval.png,alpha,val\ntest.png,alpha,test\n",
                encoding="utf-8",
            )
            cfg = {
                "seed": 123,
                "data": {
                    "manifest": str(manifest),
                    "image_height": 16,
                    "image_width": 9,
                    "batch_size": 2,
                    "shuffle_buffer_size": 4,
                },
            }

            with self.assertRaisesRegex(
                ValueError, "Manifest split 'train' produced no images"
            ):
                data._load_from_manifest(cfg)

    def test_fractional_shuffle_buffer_size_is_rejected(self):
        data = load_data_with_fake_tensorflow()
        for configured in (1.5, "1.5"):
            with self.subTest(configured=configured):
                with self.assertRaisesRegex(
                    ValueError, "data.shuffle_buffer_size must be a positive integer"
                ):
                    data._shuffle_buffer_size({"shuffle_buffer_size": configured}, 10)

        self.assertEqual(
            data._shuffle_buffer_size({"shuffle_buffer_size": "2"}, 10), 2
        )

    def test_manifest_paths_use_posix_separators(self):
        with mock.patch.object(split.os, "sep", "\\"):
            self.assertEqual(
                split.to_manifest_path(r"..\training-v1\screenshots\app\shot.png"),
                "../training-v1/screenshots/app/shot.png",
            )

    def test_hamming_uses_int_bit_count(self):
        source = inspect.getsource(dedupe.hamming)
        self.assertIn(".bit_count()", source)
        self.assertEqual(dedupe.hamming(0b1010, 0b0011), 2)


if __name__ == "__main__":
    unittest.main()
