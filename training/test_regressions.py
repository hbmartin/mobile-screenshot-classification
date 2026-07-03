import inspect
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

TRAINING_DIR = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_DIR.parent
sys.path.insert(0, str(TRAINING_DIR))

import dedupe
import split


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
        self.assertIn("keras==3.11.3", semgrep_config)
        self.assertIn("keras==3\\.13\\.[01]", semgrep_config)

    def test_directory_mode_avoids_keras_validation_split(self):
        source = (TRAINING_DIR / "data.py").read_text(encoding="utf-8")
        load_from_directory = source.split("def _load_from_directory", 1)[1].split(
            "\ndef read_manifest", 1
        )[0]
        self.assertNotIn("image_dataset_from_directory", load_from_directory)
        self.assertIn("_shuffle_buffer_size", load_from_directory)

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
