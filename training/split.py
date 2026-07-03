"""Create a fixed, stratified train/val/test manifest for the dataset.

A committed manifest makes the test set stable across runs and machines,
instead of depending on whatever random split a training run happens to
draw. Run dedupe.py first so near-duplicates can't straddle splits.

Usage:
    python split.py [--data-dir ../training-v1/screenshots] [--out manifest.csv]

Then point data.manifest in config.yaml at the manifest file.
"""

import argparse
import csv
import os
import random
from collections import Counter

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def clamp_holdout_counts(n_test, n_val, total):
    """Leave at least one image for training whenever a class has images."""
    max_holdout = max(0, total - 1)
    while n_test + n_val > max_holdout:
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
        else:
            break
    return n_test, n_val


def split_class(files, val_fraction, test_fraction, rng):
    """Assign a class's files to splits, guaranteeing at least one val and
    one test image when the class has enough images to spare."""
    files = sorted(files)
    rng.shuffle(files)
    n = len(files)
    n_test = round(n * test_fraction)
    n_val = round(n * val_fraction)
    if n >= 4:
        n_test = max(1, n_test)
        n_val = max(1, n_val)
    n_test, n_val = clamp_holdout_counts(n_test, n_val, n)
    assignments = []
    for i, name in enumerate(files):
        if i < n_test:
            assignments.append((name, "test"))
        elif i < n_test + n_val:
            assignments.append((name, "val"))
        else:
            assignments.append((name, "train"))
    return assignments


def list_image_files(class_dir):
    files = []
    for dirpath, dirnames, filenames in os.walk(class_dir):
        dirnames.sort()
        for name in sorted(filenames):
            if not name.lower().endswith(IMAGE_EXTENSIONS):
                continue
            path = os.path.join(dirpath, name)
            files.append(os.path.relpath(path, class_dir))
    return files


def to_manifest_path(path):
    return path.replace(os.sep, "/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../training-v1/screenshots")
    parser.add_argument("--out", default="manifest.csv")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.val_fraction < 0 or args.test_fraction < 0:
        parser.error("--val-fraction and --test-fraction must be non-negative")
    if args.val_fraction + args.test_fraction >= 1:
        parser.error("--val-fraction plus --test-fraction must be less than 1")

    rng = random.Random(args.seed)
    data_dir = os.path.abspath(args.data_dir)
    out_path = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_path)
    rows = []
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        files = list_image_files(class_dir)
        if not files:
            print(f"EMPTY class skipped: {class_name}")
            continue
        for name, split in split_class(files, args.val_fraction, args.test_fraction, rng):
            image_path = os.path.join(class_dir, name)
            rows.append(
                (
                    to_manifest_path(os.path.relpath(image_path, out_dir)),
                    class_name,
                    split,
                )
            )

    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "class", "split"])
        writer.writerows(rows)

    counts = Counter(split for _, _, split in rows)
    classes = len({c for _, c, _ in rows})
    print(
        f"Wrote {out_path}: {classes} classes, "
        + ", ".join(f"{counts[s]} {s}" for s in ("train", "val", "test"))
    )


if __name__ == "__main__":
    main()
