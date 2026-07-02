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
    assignments = []
    for i, name in enumerate(files):
        if i < n_test:
            assignments.append((name, "test"))
        elif i < n_test + n_val:
            assignments.append((name, "val"))
        else:
            assignments.append((name, "train"))
    return assignments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../training-v1/screenshots")
    parser.add_argument("--out", default="manifest.csv")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for class_name in sorted(os.listdir(args.data_dir)):
        class_dir = os.path.join(args.data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        files = [f for f in os.listdir(class_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]
        if not files:
            print(f"EMPTY class skipped: {class_name}")
            continue
        for name, split in split_class(files, args.val_fraction, args.test_fraction, rng):
            rows.append((os.path.join(class_dir, name), class_name, split))

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "class", "split"])
        writer.writerows(rows)

    counts = Counter(split for _, _, split in rows)
    classes = len({c for _, c, _ in rows})
    print(
        f"Wrote {args.out}: {classes} classes, "
        + ", ".join(f"{counts[s]} {s}" for s in ("train", "val", "test"))
    )


if __name__ == "__main__":
    main()
