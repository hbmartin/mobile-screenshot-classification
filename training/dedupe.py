"""Find near-duplicate screenshots with a perceptual difference hash.

App Store screenshot sets often contain near-identical frames; if such
duplicates land on both sides of the train/validation split they inflate
validation accuracy. Run this before split.py.

Usage:
    python dedupe.py [--data-dir ../training-v1/screenshots] [--delete]

Without --delete this is a dry run that only reports duplicate groups.
With --delete, every duplicate group keeps its first image (sorted order)
and the rest are removed. Duplicates spanning two different classes are
never deleted automatically — the same image labeled as two apps is label
noise that needs a human decision — they are only reported.
"""

import argparse
import os
from itertools import combinations

from PIL import Image

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
HASH_SIZE = 8  # 64-bit hash


def dhash(path):
    """Difference hash: 1 bit per horizontal brightness gradient."""
    with Image.open(path) as image:
        image = image.convert("L").resize((HASH_SIZE + 1, HASH_SIZE), Image.LANCZOS)
        pixels = list(image.getdata())
    bits = 0
    for row in range(HASH_SIZE):
        for col in range(HASH_SIZE):
            left = pixels[row * (HASH_SIZE + 1) + col]
            right = pixels[row * (HASH_SIZE + 1) + col + 1]
            bits = (bits << 1) | (left > right)
    return bits


def hamming(a, b):
    return (a ^ b).bit_count()


def find_groups(hashes, threshold):
    """Union-find over images; two images join a group when their hashes are
    within `threshold` bits."""
    paths = list(hashes)
    parent = {p: p for p in paths}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    for a, b in combinations(paths, 2):
        if hamming(hashes[a], hashes[b]) <= threshold:
            parent[find(a)] = find(b)

    groups = {}
    for p in paths:
        groups.setdefault(find(p), []).append(p)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="../training-v1/screenshots")
    parser.add_argument(
        "--threshold",
        type=int,
        default=2,
        help="Max Hamming distance (in bits, of 64) to call two images duplicates",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete same-class duplicates (keeps the first of each group)",
    )
    args = parser.parse_args()

    hashes = {}
    for dirpath, _, filenames in os.walk(args.data_dir):
        for name in filenames:
            if not name.lower().endswith(IMAGE_EXTENSIONS):
                continue
            path = os.path.join(dirpath, name)
            try:
                hashes[path] = dhash(path)
            except OSError as err:
                print(f"UNREADABLE {path}: {err}")
    print(f"Hashed {len(hashes)} images")

    removed = 0
    cross_class = 0
    for group in find_groups(hashes, args.threshold):
        classes = {os.path.basename(os.path.dirname(p)) for p in group}
        if len(classes) > 1:
            cross_class += 1
            print("CROSS-CLASS duplicates (possible label noise, not deleted):")
            for p in group:
                print(f"  {p}")
            continue
        print(f"Duplicates (keeping {group[0]}):")
        for p in group[1:]:
            print(f"  {p}")
            if args.delete:
                os.remove(p)
                removed += 1

    if args.delete:
        print(f"Deleted {removed} duplicate images")
    else:
        print("Dry run; pass --delete to remove same-class duplicates")
    if cross_class:
        print(f"{cross_class} cross-class duplicate groups need manual review")


if __name__ == "__main__":
    main()
