"""Dataset loading shared by train.py and evaluate.py.

Two modes, selected by config:
- data.manifest set: load the fixed train/val/test split written by split.py
  (preferred — the test set stays stable across runs and machines).
- data.manifest null: derive a random split from the seed, matching the
  original notebook behaviour.
"""

import csv
import os
import random

import tensorflow as tf

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
DEFAULT_SHUFFLE_BUFFER_SIZE = 10000


def load_datasets(cfg):
    """Build (train, validation, test) datasets and the class-name list."""
    if cfg["data"].get("manifest"):
        return _load_from_manifest(cfg)
    return _load_from_directory(cfg)


def _load_from_directory(cfg):
    data_cfg = cfg["data"]
    image_size = (data_cfg["image_height"], data_cfg["image_width"])
    rows, class_names = _list_directory_rows(data_cfg["dir"])
    total_count = len(rows)
    if total_count <= 0:
        raise ValueError("Directory produced no images")

    val_split = data_cfg["validation_split"]
    test_fraction = data_cfg["test_fraction"]
    if val_split <= 0 or val_split >= 1:
        raise ValueError("data.validation_split must be greater than 0 and less than 1")
    if test_fraction <= 0 or test_fraction >= 1:
        raise ValueError("data.test_fraction must be greater than 0 and less than 1")

    rows = list(rows)
    random.Random(cfg["seed"]).shuffle(rows)

    holdout_count = round(total_count * val_split)
    train_count = total_count - holdout_count
    if train_count <= 0:
        raise ValueError("Directory split produced no training images")
    if holdout_count < 2:
        raise ValueError(
            "Directory split needs at least two holdout images to create "
            "validation and test datasets; set data.manifest for small datasets"
        )

    test_count = max(1, round(holdout_count * test_fraction))
    test_count = min(test_count, holdout_count - 1)

    train_rows = rows[:train_count]
    holdout_rows = rows[train_count:]
    test_rows = holdout_rows[:test_count]
    val_rows = holdout_rows[test_count:]

    train_ds = _rows_to_dataset(
        train_rows, image_size, data_cfg, cfg["seed"], shuffle=True
    )
    val_ds = _rows_to_dataset(val_rows, image_size, data_cfg, cfg["seed"], shuffle=False)
    test_ds = _rows_to_dataset(
        test_rows, image_size, data_cfg, cfg["seed"], shuffle=False
    )
    return _prefetch(train_ds, val_ds, test_ds) + (class_names,)


def _list_directory_rows(data_dir):
    rows = []
    class_names = []
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        class_rows = []
        for dirpath, dirnames, filenames in os.walk(class_dir):
            dirnames.sort()
            for name in sorted(filenames):
                if not name.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                class_rows.append(os.path.join(dirpath, name))
        if class_rows:
            class_index = len(class_names)
            class_names.append(class_name)
            rows.extend((path, class_index) for path in class_rows)
    return rows, class_names


def _rows_to_dataset(rows, image_size, data_cfg, seed, shuffle):
    paths = [path for path, _ in rows]
    labels = [label for _, label in rows]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle and len(rows) > 1:
        ds = ds.shuffle(
            _shuffle_buffer_size(data_cfg, len(rows)),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    ds = ds.map(
        lambda path, label: (load_image(path, image_size), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(data_cfg["batch_size"])


def _shuffle_buffer_size(data_cfg, count):
    configured = data_cfg.get("shuffle_buffer_size", DEFAULT_SHUFFLE_BUFFER_SIZE)
    if isinstance(configured, bool):
        raise ValueError("data.shuffle_buffer_size must be a positive integer")
    if isinstance(configured, int):
        buffer_size = configured
    elif isinstance(configured, float):
        if not configured.is_integer():
            raise ValueError("data.shuffle_buffer_size must be a positive integer")
        buffer_size = int(configured)
    elif isinstance(configured, str) and configured.strip().isdigit():
        buffer_size = int(configured)
    else:
        raise ValueError("data.shuffle_buffer_size must be a positive integer")
    if buffer_size <= 0:
        raise ValueError("data.shuffle_buffer_size must be a positive integer")
    return min(count, buffer_size)


def read_manifest(manifest_path):
    """Return (rows, class_names); rows are (path, class, split) tuples with
    relative paths resolved against the manifest's directory."""
    base = os.path.dirname(os.path.abspath(manifest_path))
    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = row["path"]
            if not os.path.isabs(path):
                path = os.path.normpath(os.path.join(base, path))
            rows.append((path, row["class"], row["split"]))
    class_names = sorted({class_name for _, class_name, _ in rows})
    return rows, class_names


def _load_from_manifest(cfg):
    data_cfg = cfg["data"]
    image_size = (data_cfg["image_height"], data_cfg["image_width"])
    rows, class_names = read_manifest(data_cfg["manifest"])
    class_index = {name: i for i, name in enumerate(class_names)}

    def build(split, shuffle):
        paths = [p for p, _, s in rows if s == split]
        labels = [class_index[c] for _, c, s in rows if s == split]
        if not paths:
            raise ValueError(f"Manifest split '{split}' produced no images")
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle and len(paths) > 1:
            ds = ds.shuffle(
                _shuffle_buffer_size(data_cfg, len(paths)),
                seed=cfg["seed"],
                reshuffle_each_iteration=True,
            )
        ds = ds.map(
            lambda path, label: (load_image(path, image_size), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return ds.batch(data_cfg["batch_size"])

    train_ds = build("train", shuffle=True)
    val_ds = build("val", shuffle=False)
    test_ds = build("test", shuffle=False)
    return _prefetch(train_ds, val_ds, test_ds) + (class_names,)


def load_image(path, image_size):
    """Decode, center-crop to the target aspect ratio, and resize — the same
    treatment crop_to_aspect_ratio=True applies in directory mode."""
    image = tf.image.decode_image(
        tf.io.read_file(path), channels=3, expand_animations=False
    )
    shape = tf.shape(image)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    target_height, target_width = image_size
    crop_width = tf.minimum(width, tf.round(height * target_width / target_height))
    crop_height = tf.minimum(height, tf.round(width * target_height / target_width))
    top = tf.cast((height - crop_height) / 2, tf.int32)
    left = tf.cast((width - crop_width) / 2, tf.int32)
    image = image[
        top : top + tf.cast(crop_height, tf.int32),
        left : left + tf.cast(crop_width, tf.int32),
    ]
    image = tf.image.resize(image, image_size)
    image.set_shape((image_size[0], image_size[1], 3))
    return image


def _prefetch(*datasets):
    return tuple(ds.prefetch(tf.data.AUTOTUNE) for ds in datasets)


def compute_class_weights(cfg, class_names):
    """Inverse-frequency weights so apps with few screenshots aren't drowned
    out by apps with many."""
    counts = _class_counts(cfg, class_names)
    total = sum(counts)
    return {i: total / (len(class_names) * c) for i, c in enumerate(counts)}


def _class_counts(cfg, class_names):
    if cfg["data"].get("manifest"):
        rows, _ = read_manifest(cfg["data"]["manifest"])
        by_class = {name: 0 for name in class_names}
        for _, class_name, split in rows:
            if split == "train":
                by_class[class_name] += 1
        return [max(1, by_class[name]) for name in class_names]
    # Directory mode counts all files per class (not just the training split);
    # close enough, and it avoids iterating the dataset a second time.
    data_dir = cfg["data"]["dir"]
    counts = []
    for name in class_names:
        class_dir = os.path.join(data_dir, name)
        count = 0
        for _, _, files in os.walk(class_dir):
            count += sum(f.lower().endswith(IMAGE_EXTENSIONS) for f in files)
        counts.append(max(1, count))
    return counts
