"""Dataset loading shared by train.py and evaluate.py.

Two modes, selected by config:
- data.manifest set: load the fixed train/val/test split written by split.py
  (preferred — the test set stays stable across runs and machines).
- data.manifest null: derive a random split from the seed, matching the
  original notebook behaviour.
"""

import csv
import os

import tensorflow as tf

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def load_datasets(cfg):
    """Build (train, validation, test) datasets and the class-name list."""
    if cfg["data"].get("manifest"):
        return _load_from_manifest(cfg)
    return _load_from_directory(cfg)


def _load_from_directory(cfg):
    data_cfg = cfg["data"]
    common = dict(
        directory=data_cfg["dir"],
        image_size=(data_cfg["image_height"], data_cfg["image_width"]),
        batch_size=data_cfg["batch_size"],
        crop_to_aspect_ratio=True,
        validation_split=data_cfg["validation_split"],
        seed=cfg["seed"],
        shuffle=True,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(subset="training", **common)
    holdout_ds = tf.keras.utils.image_dataset_from_directory(subset="validation", **common)
    class_names = train_ds.class_names

    holdout_batches = tf.data.experimental.cardinality(holdout_ds).numpy()
    test_batches = int(holdout_batches * data_cfg["test_fraction"])
    test_ds = holdout_ds.take(test_batches)
    val_ds = holdout_ds.skip(test_batches)
    return _prefetch(train_ds, val_ds, test_ds) + (class_names,)


def read_manifest(manifest_path):
    """Return (rows, class_names); rows are (path, class, split) tuples with
    relative paths resolved against the manifest's directory."""
    base = os.path.dirname(os.path.abspath(manifest_path))
    rows = []
    with open(manifest_path, newline="") as f:
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
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if shuffle:
            ds = ds.shuffle(len(paths), seed=cfg["seed"], reshuffle_each_iteration=True)
        ds = ds.map(
            lambda path, label: (_load_image(path, image_size), label),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return ds.batch(data_cfg["batch_size"])

    train_ds = build("train", shuffle=True)
    val_ds = build("val", shuffle=False)
    test_ds = build("test", shuffle=False)
    return _prefetch(train_ds, val_ds, test_ds) + (class_names,)


def _load_image(path, image_size):
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
        files = os.listdir(os.path.join(data_dir, name))
        counts.append(max(1, sum(f.lower().endswith(IMAGE_EXTENSIONS) for f in files)))
    return counts
