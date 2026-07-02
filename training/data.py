"""Dataset loading shared by train.py and evaluate.py."""

import os

import tensorflow as tf

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def load_datasets(cfg):
    """Build (train, validation, test) datasets and the class-name list.

    The split is deterministic for a given seed, so train.py and evaluate.py
    see the same partition.
    """
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

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names


def compute_class_weights(cfg, class_names):
    """Inverse-frequency weights so apps with few screenshots aren't drowned
    out by apps with many.

    Counts all files per class directory (not just the training split) —
    close enough given the split is stratified only by chance, and it avoids
    iterating the dataset a second time.
    """
    data_dir = cfg["data"]["dir"]
    counts = []
    for name in class_names:
        files = os.listdir(os.path.join(data_dir, name))
        counts.append(max(1, sum(f.lower().endswith(IMAGE_EXTENSIONS) for f in files)))
    total = sum(counts)
    return {i: total / (len(class_names) * c) for i, c in enumerate(counts)}
