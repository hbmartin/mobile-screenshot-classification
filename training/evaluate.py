"""Evaluate a trained run on the validation and test splits.

Usage:
    python evaluate.py models/<run-name>

Uses the config.yaml saved with the run, so the data split matches what the
model was trained against. Writes the test-set confusion matrix to
<run-dir>/confusion_matrix.csv and prints the most confused class pairs and
the weakest classes.
"""

import argparse
import csv
import json
import os

import numpy as np
import tensorflow as tf
import yaml

from data import load_datasets
from model import build_metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="Run directory created by train.py")
    parser.add_argument(
        "--top-confusions", type=int, default=10, help="How many confused pairs to print"
    )
    return parser.parse_args()


def collect_predictions(model, dataset):
    y_true, y_pred = [], []
    for images, labels in dataset:
        logits = model.predict_on_batch(images)
        y_true.append(labels.numpy())
        y_pred.append(np.argmax(logits, axis=-1))
    return np.concatenate(y_true), np.concatenate(y_pred)


def report_confusions(y_true, y_pred, class_names, out_path, top_confusions):
    num_classes = len(class_names)
    matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=num_classes).numpy()

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\predicted"] + class_names)
        for name, row in zip(class_names, matrix):
            writer.writerow([name] + row.tolist())
    print(f"Confusion matrix written to {out_path}")

    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    pairs = np.dstack(np.unravel_index(np.argsort(off_diagonal, axis=None)[::-1], matrix.shape))[0]
    printed = 0
    print("\nMost confused pairs (true -> predicted):")
    for i, j in pairs:
        if off_diagonal[i, j] == 0 or printed >= top_confusions:
            break
        print(f"  {class_names[i]} -> {class_names[j]}: {off_diagonal[i, j]}")
        printed += 1
    if printed == 0:
        print("  (none)")

    support = matrix.sum(axis=1)
    seen = support > 0
    per_class_accuracy = np.divide(
        np.diag(matrix), support, out=np.zeros(num_classes), where=seen
    )
    print("\nWeakest classes present in the test set:")
    for index in np.argsort(per_class_accuracy + ~seen)[:10]:
        if not seen[index]:
            continue
        print(
            f"  {class_names[index]}: "
            f"{per_class_accuracy[index]:.0%} of {support[index]} images"
        )


def main():
    args = parse_args()
    with open(os.path.join(args.model_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(args.model_dir, "class_names.json")) as f:
        class_names = json.load(f)

    tf.keras.utils.set_random_seed(cfg["seed"])
    _, val_ds, test_ds, _ = load_datasets(cfg)

    # compile=False because the training loss may be a custom (label-smoothed)
    # function; recompile with the plain sparse loss for scoring.
    model = tf.keras.models.load_model(
        os.path.join(args.model_dir, "model.keras"), compile=False
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=build_metrics(),
    )

    for name, dataset in (("Validation", val_ds), ("Test", test_ds)):
        results = model.evaluate(dataset, return_dict=True, verbose=0)
        print(
            f"{name}: loss {results['loss']:.4f}  "
            f"accuracy {results['accuracy']:.4f}  "
            f"top-5 accuracy {results['top5_accuracy']:.4f}"
        )

    y_true, y_pred = collect_predictions(model, test_ds)
    report_confusions(
        y_true,
        y_pred,
        class_names,
        os.path.join(args.model_dir, "confusion_matrix.csv"),
        args.top_confusions,
    )


if __name__ == "__main__":
    main()
