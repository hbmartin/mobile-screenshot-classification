"""Evaluate a trained run on the validation and test splits.

Usage:
    python evaluate.py models/<run-name>

Uses the config.yaml saved with the run, so the data split matches what the
model was trained against.
"""

import argparse
import os

import tensorflow as tf
import yaml

from data import load_datasets


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="Run directory created by train.py")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(os.path.join(args.model_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    tf.keras.utils.set_random_seed(cfg["seed"])
    _, val_ds, test_ds, _ = load_datasets(cfg)
    model = tf.keras.models.load_model(os.path.join(args.model_dir, "model.keras"))

    val_loss, val_accuracy = model.evaluate(val_ds)
    print(f"Validation loss: {val_loss:.4f}  accuracy: {val_accuracy:.4f}")
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"Test loss: {test_loss:.4f}  accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()
