"""Train the screenshot classifier described by config.yaml.

Usage:
    python train.py [--config config.yaml] [--run-name my-experiment]

Each run writes to <output.model_dir>/<run-name>/:
    model.keras       trained model (preprocessing included in the graph)
    class_names.json  index -> app name mapping
    config.yaml       copy of the config the run used
and TensorBoard logs to <output.log_dir>/<run-name>/.
"""

import argparse
import json
import os
import shutil
from datetime import datetime

import tensorflow as tf
import yaml

from data import load_datasets
from model import build_model, unfreeze_top_layers


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Name for this run's model/log directories (default: timestamp)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Seeds Python, NumPy, and TensorFlow in one call.
    tf.keras.utils.set_random_seed(cfg["seed"])

    run_name = args.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(cfg["output"]["model_dir"], run_name)
    log_dir = os.path.join(cfg["output"]["log_dir"], run_name)
    os.makedirs(model_dir, exist_ok=True)

    train_ds, val_ds, test_ds, class_names = load_datasets(cfg)
    model, base_model = build_model(len(class_names), cfg)

    train_cfg = cfg["train"]
    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=log_dir)]

    # Stage 1: feature extraction with the backbone frozen.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=train_cfg["learning_rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=train_cfg["epochs"],
        callbacks=callbacks,
    )

    # Stage 2: unfreeze the top of the backbone and continue at a lower LR.
    if train_cfg["fine_tune"]:
        unfreeze_top_layers(base_model, train_cfg["fine_tune_at"])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=train_cfg["fine_tune_learning_rate"]
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=train_cfg["epochs"] + train_cfg["fine_tune_epochs"],
            initial_epoch=train_cfg["epochs"],
            callbacks=callbacks,
        )

    model.save(os.path.join(model_dir, "model.keras"))
    with open(os.path.join(model_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    shutil.copyfile(args.config, os.path.join(model_dir, "config.yaml"))

    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_accuracy:.4f}")
    print(f"Saved run to {model_dir}")


if __name__ == "__main__":
    main()
