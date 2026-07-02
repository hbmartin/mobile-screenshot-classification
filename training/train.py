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

from data import compute_class_weights, load_datasets
from model import build_loss, build_metrics, build_model, unfreeze_top_layers


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
    loss = build_loss(len(class_names), train_cfg.get("label_smoothing", 0.0))
    class_weight = (
        compute_class_weights(cfg, class_names)
        if train_cfg.get("class_weights")
        else None
    )

    def make_callbacks():
        # Fresh instances per stage so EarlyStopping state doesn't leak from
        # feature extraction into fine-tuning.
        return [
            tf.keras.callbacks.TensorBoard(log_dir=log_dir),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=train_cfg["early_stopping_patience"],
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=train_cfg["reduce_lr_patience"],
            ),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(model_dir, "checkpoint.keras"),
                monitor="val_accuracy",
                save_best_only=True,
            ),
        ]

    # Stage 1: feature extraction with the backbone frozen.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=train_cfg["learning_rate"]),
        loss=loss,
        metrics=build_metrics(),
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=train_cfg["epochs"],
        class_weight=class_weight,
        callbacks=make_callbacks(),
    )

    # Stage 2: unfreeze the top of the backbone and continue at a lower LR.
    if train_cfg["fine_tune"]:
        unfreeze_top_layers(base_model, train_cfg["fine_tune_at"])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=train_cfg["fine_tune_learning_rate"]
            ),
            loss=loss,
            metrics=build_metrics(),
        )
        last_epoch = history.epoch[-1] + 1
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=last_epoch + train_cfg["fine_tune_epochs"],
            initial_epoch=last_epoch,
            class_weight=class_weight,
            callbacks=make_callbacks(),
        )

    model.save(os.path.join(model_dir, "model.keras"))
    with open(os.path.join(model_dir, "class_names.json"), "w") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    shutil.copyfile(args.config, os.path.join(model_dir, "config.yaml"))

    results = model.evaluate(test_ds, return_dict=True)
    print(
        f"Test loss: {results['loss']:.4f}  "
        f"accuracy: {results['accuracy']:.4f}  "
        f"top-5 accuracy: {results['top5_accuracy']:.4f}"
    )
    print(f"Saved run to {model_dir}")


if __name__ == "__main__":
    main()
