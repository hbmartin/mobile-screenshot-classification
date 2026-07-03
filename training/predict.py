"""Classify screenshots with a trained run.

Usage:
    python predict.py models/<run-name> screenshot1.png [screenshot2.png ...]
"""

import argparse
import json
import os

import numpy as np
import tensorflow as tf
import yaml

from data import load_image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="Run directory created by train.py")
    parser.add_argument("images", nargs="+", help="Screenshot image files")
    parser.add_argument("--top", type=int, default=3, help="How many guesses to print")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(os.path.join(args.model_dir, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(args.model_dir, "class_names.json"), encoding="utf-8") as f:
        class_names = json.load(f)

    model = tf.keras.models.load_model(
        os.path.join(args.model_dir, "model.keras"), compile=False
    )
    image_size = (cfg["data"]["image_height"], cfg["data"]["image_width"])

    batch = tf.stack([load_image(path, image_size) for path in args.images])
    probabilities = tf.nn.softmax(model.predict(batch), axis=-1).numpy()

    for path, probs in zip(args.images, probabilities):
        print(path)
        for index in np.argsort(probs)[::-1][: args.top]:
            print(f"  {class_names[index]}: {probs[index]:.1%}")


if __name__ == "__main__":
    main()
