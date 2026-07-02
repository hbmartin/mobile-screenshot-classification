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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="Run directory created by train.py")
    parser.add_argument("images", nargs="+", help="Screenshot image files")
    parser.add_argument("--top", type=int, default=3, help="How many guesses to print")
    return parser.parse_args()


def load_image(path, image_size):
    """Load an image, center-crop it to the target aspect ratio, and resize.

    Mirrors crop_to_aspect_ratio=True used by the training pipeline.
    """
    arr = tf.keras.utils.img_to_array(tf.keras.utils.load_img(path))
    height, width = arr.shape[0], arr.shape[1]
    target_height, target_width = image_size
    crop_width = min(width, round(height * target_width / target_height))
    crop_height = min(height, round(width * target_height / target_width))
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    arr = arr[top : top + crop_height, left : left + crop_width]
    return tf.image.resize(arr, image_size)


def main():
    args = parse_args()
    with open(os.path.join(args.model_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(args.model_dir, "class_names.json")) as f:
        class_names = json.load(f)

    model = tf.keras.models.load_model(os.path.join(args.model_dir, "model.keras"))
    image_size = (cfg["data"]["image_height"], cfg["data"]["image_width"])

    batch = tf.stack([load_image(path, image_size) for path in args.images])
    probabilities = tf.nn.softmax(model.predict(batch), axis=-1).numpy()

    for path, probs in zip(args.images, probabilities):
        print(path)
        for index in np.argsort(probs)[::-1][: args.top]:
            print(f"  {class_names[index]}: {probs[index]:.1%}")


if __name__ == "__main__":
    main()
