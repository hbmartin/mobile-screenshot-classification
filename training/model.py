"""Model construction for the screenshot classifier."""

import tensorflow as tf

# backbone name -> (constructor, matching preprocess function)
BACKBONES = {
    "mobilenet_v2": (
        tf.keras.applications.MobileNetV2,
        tf.keras.applications.mobilenet_v2.preprocess_input,
    ),
}


def build_model(num_classes, cfg):
    """Return (model, base_model) with preprocessing baked into the graph.

    The base model starts fully frozen (feature extraction); train.py
    unfreezes its top layers for the fine-tuning stage.
    """
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    input_shape = (data_cfg["image_height"], data_cfg["image_width"], 3)

    if model_cfg["backbone"] not in BACKBONES:
        raise ValueError(
            f"Unknown backbone {model_cfg['backbone']!r}; "
            f"choose one of {sorted(BACKBONES)}"
        )
    constructor, preprocess = BACKBONES[model_cfg["backbone"]]
    base_model = constructor(input_shape=input_shape, include_top=False, weights="imagenet")
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = preprocess(inputs)
    # training=False keeps BatchNormalization in inference mode so its
    # statistics survive later unfreezing.
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(model_cfg["dropout"])(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs, outputs), base_model


def unfreeze_top_layers(base_model, fine_tune_at):
    """Unfreeze the backbone above `fine_tune_at` for fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
