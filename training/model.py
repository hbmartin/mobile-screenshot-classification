"""Model construction for the screenshot classifier."""

import tensorflow as tf

# backbone name -> (constructor, matching preprocess function)
BACKBONES = {
    "mobilenet_v2": (
        tf.keras.applications.MobileNetV2,
        tf.keras.applications.mobilenet_v2.preprocess_input,
    ),
    "efficientnet_v2_s": (
        tf.keras.applications.EfficientNetV2S,
        tf.keras.applications.efficientnet_v2.preprocess_input,
    ),
}


def build_augmentation(cfg):
    """Augmentation layers suited to screenshots.

    Deliberately no flips or rotations: UI text and layout are not
    flip-invariant, and a mirrored screenshot is not a plausible input.
    These layers are only active during training.
    """
    aug_cfg = cfg.get("augmentation") or {}
    layers = []
    if aug_cfg.get("translation"):
        layers.append(
            tf.keras.layers.RandomTranslation(
                aug_cfg["translation"], aug_cfg["translation"], fill_mode="constant"
            )
        )
    if aug_cfg.get("zoom"):
        layers.append(tf.keras.layers.RandomZoom(aug_cfg["zoom"], fill_mode="constant"))
    if aug_cfg.get("contrast"):
        layers.append(tf.keras.layers.RandomContrast(aug_cfg["contrast"]))
    if aug_cfg.get("brightness"):
        layers.append(tf.keras.layers.RandomBrightness(aug_cfg["brightness"]))
    return layers


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
    x = inputs
    for layer in build_augmentation(cfg):
        x = layer(x)
    x = preprocess(x)
    # training=False keeps BatchNormalization in inference mode so its
    # statistics survive later unfreezing.
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(model_cfg["dropout"])(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs, outputs), base_model


def build_loss(num_classes, label_smoothing):
    """Cross-entropy over integer labels, with optional label smoothing.

    SparseCategoricalCrossentropy has no label_smoothing argument, so when
    smoothing is requested the integer labels are one-hot encoded inside the
    loss.
    """
    if not label_smoothing:
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    cce = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, label_smoothing=label_smoothing
    )

    def loss(y_true, y_pred):
        y_true = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), num_classes)
        return cce(y_true, y_pred)

    return loss


def build_metrics():
    return [
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
    ]


def unfreeze_top_layers(base_model, fine_tune_at):
    """Unfreeze the backbone above `fine_tune_at` for fine-tuning."""
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
