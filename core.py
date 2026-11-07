"""Shared inference + Grad-CAM helpers (used by app.py and train.py).

The model takes raw 0-255 RGB pixels: Keras' EfficientNet contains its own
rescaling/normalisation layers, so no extra preprocess_input call is needed
(it is a pass-through for EfficientNet anyway).
"""
import os

import matplotlib
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_SIZE = 224
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]  # index 0 / 1  (sigmoid output = P(PNEUMONIA))
MODEL_FILENAMES = ["pneumonia_detector_model.keras", "pneumonia_detector_model.h5"]
HERE = os.path.dirname(os.path.abspath(__file__))


def resolve_model_path():
    """MODEL_PATH env var wins; otherwise look next to this file. None if missing."""
    env = os.environ.get("MODEL_PATH")
    if env:
        return env if os.path.exists(env) else None
    for name in MODEL_FILENAMES:
        path = os.path.join(HERE, name)
        if os.path.exists(path):
            return path
    return None


def load_model(path):
    return tf.keras.models.load_model(path, compile=False)


def preprocess(pil_image, img_size=IMG_SIZE):
    """PIL image -> float32 batch of shape (1, H, W, 3), values 0-255."""
    arr = np.asarray(pil_image.convert("RGB"), dtype=np.float32)
    # Same op (bilinear tf.image.resize) that train.py's data pipeline uses, so what the app
    # feeds the model matches what the model was trained and evaluated on.
    arr = tf.image.resize(arr, (img_size, img_size)).numpy()
    return np.expand_dims(arr, axis=0)


def predict(model, batch):
    """Returns (label, class_index, p_pneumonia, confidence_in_label)."""
    p = float(model.predict(batch, verbose=0)[0][0])
    idx = int(p > 0.5)
    confidence = p if idx == 1 else 1.0 - p
    return CLASS_NAMES[idx], idx, p, confidence


def find_gradcam_layer(model):
    """Prefer EfficientNet's final feature map; else the last (depthwise) conv."""
    if any(layer.name == "top_activation" for layer in model.layers):
        return "top_activation"
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    return None


def make_gradcam_heatmap(batch, model, layer_name, target_pneumonia=True):
    """Grad-CAM for a single-sigmoid-output model.

    The model has ONE output (P(PNEUMONIA)), so the score is that output when the
    predicted class is PNEUMONIA and (1 - output) when it is NORMAL. That way the
    heatmap always explains the class that was actually predicted.
    """
    grad_model = tf.keras.models.Model(
        model.input, [model.get_layer(layer_name).output, model.output]
    )
    x = tf.convert_to_tensor(batch)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        score = preds[:, 0] if target_pneumonia else 1.0 - preds[:, 0]
    grads = tape.gradient(score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)  # no NaN on all-zero maps
    return heatmap.numpy()


def _jet_lut():
    try:
        cmap = matplotlib.colormaps["jet"]          # matplotlib >= 3.5
    except AttributeError:                           # pragma: no cover
        cmap = matplotlib.cm.get_cmap("jet")
    return (cmap(np.arange(256))[:, :3] * 255).astype(np.uint8)


def overlay_heatmap(original, heatmap, alpha=0.4):
    """Alpha-blend a jet heatmap over a PIL image (clipped -> no uint8 wrap-around)."""
    idx = np.uint8(255 * np.clip(heatmap, 0, 1))
    jet = Image.fromarray(_jet_lut()[idx]).resize(original.size, Image.BILINEAR)
    base = np.asarray(original.convert("RGB"), dtype=np.float32)
    blended = (1.0 - alpha) * base + alpha * np.asarray(jet, dtype=np.float32)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
