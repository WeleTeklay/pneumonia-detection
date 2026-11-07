#!/usr/bin/env python3
"""Train the pneumonia classifier on the Kaggle chest X-ray dataset.

    python train.py                       # downloads the dataset from Kaggle (kagglehub)
    python train.py --data-dir /path/to/chest_xray   # use an existing copy instead

Outputs (next to this file):
    pneumonia_detector_model.keras        the model loaded by app.py
    artifacts/metrics.json + PNG plots    shown on the app's About page
"""
import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import class_weight

import core

KAGGLE_HANDLE = "paultimothymooney/chest-xray-pneumonia"
CLASS_NAMES = core.CLASS_NAMES
IMG_EXTS = (".jpeg", ".jpg", ".png")
SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
AUTOTUNE = tf.data.AUTOTUNE


# ----------------------------------------------------------------------------- data
def download_dataset():
    import kagglehub

    print(f"Downloading {KAGGLE_HANDLE} from Kaggle ...")
    return kagglehub.dataset_download(KAGGLE_HANDLE)


def find_dataset_root(start):
    """Shallowest folder that has train/ and test/ with NORMAL/ and PNEUMONIA/ inside."""
    for dirpath, _, _ in os.walk(start):
        if "__MACOSX" in dirpath:
            continue
        if all(os.path.isdir(os.path.join(dirpath, s, c)) for s in ("train", "test") for c in CLASS_NAMES):
            return dirpath
    raise FileNotFoundError(f"No train/test/NORMAL/PNEUMONIA layout found under {start}")


def list_split(split_dir):
    paths, labels = [], []
    if not os.path.isdir(split_dir):
        return paths, labels
    for label, cls in enumerate(CLASS_NAMES):
        cdir = os.path.join(split_dir, cls)
        for name in sorted(os.listdir(cdir)):
            if name.lower().endswith(IMG_EXTS) and not name.startswith("."):
                paths.append(os.path.join(cdir, name))
                labels.append(label)
    return paths, labels


def patient_group(path, label):
    """Best-effort patient id from Kaggle file names, so one patient never spans splits.
    PNEUMONIA: person123_bacteria_456.jpeg -> person123 ; NORMAL: IM-0115-0001.jpeg -> IM-0115"""
    name = os.path.basename(path)
    m = re.match(r"person(\d+)", name, re.I) if label == 1 else re.search(r"IM-(\d+)", name, re.I)
    return f"{label}-{m.group(1)}" if m else f"{label}-{name}"


def make_ds(paths, labels, batch_size, img_size, training, augment=None):
    ds = tf.data.Dataset.from_tensor_slices((np.array(paths), np.array(labels, dtype="float32")))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    def load(p, y):
        img = tf.io.decode_image(tf.io.read_file(p), channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        return tf.cast(img, tf.float32), y  # raw 0-255; the model normalises internally

    ds = ds.map(load, num_parallel_calls=AUTOTUNE).batch(batch_size)
    if training and augment is not None:
        ds = ds.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


# ----------------------------------------------------------------------------- model
def build_model(img_size, pretrained):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=(img_size, img_size, 3),
    )
    base.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(base.input, out, name="pneumonia_effnetb0"), base


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
    ]


# ----------------------------------------------------------------------------- plots
def plot_curves(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, key, title in zip(axes, ["loss", "accuracy", "auc"], ["Loss", "Accuracy", "AUC"]):
        ax.plot(history[key], label="train")
        ax.plot(history["val_" + key], label="val")
        if history.get("phase_boundary"):
            ax.axvline(history["phase_boundary"] - 0.5, color="gray", ls="--", label="fine-tune starts")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_roc(fpr, tpr, auc_value, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC (test)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def gradcam_samples(model, test_paths, test_labels, path, per_class=2):
    layer = core.find_gradcam_layer(model)
    picks = []
    for label in (0, 1):
        picks += [p for p, y in zip(test_paths, test_labels) if y == label][:per_class]
    fig, axes = plt.subplots(2, len(picks), figsize=(3.2 * len(picks), 6.5))
    for col, p in enumerate(picks):
        img = Image.open(p).convert("RGB")
        batch = core.preprocess(img)
        label, idx, p_pneu, conf = core.predict(model, batch)
        heat = core.make_gradcam_heatmap(batch, model, layer, target_pneumonia=bool(idx))
        axes[0, col].imshow(img, cmap="gray")
        axes[0, col].set_title(f"pred {label}\n({conf:.2f})", fontsize=9)
        axes[1, col].imshow(core.overlay_heatmap(img, heat))
        for r in (0, 1):
            axes[r, col].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", help="existing chest_xray folder (skips the Kaggle download)")
    ap.add_argument("--output", default=os.path.join(HERE, "pneumonia_detector_model.keras"))
    ap.add_argument("--img-size", type=int, default=core.IMG_SIZE)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs-head", type=int, default=8, help="phase 1: frozen backbone")
    ap.add_argument("--epochs-ft", type=int, default=10, help="phase 2: fine-tune top layers")
    ap.add_argument("--unfreeze", type=int, default=30, help="layers to unfreeze in phase 2")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--no-pretrained", action="store_true", help="random init (smoke tests only)")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(SEED)
    print("TensorFlow", tf.__version__, "| GPUs:", tf.config.list_physical_devices("GPU"))

    root = find_dataset_root(args.data_dir or download_dataset())
    print("Dataset root:", root)

    # ---- splits: original train + val are pooled, then re-split by patient (val in the
    # original Kaggle data has only 16 images, far too small to monitor training).
    tr_p, tr_y = list_split(os.path.join(root, "train"))
    va_p, va_y = list_split(os.path.join(root, "val"))
    te_p, te_y = list_split(os.path.join(root, "test"))
    pool_p, pool_y = np.array(tr_p + va_p), np.array(tr_y + va_y)
    groups = np.array([patient_group(p, y) for p, y in zip(pool_p, pool_y)])
    tr_idx, va_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=SEED).split(pool_p, pool_y, groups)
    )
    assert not (set(groups[tr_idx]) & set(groups[va_idx])), "patient leakage between train and val"
    train_p, train_y = pool_p[tr_idx], pool_y[tr_idx]
    val_p, val_y = pool_p[va_idx], pool_y[va_idx]
    te_p, te_y = np.array(te_p), np.array(te_y)

    def counts(y):
        return {CLASS_NAMES[i]: int((np.array(y) == i).sum()) for i in (0, 1)}

    print("train:", counts(train_y), "| val:", counts(val_y), "| test:", counts(te_y))

    cw = class_weight.compute_class_weight("balanced", classes=np.array([0, 1]), y=train_y)
    cw = {0: float(cw[0]), 1: float(cw[1])}
    print("class weights:", cw)

    augment = tf.keras.Sequential(
        [  # no horizontal flip on purpose: it would mirror chest anatomy
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomContrast(0.1),
        ]
    )
    train_ds = make_ds(train_p, train_y, args.batch_size, args.img_size, True, augment)
    val_ds = make_ds(val_p, val_y, args.batch_size, args.img_size, False)
    test_ds = make_ds(te_p, te_y, args.batch_size, args.img_size, False)

    model, base = build_model(args.img_size, pretrained=not args.no_pretrained)

    # ---- phase 1: train the new head on a frozen backbone
    print("\n=== Phase 1: frozen backbone ===")
    compile_model(model, 1e-3)
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head,
                   class_weight=cw, callbacks=callbacks(), verbose=2, shuffle=False).history

    # ---- phase 2: fine-tune the top layers (BatchNorm stays frozen)
    print("\n=== Phase 2: fine-tuning top layers ===")
    base.trainable = True
    for layer in base.layers[: -args.unfreeze]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, 1e-5)
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_ft,
                   class_weight=cw, callbacks=callbacks(), verbose=2, shuffle=False).history

    history = {k: h1.get(k, []) + h2.get(k, []) for k in set(h1) | set(h2)}
    history["phase_boundary"] = len(h1["loss"])

    # ---- evaluation on the untouched Kaggle test split
    print("\n=== Test evaluation ===")
    y_prob = model.predict(test_ds, verbose=0).ravel()
    y_pred = (y_prob > 0.5).astype(int)
    print(classification_report(te_y, y_pred, target_names=CLASS_NAMES, digits=3, zero_division=0))
    cm = confusion_matrix(te_y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    roc_auc = float(roc_auc_score(te_y, y_prob))
    fpr, tpr, _ = roc_curve(te_y, y_prob)

    os.makedirs(os.path.join(HERE, "artifacts"), exist_ok=True)
    art = lambda n: os.path.join(HERE, "artifacts", n)  # noqa: E731
    plot_curves(history, art("training_curves.png"))
    plot_confusion(cm, art("confusion_matrix.png"))
    plot_roc(fpr, tpr, roc_auc, art("roc_curve.png"))

    metrics = {
        "test": {
            "n": int(len(te_y)),
            "accuracy": float((tp + tn) / cm.sum()),
            "precision_pneumonia": float(tp / max(tp + fp, 1)),
            "recall_pneumonia": float(tp / max(tp + fn, 1)),
            "specificity": float(tn / max(tn + fp, 1)),
            "roc_auc": roc_auc,
            "confusion_matrix": cm.tolist(),
        },
        "data": {"train": counts(train_y), "val": counts(val_y), "test": counts(te_y)},
        "config": {**{k: v for k, v in vars(args).items() if k != "output"}, "class_weights": cw},
    }
    with open(art("metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    model.save(args.output)
    print("Saved model ->", args.output)
    try:
        gradcam_samples(model, te_p.tolist(), te_y.tolist(), art("gradcam_samples.png"))
    except Exception as e:  # noqa: BLE001  (plots are a bonus, never fail the run)
        print("Grad-CAM sample plot skipped:", e)
    print(json.dumps(metrics["test"], indent=2))


if __name__ == "__main__":
    main()
