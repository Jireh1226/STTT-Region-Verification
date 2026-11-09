"""
Retrain the plausibility classifier with a genuinely disjoint real-image
holdout.

The original train_plausibility_classifier.py applied its 90/10 split to the
combined shuffled real+fake set, so ~90% of the 10,000 real images were
training positives and no identified real-image holdout exists. This variant
splits the REAL images first (90% train / 10% audit-holdout, seed 42), builds
all four fake families from the training reals only (so the holdout cannot
leak through corner bounds, blends, or high-frequency bases), trains the same
128 -> 64 -> 16 -> 1 architecture, and saves the holdout indices so the
calibration audit can score genuinely unseen real images.

Outputs:
  plausibility_classifier_holdout.npz       (weights + holdout indices)
  plausibility_classifier_holdout_metrics.json
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
import h5py

SEED = 42
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

from tensorflow import keras
from tensorflow.keras import layers

DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
WEIGHTS_PATH = "plausibility_classifier_holdout.npz"
METRICS_PATH = "plausibility_classifier_holdout_metrics.json"

NUM_PIXELS = 128
HIDDEN_1 = 64
HIDDEN_2 = 16
EPOCHS = 100
BATCH_SIZE = 128
LR = 1e-3
HOLDOUT_FRAC = 0.10


def make_fakes(real_images, n_per_type, rng):
    """Same four fake families as train_plausibility_classifier.py, built
    from the supplied (training-only) real images."""
    fakes = []
    pixel_min = real_images.min(axis=0)
    pixel_max = real_images.max(axis=0)

    corners = np.zeros((n_per_type, NUM_PIXELS), dtype=np.float32)
    for i in range(n_per_type):
        pick_high = rng.random(NUM_PIXELS) > 0.5
        corners[i] = np.where(pick_high, pixel_max, pixel_min)
    fakes.append(corners)

    blends = np.zeros((n_per_type, NUM_PIXELS), dtype=np.float32)
    for i in range(n_per_type):
        img_indices = rng.integers(0, len(real_images), NUM_PIXELS)
        blends[i] = real_images[img_indices, np.arange(NUM_PIXELS)]
    fakes.append(blends)

    noise_min = float(pixel_min.min())
    noise_max = float(pixel_max.max())
    uniform = rng.uniform(noise_min, noise_max,
                          (n_per_type, NUM_PIXELS)).astype(np.float32)
    fakes.append(uniform)

    hf = np.zeros((n_per_type, NUM_PIXELS), dtype=np.float32)
    for i in range(n_per_type):
        base = real_images[rng.integers(0, len(real_images))].copy()
        if rng.random() < 0.5:
            mask = rng.random(NUM_PIXELS) < 0.3
            flip = rng.random(NUM_PIXELS) > 0.5
            base[mask] = np.where(flip[mask], pixel_max[mask], pixel_min[mask])
        else:
            checkerboard = ((np.arange(NUM_PIXELS) // 16
                             + np.arange(NUM_PIXELS) % 16) % 2) * 0.3
            base = np.clip(base + checkerboard - 0.15, 0, 1)
        hf[i] = base
    fakes.append(hf)

    return np.concatenate(fakes, axis=0)


def build_classifier():
    inp = keras.Input(shape=(NUM_PIXELS,), name="pixels_in")
    h = layers.Dense(HIDDEN_1, activation="relu", name="c1")(inp)
    h = layers.Dense(HIDDEN_2, activation="relu", name="c2")(h)
    logit = layers.Dense(1, activation="linear", name="score")(h)
    return keras.Model(inp, logit, name="plausibility_classifier_holdout")


def main():
    rng = np.random.default_rng(SEED)
    with h5py.File(DATA_PATH, "r") as f:
        real = f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float32)
    n_real = len(real)

    # Disjoint real-image split BEFORE any fake generation
    perm = rng.permutation(n_real)
    n_hold = int(HOLDOUT_FRAC * n_real)
    holdout_idx = np.sort(perm[:n_hold])
    train_idx = np.sort(perm[n_hold:])
    real_train = real[train_idx]
    print(f"Reals: {len(train_idx)} train / {len(holdout_idx)} audit-holdout")

    n_per_type = len(real_train) // 4
    fake = make_fakes(real_train, n_per_type, rng)
    print(f"Fakes (from train reals only): {len(fake)}")

    x = np.concatenate([real_train, fake], axis=0)
    y = np.concatenate([np.ones(len(real_train)),
                        np.zeros(len(fake))]).astype(np.float32)
    shuffle = rng.permutation(len(x))
    x, y = x[shuffle], y[shuffle]
    n_val = int(0.1 * len(x))
    x_val, x_train = x[:n_val], x[n_val:]
    y_val, y_train = y[:n_val], y[n_val:]

    clf = build_classifier()
    clf.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(threshold=0.0)],
    )
    history = clf.fit(x_train, y_train, validation_data=(x_val, y_val),
                      epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)

    weights = {
        "c1_W": clf.get_layer("c1").kernel.numpy(),
        "c1_b": clf.get_layer("c1").bias.numpy(),
        "c2_W": clf.get_layer("c2").kernel.numpy(),
        "c2_b": clf.get_layer("c2").bias.numpy(),
        "score_W": clf.get_layer("score").kernel.numpy(),
        "score_b": clf.get_layer("score").bias.numpy(),
        "holdout_idx": holdout_idx,
        "train_idx": train_idx,
    }
    np.savez(WEIGHTS_PATH, **weights)

    metrics = {
        "seed": SEED,
        "epochs": EPOCHS,
        "n_real_train": int(len(real_train)),
        "n_real_holdout": int(len(holdout_idx)),
        "n_fake_train": int(len(fake)),
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_val_accuracy": float(history.history["val_binary_accuracy"][-1]),
        "note": "real images split 90/10 BEFORE fake generation; "
                "fakes derived from training reals only",
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved {WEIGHTS_PATH}, {METRICS_PATH}")


if __name__ == "__main__":
    main()
