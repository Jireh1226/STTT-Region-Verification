"""
Train one ReLU autoencoder per input region (centered, left, right, off_center).

This avoids the cross-pollination problem: a single AE trained on the whole
dataset learns to interpolate between centered and off-center images, producing
blended/averaged outputs. A subset-specific AE can only generate images from
its own regime, giving a tighter plausibility constraint per property.

Mapping:
  AE_centered   -> P1a, P1b  (all over C_centered: |CTE| < 1.0)
  AE_left       -> P4 C_left (CTE > 2.0)
  AE_right      -> P4 C_right (CTE < -2.0)
  AE_off_center -> P5 deadzone (|CTE| > 1.0)
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

NUM_PIXELS = 128
LATENT_DIM = 8
HIDDEN_1 = 64
HIDDEN_2 = 16

BATCH_SIZE = 128
EPOCHS = 200
LR = 1e-3


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float32)
        labels = f["X_train"][:]
    return images, labels


def build_autoencoder():
    inp = keras.Input(shape=(NUM_PIXELS,), name="pixels_in")
    h = layers.Dense(HIDDEN_1, activation="relu", name="enc1")(inp)
    h = layers.Dense(HIDDEN_2, activation="relu", name="enc2")(h)
    z = layers.Dense(LATENT_DIM, activation="linear", name="latent")(h)
    h = layers.Dense(HIDDEN_2, activation="relu", name="dec1")(z)
    h = layers.Dense(HIDDEN_1, activation="relu", name="dec2")(h)
    out = layers.Dense(NUM_PIXELS, activation="linear", name="pixels_out")(h)

    ae = keras.Model(inp, out, name="autoencoder")
    encoder = keras.Model(inp, z, name="encoder")
    return ae, encoder


def train_one(name, images):
    print(f"\n{'='*60}")
    print(f"Training AE: {name}  (n={len(images)})")
    print(f"{'='*60}")
    tf.keras.utils.set_random_seed(SEED)
    ae, encoder = build_autoencoder()
    ae.compile(optimizer=keras.optimizers.Adam(LR), loss="mse")

    # Split 90/10
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(images))
    n_val = max(10, int(0.1 * len(images)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    x_train, x_val = images[train_idx], images[val_idx]

    history = ae.fit(x_train, x_train,
                     validation_data=(x_val, x_val),
                     epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    final_tl = history.history["loss"][-1]
    final_vl = history.history["val_loss"][-1]

    # Evaluate reconstruction
    recon = ae.predict(images, batch_size=256, verbose=0)
    err = images - recon
    mse_per = np.mean(err ** 2, axis=1)
    linf_per = np.max(np.abs(err), axis=1)

    # Latents for all training images (for verification region)
    latents = encoder.predict(images, batch_size=256, verbose=0)

    metrics = {
        "name": name,
        "n_images": int(len(images)),
        "final_train_loss": float(final_tl),
        "final_val_loss": float(final_vl),
        "recon_mse_mean": float(mse_per.mean()),
        "recon_mse_median": float(np.median(mse_per)),
        "recon_mse_max": float(mse_per.max()),
        "recon_linf_mean": float(linf_per.mean()),
        "recon_linf_median": float(np.median(linf_per)),
        "recon_linf_max": float(linf_per.max()),
        "recon_linf_p95": float(np.percentile(linf_per, 95)),
        "recon_linf_p99": float(np.percentile(linf_per, 99)),
        "latent_min": latents.min(axis=0).tolist(),
        "latent_max": latents.max(axis=0).tolist(),
    }
    print(f"  Train loss: {final_tl:.6f} | Val loss: {final_vl:.6f}")
    print(f"  Recon MSE  (mean/max):  {metrics['recon_mse_mean']:.6f} / {metrics['recon_mse_max']:.6f}")
    print(f"  Recon Linf (mean/max):  {metrics['recon_linf_mean']:.6f} / {metrics['recon_linf_max']:.6f}")
    print(f"  Linf p95 / p99:         {metrics['recon_linf_p95']:.6f} / {metrics['recon_linf_p99']:.6f}")

    # Save decoder weights + latents
    saved = {
        "seed": SEED,
        "latent_dim": LATENT_DIM,
        "n_images": len(images),
        "latents": latents.astype(np.float32),
        "latent_min": latents.min(axis=0),
        "latent_max": latents.max(axis=0),
    }
    for layer_name in ["dec1", "dec2", "pixels_out"]:
        layer = ae.get_layer(layer_name)
        W, b = layer.get_weights()
        saved[f"{layer_name}_W"] = W.astype(np.float64)
        saved[f"{layer_name}_b"] = b.astype(np.float64)

    out_path = f"autoencoder_{name}_weights.npz"
    np.savez(out_path, **saved)
    print(f"  Saved: {out_path}")
    return metrics


def main():
    images, labels = load_data()
    cte = labels[:, 0]

    subsets = {
        "centered":   np.abs(cte) < 1.0,
        "left":       cte > 2.0,
        "right":      cte < -2.0,
        "off_center": np.abs(cte) > 1.0,
    }

    all_metrics = []
    for name, mask in subsets.items():
        metrics = train_one(name, images[mask])
        all_metrics.append(metrics)

    with open("autoencoder_subset_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Subset':<14} {'N':<6} {'Train MSE':<12} {'Linf mean':<12} {'Linf max':<12}")
    for m in all_metrics:
        print(f"{m['name']:<14} {m['n_images']:<6} {m['final_train_loss']:<12.6f} "
              f"{m['recon_linf_mean']:<12.4f} {m['recon_linf_max']:<12.4f}")


if __name__ == "__main__":
    main()
