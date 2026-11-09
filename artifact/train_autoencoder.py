"""
Train a ReLU-only autoencoder on the VerifyGAN TinyTaxiNet dataset.

The decoder will be composed with TinyTaxiNet in Marabou for plausibility-
constrained verification. All activations are ReLU (except the linear output
layer) so Marabou can encode the decoder natively.

Architecture:
  Encoder: 128 -> 32 -> 8 -> LATENT_DIM (ReLU on hidden, linear on latent)
  Decoder: LATENT_DIM -> 8 -> 32 -> 128 (ReLU on hidden, linear on output)

Outputs:
  - autoencoder_weights.npz: decoder weights (W, b per layer) and encoded training latents
  - autoencoder_metrics.json: reconstruction quality metrics

The script is deterministic: seed is fixed for numpy and TensorFlow.
"""
import json
import os
# Suppress TF noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import h5py

# Determinism
SEED = 42
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

from tensorflow import keras
from tensorflow.keras import layers

DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
WEIGHTS_PATH = "autoencoder_weights.npz"
METRICS_PATH = "autoencoder_metrics.json"

NUM_PIXELS = 128
LATENT_DIM = 8          # linear latent; use all 8 dims
HIDDEN_1 = 64
HIDDEN_2 = 16

BATCH_SIZE = 128
EPOCHS = 200
LR = 1e-3


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float32)
    return images


def build_autoencoder():
    inp = keras.Input(shape=(NUM_PIXELS,), name="pixels_in")
    # Encoder
    h = layers.Dense(HIDDEN_1, activation="relu", name="enc1")(inp)
    h = layers.Dense(HIDDEN_2, activation="relu", name="enc2")(h)
    z = layers.Dense(LATENT_DIM, activation="linear", name="latent")(h) # Linear latent
    # Decoder
    h = layers.Dense(HIDDEN_2, activation="relu", name="dec1")(z)
    h = layers.Dense(HIDDEN_1, activation="relu", name="dec2")(h)
    out = layers.Dense(NUM_PIXELS, activation="linear", name="pixels_out")(h)

    full = keras.Model(inp, out, name="autoencoder")
    encoder = keras.Model(inp, z, name="encoder")

    latent_in = keras.Input(shape=(LATENT_DIM,), name="latent_in")
    h2 = full.get_layer("dec1")(latent_in)
    h2 = full.get_layer("dec2")(h2)
    out2 = full.get_layer("pixels_out")(h2)
    decoder = keras.Model(latent_in, out2, name="decoder")

    return full, encoder, decoder


def main():
    print("Loading data...")
    images = load_data()
    print(f"Loaded {len(images)} images, shape {images.shape}")

    print("Building autoencoder...")
    ae, encoder, decoder = build_autoencoder()
    ae.compile(optimizer=keras.optimizers.Adam(LR), loss="mse")
    ae.summary()

    # Split for validation
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(images))
    n_val = int(0.1 * len(images))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    x_train = images[train_idx]
    x_val = images[val_idx]

    print(f"\nTraining on {len(x_train)} images, validating on {len(x_val)}...")
    history = ae.fit(
        x_train, x_train,
        validation_data=(x_val, x_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    # Reconstruction quality
    recon = ae.predict(images, batch_size=256, verbose=0)
    err = images - recon
    mse_per_image = np.mean(err ** 2, axis=1)
    linf_per_image = np.max(np.abs(err), axis=1)

    # Latent codes for all training images (to compute verification region)
    latents = encoder.predict(images, batch_size=256, verbose=0)

    metrics = {
        "seed": SEED,
        "latent_dim": LATENT_DIM,
        "architecture": f"128 -> {HIDDEN_1} -> {HIDDEN_2} -> {LATENT_DIM} -> {HIDDEN_2} -> {HIDDEN_1} -> 128",
        "activations": "ReLU (hidden), linear output",
        "epochs": EPOCHS,
        "final_train_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "recon_mse_mean": float(mse_per_image.mean()),
        "recon_mse_median": float(np.median(mse_per_image)),
        "recon_mse_max": float(mse_per_image.max()),
        "recon_linf_mean": float(linf_per_image.mean()),
        "recon_linf_median": float(np.median(linf_per_image)),
        "recon_linf_max": float(linf_per_image.max()),
        "recon_linf_p95": float(np.percentile(linf_per_image, 95)),
        "recon_linf_p99": float(np.percentile(linf_per_image, 99)),
        "latent_min": latents.min(axis=0).tolist(),
        "latent_max": latents.max(axis=0).tolist(),
        "latent_mean": latents.mean(axis=0).tolist(),
        "latent_std": latents.std(axis=0).tolist(),
    }
    print(f"\nReconstruction metrics:")
    print(f"  Train loss: {metrics['final_train_loss']:.6f}")
    print(f"  Val   loss: {metrics['final_val_loss']:.6f}")
    print(f"  MSE  (mean/median/max): {metrics['recon_mse_mean']:.6f} / {metrics['recon_mse_median']:.6f} / {metrics['recon_mse_max']:.6f}")
    print(f"  Linf (mean/median/max): {metrics['recon_linf_mean']:.6f} / {metrics['recon_linf_median']:.6f} / {metrics['recon_linf_max']:.6f}")
    print(f"  Linf p95 / p99: {metrics['recon_linf_p95']:.6f} / {metrics['recon_linf_p99']:.6f}")
    print(f"  Latent min: {metrics['latent_min']}")
    print(f"  Latent max: {metrics['latent_max']}")

    # Extract decoder weights
    dec_layers = ["dec1", "dec2", "pixels_out"]
    saved = {
        "seed": SEED,
        "latent_dim": LATENT_DIM,
        "latents": latents.astype(np.float32),
        "latent_min": latents.min(axis=0),
        "latent_max": latents.max(axis=0),
    }
    for name in dec_layers:
        layer = ae.get_layer(name)
        W, b = layer.get_weights()
        saved[f"{name}_W"] = W.astype(np.float64)      # Keras: shape (input, output)
        saved[f"{name}_b"] = b.astype(np.float64)

    np.savez(WEIGHTS_PATH, **saved)
    print(f"\nSaved weights to {WEIGHTS_PATH}")
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
