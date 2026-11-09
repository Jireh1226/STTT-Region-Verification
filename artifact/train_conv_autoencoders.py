"""
Train one CONVOLUTIONAL ReLU autoencoder per subset (centered/left/right/off_center)
as a head-to-head against the existing fully-connected AE
(train_subset_autoencoders.py).

The conv AE tests whether the FC AE verdicts depend on a dense decoder without
spatial inductive bias. It encodes locality and translation-equivariance
explicitly while remaining Marabou-tractable (Conv2D, ReLU, Dense, Reshape,
Flatten are all piecewise-linear).

Architecture (Marabou-compatible end-to-end):
  Input(128) -> Reshape(8,16,1)
    -> Conv2D(4, 3x3, padding=same, ReLU)
    -> Conv2D(8, 3x3, padding=same, ReLU)
    -> Flatten(8*16*8 = 1024)
    -> Dense(32, ReLU)
    -> Dense(LATENT_DIM, linear)        # latent
    -> Dense(32, ReLU)
    -> Dense(1024, ReLU)
    -> Reshape(8,16,8)
    -> Conv2D(4, 3x3, padding=same, ReLU)
    -> Conv2D(1, 3x3, padding=same, linear)
    -> Flatten(128)

Saves: conv_autoencoder_{subset}_weights.npz, conv_autoencoder_{subset}_keras.h5
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
BATCH_SIZE = 128
EPOCHS = 200
LR = 1e-3


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float32)
        labels = f["X_train"][:]
    return images, labels


def build_conv_ae():
    inp = keras.Input(shape=(NUM_PIXELS,), name="pixels_in")
    h = layers.Reshape((8, 16, 1), name="enc_reshape")(inp)
    h = layers.Conv2D(4, 3, padding="same", activation="relu", name="enc_conv1")(h)
    h = layers.Conv2D(8, 3, padding="same", activation="relu", name="enc_conv2")(h)
    h = layers.Flatten(name="enc_flat")(h)
    h = layers.Dense(32, activation="relu", name="enc_dense")(h)
    z = layers.Dense(LATENT_DIM, activation="linear", name="latent")(h)

    h = layers.Dense(32, activation="relu", name="dec_dense1")(z)
    h = layers.Dense(8 * 16 * 8, activation="relu", name="dec_dense2")(h)
    h = layers.Reshape((8, 16, 8), name="dec_reshape")(h)
    h = layers.Conv2D(4, 3, padding="same", activation="relu", name="dec_conv1")(h)
    h = layers.Conv2D(1, 3, padding="same", activation="linear", name="dec_conv2")(h)
    out = layers.Flatten(name="pixels_out")(h)

    ae = keras.Model(inp, out, name="conv_autoencoder")
    encoder = keras.Model(inp, z, name="conv_encoder")
    return ae, encoder


def train_one(name, images):
    print(f"\n{'='*60}\nTraining Conv AE: {name}  (n={len(images)})\n{'='*60}", flush=True)
    tf.keras.utils.set_random_seed(SEED)
    ae, encoder = build_conv_ae()
    ae.compile(optimizer=keras.optimizers.Adam(LR), loss="mse")

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(images))
    n_val = max(10, int(0.1 * len(images)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    x_train, x_val = images[train_idx], images[val_idx]

    history = ae.fit(x_train, x_train,
                     validation_data=(x_val, x_val),
                     epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    final_tl = float(history.history["loss"][-1])
    final_vl = float(history.history["val_loss"][-1])

    recon = ae.predict(images, batch_size=256, verbose=0)
    err = images - recon
    linf_per = np.max(np.abs(err), axis=1)
    mse_per = np.mean(err ** 2, axis=1)

    latents = encoder.predict(images, batch_size=256, verbose=0)

    metrics = {
        "name": name,
        "n_images": int(len(images)),
        "final_train_loss": final_tl,
        "final_val_loss": final_vl,
        "recon_mse_mean": float(mse_per.mean()),
        "recon_mse_max": float(mse_per.max()),
        "recon_linf_mean": float(linf_per.mean()),
        "recon_linf_median": float(np.median(linf_per)),
        "recon_linf_max": float(linf_per.max()),
        "recon_linf_p95": float(np.percentile(linf_per, 95)),
        "recon_linf_p99": float(np.percentile(linf_per, 99)),
        "latent_min": latents.min(axis=0).tolist(),
        "latent_max": latents.max(axis=0).tolist(),
        "n_params": int(ae.count_params()),
    }
    print(f"  params={metrics['n_params']}  train={final_tl:.6f}  val={final_vl:.6f}", flush=True)
    print(f"  Linf mean={metrics['recon_linf_mean']:.4f}  max={metrics['recon_linf_max']:.4f}  "
          f"p95={metrics['recon_linf_p95']:.4f}", flush=True)

    # Save Keras model for ONNX composition
    ae.save(f"conv_autoencoder_{name}.keras")
    encoder.save(f"conv_encoder_{name}.keras")

    # Save weights for reproducibility and support audit
    saved = {
        "seed": SEED,
        "latent_dim": LATENT_DIM,
        "n_images": len(images),
        "latents": latents.astype(np.float32),
        "latent_min": latents.min(axis=0),
        "latent_max": latents.max(axis=0),
    }
    np.savez(f"conv_autoencoder_{name}_weights.npz", **saved)
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

    out = []
    for name, mask in subsets.items():
        out.append(train_one(name, images[mask]))

    with open("conv_autoencoder_subset_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved conv_autoencoder_subset_metrics.json", flush=True)


if __name__ == "__main__":
    main()
