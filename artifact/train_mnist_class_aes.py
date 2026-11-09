"""
Train one ReLU autoencoder per MNIST class (10 AEs).

Architecture: 784 -> 32 -> 8 -> 32 -> 784  (encoder ReLU, decoder ReLU+linear).
Decoder structure matches mnist_regions.add_R3_ae_decoder():
  decoder_W1: (h=32, d=8), decoder_b1: (32,)   [latent -> ReLU hidden]
  decoder_W2: (784, 32),    decoder_b2: (784,) [hidden -> linear pixels]

Per-class to mirror TinyTaxiNet's subset-specific AEs.
For each class k:
  - train AE_k on training images of class k
  - record latent box (per-dim min/max over training latents)
  - record reconstruction L_inf median/p95 and classifier-output preservation
    (how much does the AE-reconstruction perturb the classifier logits?)

Output:
  mnist_class_aes.npz   - per-class decoder weights + latent boxes + audit metrics
"""
import json
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_ae():
    enc_in = layers.Input(shape=(784,))
    h_enc = layers.Dense(32, activation="relu")(enc_in)
    z = layers.Dense(8, activation=None)(h_enc)
    h_dec = layers.Dense(32, activation="relu")(z)
    out = layers.Dense(784, activation=None)(h_dec)
    ae = models.Model(enc_in, out)
    enc = models.Model(enc_in, z)
    return ae, enc


def main():
    (X_tr, y_tr), (X_te, y_te) = tf.keras.datasets.mnist.load_data()
    X_tr = X_tr.astype(np.float32).reshape(-1, 784) / 255.0
    X_te = X_te.astype(np.float32).reshape(-1, 784) / 255.0

    classifier = tf.keras.models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(10, activation=None),
    ])
    classifier.build((None, 784))
    w = np.load("mnist_mlp_weights.npz")
    classifier.layers[0].set_weights([w["W0"].T.astype(np.float32), w["b0"].astype(np.float32)])
    classifier.layers[1].set_weights([w["W1"].T.astype(np.float32), w["b1"].astype(np.float32)])
    classifier.layers[2].set_weights([w["W2"].T.astype(np.float32), w["b2"].astype(np.float32)])

    out = {"provenance": {
        "script": "train_mnist_class_aes.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "arch": "784->32->8->32->784, ReLU encoder, ReLU+linear decoder",
    }, "per_class": {}}

    weights_npz = {}

    for class_k in range(10):
        print(f"\n=== AE class {class_k} ===", flush=True)
        Xk = X_tr[y_tr == class_k]
        Xk_te = X_te[y_te == class_k]
        n_tr, n_te = Xk.shape[0], Xk_te.shape[0]

        ae, enc = build_ae()
        ae.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
        ae.fit(Xk, Xk, validation_data=(Xk_te, Xk_te), epochs=15, batch_size=64, verbose=0)

        Z_tr = enc.predict(Xk, batch_size=256, verbose=0)
        Z_te = enc.predict(Xk_te, batch_size=256, verbose=0)
        latent_lo = Z_tr.min(axis=0)
        latent_hi = Z_tr.max(axis=0)
        held_out_in_box = np.mean(np.all((Z_te >= latent_lo) & (Z_te <= latent_hi), axis=1))

        X_recon = ae.predict(Xk, batch_size=256, verbose=0)
        X_recon = np.clip(X_recon, 0.0, 1.0)
        linf_per_img = np.max(np.abs(X_recon - Xk), axis=1)
        recon_l_inf_med = float(np.median(linf_per_img))
        recon_l_inf_p95 = float(np.percentile(linf_per_img, 95))

        logits_real = classifier.predict(Xk, batch_size=256, verbose=0)
        logits_recon = classifier.predict(X_recon, batch_size=256, verbose=0)
        argmax_real = np.argmax(logits_real, axis=1)
        argmax_recon = np.argmax(logits_recon, axis=1)
        argmax_pres = float(np.mean(argmax_real == argmax_recon))
        margin_real = logits_real[np.arange(n_tr), argmax_real] - np.sort(logits_real, axis=1)[:, -2]
        margin_recon = logits_recon[np.arange(n_tr), argmax_real] - np.sort(logits_recon, axis=1)[:, -2]
        margin_shift_med = float(np.median(np.abs(margin_real - margin_recon)))
        margin_shift_p95 = float(np.percentile(np.abs(margin_real - margin_recon), 95))

        dec_W1, dec_b1 = ae.layers[3].get_weights()
        dec_W2, dec_b2 = ae.layers[4].get_weights()
        weights_npz[f"dec1_W_class{class_k}"] = dec_W1.T.astype(np.float64)
        weights_npz[f"dec1_b_class{class_k}"] = dec_b1.astype(np.float64)
        weights_npz[f"dec2_W_class{class_k}"] = dec_W2.T.astype(np.float64)
        weights_npz[f"dec2_b_class{class_k}"] = dec_b2.astype(np.float64)
        weights_npz[f"latent_lo_class{class_k}"] = latent_lo.astype(np.float64)
        weights_npz[f"latent_hi_class{class_k}"] = latent_hi.astype(np.float64)

        out["per_class"][str(class_k)] = {
            "n_train": int(n_tr),
            "n_test": int(n_te),
            "held_out_in_train_latent_box": float(held_out_in_box),
            "recon_L_inf_median": recon_l_inf_med,
            "recon_L_inf_p95": recon_l_inf_p95,
            "classifier_argmax_preservation": argmax_pres,
            "classifier_margin_shift_median": margin_shift_med,
            "classifier_margin_shift_p95": margin_shift_p95,
        }
        print(f"  n={n_tr}, in-box={held_out_in_box:.3f}, "
              f"recon L_inf med={recon_l_inf_med:.3f} p95={recon_l_inf_p95:.3f}, "
              f"argmax-pres={argmax_pres:.3f}, "
              f"margin shift med={margin_shift_med:.3f} p95={margin_shift_p95:.3f}",
              flush=True)

    np.savez("mnist_class_aes.npz", **weights_npz)
    with open("mnist_class_aes_audit.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved mnist_class_aes.npz and mnist_class_aes_audit.json", flush=True)


if __name__ == "__main__":
    main()
