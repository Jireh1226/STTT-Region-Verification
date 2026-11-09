"""
Train a real-vs-fake MNIST plausibility classifier.

Discriminates real MNIST digits from four hand-designed fake families that
solver-constructed witnesses are most likely to resemble:
  1. Per-pixel box corners: uniform sample inside per-pixel [min, max] over
     a class subset, biased toward corners (Bernoulli of the corner choice).
  2. Cross-image blends: convex combination of two random same-class images
     at random weight (mimics box-corner witnesses that mix patches).
  3. Uniform noise: i.i.d. Uniform(0, 1) per pixel.
  4. High-frequency texture: real image + high-frequency Gaussian noise with
     standard deviation 0.1.

Architecture: 784 -> 64 -> 32 -> 1 sigmoid (kept small for transparency).

Outputs:
  mnist_plausibility_classifier_weights.npz
  mnist_plausibility_classifier_metrics.json
"""
import json
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
rng = np.random.default_rng(SEED)


def gen_box_corner(X_train, y_train, n, class_k):
    Xk = X_train[y_train == class_k]
    lo = Xk.min(axis=0)
    hi = Xk.max(axis=0)
    p_corner = 0.7
    out = np.zeros((n, 784), dtype=np.float32)
    for i in range(n):
        u = rng.random(784)
        is_corner = rng.random(784) < p_corner
        corner_choice = rng.integers(0, 2, 784)
        v = np.where(is_corner,
                     np.where(corner_choice == 0, lo, hi),
                     lo + u * (hi - lo))
        out[i] = np.clip(v, 0.0, 1.0)
    return out


def gen_cross_blend(X_train, y_train, n, class_k):
    Xk = X_train[y_train == class_k]
    idx_a = rng.integers(0, len(Xk), n)
    idx_b = rng.integers(0, len(Xk), n)
    w = rng.random(n).reshape(-1, 1)
    blend = w * Xk[idx_a] + (1 - w) * Xk[idx_b]
    spike = (rng.random(blend.shape) < 0.1) * (rng.random(blend.shape) * 2 - 1) * 0.3
    return np.clip(blend + spike, 0.0, 1.0)


def gen_uniform_noise(n):
    return rng.random((n, 784)).astype(np.float32)


def gen_hf_noise(X_train, y_train, n, class_k):
    Xk = X_train[y_train == class_k]
    base = Xk[rng.integers(0, len(Xk), n)]
    noise = rng.normal(0, 0.1, base.shape).astype(np.float32)
    return np.clip(base + noise, 0.0, 1.0)


def build_dataset(X_tr, y_tr):
    X_real = X_tr.astype(np.float32)
    n_per_class = 600
    n_class = len(np.unique(y_tr))

    fake_lists = []
    for c in range(n_class):
        fake_lists.append(gen_box_corner(X_tr, y_tr, n_per_class // 4, c))
        fake_lists.append(gen_cross_blend(X_tr, y_tr, n_per_class // 4, c))
        fake_lists.append(gen_uniform_noise(n_per_class // 4))
        fake_lists.append(gen_hf_noise(X_tr, y_tr, n_per_class // 4, c))

    X_fake = np.concatenate(fake_lists, axis=0).astype(np.float32)
    n_real_subsample = X_fake.shape[0]
    real_idx = rng.choice(len(X_real), n_real_subsample, replace=False)
    X_real_sub = X_real[real_idx]

    X = np.concatenate([X_real_sub, X_fake], axis=0)
    y = np.concatenate([np.ones(n_real_subsample), np.zeros(X_fake.shape[0])])

    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    return X, y


def main():
    (X_tr_raw, y_tr), (X_te_raw, y_te) = tf.keras.datasets.mnist.load_data()
    X_tr = X_tr_raw.reshape(-1, 784).astype(np.float32) / 255.0
    X_te = X_te_raw.reshape(-1, 784).astype(np.float32) / 255.0

    X_train, y_train = build_dataset(X_tr, y_tr)
    X_val, y_val = build_dataset(X_te, y_te)

    print(f"train: real={int(y_train.sum())}, fake={int(len(y_train) - y_train.sum())}", flush=True)
    print(f"val  : real={int(y_val.sum())}, fake={int(len(y_val) - y_val.sum())}", flush=True)

    model = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=10, batch_size=128, verbose=2)

    val_pred = model.predict(X_val, batch_size=256, verbose=0).flatten()
    metrics = {
        "provenance": {
            "script": "train_mnist_plausibility_classifier.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "seed": SEED,
        },
        "arch": "784 -> 64 -> 32 -> 1 sigmoid",
        "fake_families": ["box_corner", "cross_blend", "uniform_noise", "hf_noise"],
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "val_accuracy": float(np.mean((val_pred > 0.5) == y_val)),
        "real_recall": float(np.mean(val_pred[y_val == 1] > 0.5)),
        "fake_recall": float(np.mean(val_pred[y_val == 0] < 0.5)),
        "real_mean_prob": float(np.mean(val_pred[y_val == 1])),
        "fake_mean_prob": float(np.mean(val_pred[y_val == 0])),
    }
    print(f"val acc={metrics['val_accuracy']:.4f}, real recall={metrics['real_recall']:.4f}, "
          f"fake recall={metrics['fake_recall']:.4f}", flush=True)

    weights = {}
    for k, layer in enumerate(model.layers):
        W, b = layer.get_weights()
        weights[f"W{k}"] = W.T.astype(np.float64)
        weights[f"b{k}"] = b.astype(np.float64)
    np.savez("mnist_plausibility_classifier_weights.npz", **weights)
    with open("mnist_plausibility_classifier_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved.", flush=True)


if __name__ == "__main__":
    main()
