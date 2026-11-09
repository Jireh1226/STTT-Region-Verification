"""
Train a small Marabou-tractable ReLU MLP on MNIST and export to ONNX.

Architecture: 784 -> 32 -> 16 -> 10, all ReLU hidden, linear output (logits).
Matches TinyTaxiNet's compact scale (32 + 16 = 48 hidden ReLUs, vs TaxiNet's 32)
so Marabou queries solve quickly. Inputs normalized to [0,1].

Outputs:
  mnist_mlp_weights.npz   - layer weights/biases as numpy
  mnist_mlp.onnx          - Marabou-loadable ONNX (10 logits, no softmax)
  mnist_mlp_metrics.json  - test accuracy, train accuracy, per-class acc
  mnist_test_data.npz     - test images and labels in [0,1] normalized form
"""
import json
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import tf2onnx


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def main():
    (X_tr, y_tr), (X_te, y_te) = tf.keras.datasets.mnist.load_data()
    X_tr = X_tr.astype(np.float32).reshape(-1, 784) / 255.0
    X_te = X_te.astype(np.float32).reshape(-1, 784) / 255.0
    print(f"train: {X_tr.shape}, test: {X_te.shape}, range: [{X_tr.min()}, {X_tr.max()}]")

    model = models.Sequential([
        layers.Input(shape=(784,), name="image"),
        layers.Dense(32, activation="relu", name="h1"),
        layers.Dense(16, activation="relu", name="h2"),
        layers.Dense(10, activation=None, name="logits"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    print(model.summary())

    history = model.fit(
        X_tr, y_tr,
        validation_split=0.1,
        epochs=30,
        batch_size=128,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_te, y_te, verbose=0)
    train_loss, train_acc = model.evaluate(X_tr, y_tr, verbose=0)
    preds_te = np.argmax(model.predict(X_te, batch_size=256, verbose=0), axis=1)

    per_class_acc = {}
    for c in range(10):
        m = (y_te == c)
        per_class_acc[str(c)] = float(np.mean(preds_te[m] == c))

    print(f"\ntrain_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
    print(f"per-class: {per_class_acc}")

    Ws, bs = [], []
    for layer in model.layers:
        if isinstance(layer, layers.Dense):
            W, b = layer.get_weights()
            Ws.append(W.T.astype(np.float64))
            bs.append(b.astype(np.float64))
    print(f"weight shapes: {[w.shape for w in Ws]}")
    print(f"bias shapes  : {[b.shape for b in bs]}")
    np.savez("mnist_mlp_weights.npz",
             W0=Ws[0], b0=bs[0], W1=Ws[1], b1=bs[1], W2=Ws[2], b2=bs[2])

    spec = (tf.TensorSpec((1, 784), tf.float32, name="image"),)
    onnx_model, _ = tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=13,
        output_path="mnist_mlp.onnx",
    )
    print("ONNX saved.")

    np.savez("mnist_test_data.npz",
             X_test=X_te.astype(np.float32),
             y_test=y_te.astype(np.int64),
             preds=preds_te.astype(np.int64))
    print("Test data saved.")

    with open("mnist_mlp_metrics.json", "w") as f:
        json.dump({
            "provenance": {
                "script": "train_mnist_marabou_mlp.py",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "seed": SEED,
                "tf_version": tf.__version__,
            },
            "architecture": "784 -> 32 -> 16 -> 10 (ReLU hidden, linear logits)",
            "train_acc": float(train_acc),
            "test_acc":  float(test_acc),
            "per_class_test_acc": per_class_acc,
            "epochs": 30,
            "input_normalization": "X / 255.0",
            "test_set_size": int(X_te.shape[0]),
        }, f, indent=2)
    print("Metrics saved.")


if __name__ == "__main__":
    main()
