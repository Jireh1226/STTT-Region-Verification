"""
Marabou sanity check on the trained MNIST MLP.

1. Load mnist_mlp.onnx into Marabou.
2. Confirm inputs/outputs match the expected shape (784 in, 10 out).
3. Forward-evaluate Marabou's network on a real test image and compare against
   numpy chain-rule evaluation using the saved weights.
4. Run a trivial R1-style local L_inf robustness query on a single image:
   eps = 0.001 around image 0, check whether the true-class logit can be
   exceeded by any other class. This is the smallest meaningful smoke test.
"""
import time
import numpy as np
from maraboupy import Marabou


ONNX_PATH = "mnist_mlp.onnx"
DATA_PATH = "mnist_test_data.npz"
WEIGHTS_PATH = "mnist_mlp_weights.npz"
TIMEOUT_S = 60


def numpy_forward(W0, b0, W1, b1, W2, b2, x):
    h1 = np.maximum(W0 @ x + b0, 0)
    h2 = np.maximum(W1 @ h1 + b1, 0)
    return W2 @ h2 + b2


def main():
    print(f"Loading {ONNX_PATH} into Marabou...", flush=True)
    net = Marabou.read_onnx(ONNX_PATH)
    ivars = np.asarray(net.inputVars[0]).flatten().tolist()
    ovars = np.asarray(net.outputVars[0]).flatten().tolist()
    print(f"  input  vars: {len(ivars)} (first 3: {ivars[:3]})")
    print(f"  output vars: {len(ovars)} (all: {ovars})")
    assert len(ivars) == 784, f"expected 784 inputs, got {len(ivars)}"
    assert len(ovars) == 10,  f"expected 10 outputs, got {len(ovars)}"

    w = np.load(WEIGHTS_PATH)
    W0, b0, W1, b1, W2, b2 = w["W0"], w["b0"], w["W1"], w["b1"], w["W2"], w["b2"]
    print(f"  weight shapes: {W0.shape}, {W1.shape}, {W2.shape}")

    data = np.load(DATA_PATH)
    X_te, y_te = data["X_test"], data["y_test"]

    print("\nForward-eval cross-check on image 0 (label=" + str(int(y_te[0])) + "):", flush=True)
    x0 = X_te[0].astype(np.float64)
    logits_np = numpy_forward(W0, b0, W1, b1, W2, b2, x0)
    logits_mb = net.evaluateWithoutMarabou([X_te[0:1]])[0].flatten()
    diff = np.linalg.norm(logits_np - logits_mb)
    print(f"  numpy logits: {logits_np[:5].round(4)} ...")
    print(f"  Marabou logits: {logits_mb[:5].round(4)} ...")
    print(f"  L2 diff: {diff:.2e}")
    print(f"  numpy argmax: {int(np.argmax(logits_np))}, Marabou argmax: {int(np.argmax(logits_mb))}")

    print("\nSmoke test: L_inf eps=0.001 around image 0; check whether any other class can beat the true class.")
    img_idx = 0
    eps = 0.001
    true_class = int(y_te[img_idx])
    x_ref = X_te[img_idx].astype(np.float64)

    # Find the runner-up class to use as the adversarial target
    other_logits = logits_np.copy()
    other_logits[true_class] = -np.inf
    target_class = int(np.argmax(other_logits))
    print(f"  true class: {true_class}, runner-up class: {target_class}")
    print(f"  margin true - runner-up: {logits_np[true_class] - logits_np[target_class]:.4f}")

    net2 = Marabou.read_onnx(ONNX_PATH)
    ivars2 = np.asarray(net2.inputVars[0]).flatten().tolist()
    ovars2 = np.asarray(net2.outputVars[0]).flatten().tolist()
    for i in range(784):
        net2.setLowerBound(int(ivars2[i]), float(max(0.0, x_ref[i] - eps)))
        net2.setUpperBound(int(ivars2[i]), float(min(1.0, x_ref[i] + eps)))
    # Postcondition (negated): target_logit >= true_logit
    # i.e. target - true >= 0   ->   true - target <= 0
    net2.addInequality(
        [int(ovars2[true_class]), int(ovars2[target_class])],
        [1.0, -1.0],
        0.0,
    )

    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    status, vals, _ = net2.solve(verbose=False, options=opts)
    elapsed = time.time() - t0
    status = str(status).strip().lower()
    print(f"  Marabou: {status.upper()}, t={elapsed:.2f}s")
    if "sat" in status and vals:
        x_witness = np.array([vals[int(ivars2[i])] for i in range(784)])
        l_inf = np.max(np.abs(x_witness - x_ref))
        logits_w = numpy_forward(W0, b0, W1, b1, W2, b2, x_witness)
        print(f"    witness L_inf from x_ref: {l_inf:.4f} (eps={eps})")
        print(f"    witness logits[true={true_class}]={logits_w[true_class]:.4f}, "
              f"logits[target={target_class}]={logits_w[target_class]:.4f}")
        print(f"    witness argmax: {int(np.argmax(logits_w))}")


if __name__ == "__main__":
    main()
