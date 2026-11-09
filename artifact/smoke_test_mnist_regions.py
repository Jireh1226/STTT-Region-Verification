"""
Quick smoke test for verify_mnist_protocol.py components.

Runs ONE query of EACH region type on ONE anchor (image 0, class 7).
Validates the region constructors compile and Marabou solves without errors.
Times each region so we know typical query cost before launching the full run.
"""
import time, math
import numpy as np
import tensorflow as tf
from maraboupy import Marabou
import mnist_regions as mr


ONNX_PATH = "mnist_mlp.onnx"
DATA_PATH = "mnist_test_data.npz"
AE_PATH = "mnist_class_aes.npz"
WEIGHTS_PATH = "mnist_mlp_weights.npz"
TIMEOUT_S = 60


def numpy_forward(W0, b0, W1, b1, W2, b2, x):
    h1 = np.maximum(W0 @ x + b0, 0)
    h2 = np.maximum(W1 @ h1 + b1, 0)
    return W2 @ h2 + b2


def solve(net, label):
    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    try:
        status, vals, _ = net.solve(verbose=False, options=opts)
    except Exception as e:
        elapsed = time.time() - t0
        return f"error: {str(e)[:80]}", elapsed, {}
    elapsed = time.time() - t0
    return str(status).strip().lower(), elapsed, vals


def main():
    data = np.load(DATA_PATH)
    X_te, y_te = data["X_test"], data["y_test"]

    w = np.load(WEIGHTS_PATH)
    Ws = (w["W0"], w["b0"], w["W1"], w["b1"], w["W2"], w["b2"])
    ae = np.load(AE_PATH)

    (X_tr, y_tr), _ = tf.keras.datasets.mnist.load_data()
    X_tr = X_tr.astype(np.float64).reshape(-1, 784) / 255.0

    anchor = 0
    x_ref = X_te[anchor].astype(np.float64)
    true_c = int(y_te[anchor])
    logits = numpy_forward(*Ws, x_ref)
    other = logits.copy(); other[true_c] = -np.inf
    target_c = int(np.argmax(other))
    print(f"anchor {anchor}: true_class={true_c}, target={target_c}, "
          f"margin={logits[true_c]-logits[target_c]:.4f}")

    Xk = X_tr[y_tr == true_c]
    lo, hi = Xk.min(axis=0), Xk.max(axis=0)
    sum_mu = float(Xk.sum(axis=1).mean())
    sum_sd = float(Xk.sum(axis=1).std())
    D = np.zeros((len(Xk), len(mr.PAIRS)))
    for k, (a, b) in enumerate(mr.PAIRS):
        D[:, k] = np.abs(Xk[:, a] - Xk[:, b])
    tv_p90 = float(np.percentile(D.sum(axis=1), 90))
    Xk2d = Xk.reshape(-1, 28, 28)
    F = np.fft.fft2(Xk2d, norm="ortho")
    mask = np.zeros((28, 28), bool)
    for k in range(28):
        for l in range(28):
            if k >= 14 or l >= 14:
                mask[k, l] = True
    hf_p90 = float(np.percentile((np.abs(F.real) + np.abs(F.imag))[:, mask].sum(axis=1), 90))

    eps = 0.01

    def fresh():
        net = Marabou.read_onnx(ONNX_PATH)
        ivars = [int(v) for v in np.asarray(net.inputVars[0]).flatten()]
        ovars = [int(v) for v in np.asarray(net.outputVars[0]).flatten()]
        return net, ivars, ovars

    def post(net, ovars):
        net.addInequality([int(ovars[true_c]), int(ovars[target_c])], [1.0, -1.0], 0.0)

    print("\n--- region smoke tests ---")
    regions = []

    # R1 L_inf
    net, iv, ov = fresh()
    for i in range(784):
        net.setLowerBound(int(iv[i]), float(max(0.0, x_ref[i] - eps)))
        net.setUpperBound(int(iv[i]), float(min(1.0, x_ref[i] + eps)))
    post(net, ov)
    regions.append(("R1_Linf_eps0.01", solve(net, "R1L")))

    # R2 L_inf + sum invariant
    net, iv, ov = fresh()
    for i in range(784):
        net.setLowerBound(int(iv[i]), float(max(0.0, x_ref[i] - eps)))
        net.setUpperBound(int(iv[i]), float(min(1.0, x_ref[i] + eps)))
    mr.add_R2_invariant(net, iv, sum_mu, sum_sd, n_std=2.0)
    post(net, ov)
    regions.append(("R2_Linf+sum", solve(net, "R2L")))

    # R1 class-box
    net, iv, ov = fresh()
    mr.add_R1_class_box(net, iv, lo, hi)
    post(net, ov)
    regions.append(("R1_class_box", solve(net, "R1cb")))

    # R3 AE
    net, iv, ov = fresh()
    mr.add_R3_ae_decoder(net, iv,
                          ae[f"dec1_W_class{true_c}"], ae[f"dec1_b_class{true_c}"],
                          ae[f"dec2_W_class{true_c}"], ae[f"dec2_b_class{true_c}"],
                          ae[f"latent_lo_class{true_c}"], ae[f"latent_hi_class{true_c}"])
    post(net, ov)
    regions.append(("R3_AE", solve(net, "R3")))

    # R4 PCA
    net, iv, ov = fresh()
    V_top, sigmas = mr.compute_local_pca(X_tr, y_tr, x_ref, true_c, d=8)
    mr.add_R4_local_pca(net, iv, x_ref, V_top, sigmas, alpha=2.0)
    post(net, ov)
    regions.append(("R4_PCA_a2", solve(net, "R4")))

    # R5 label-cond
    net, iv, ov = fresh()
    Xn = mr.get_same_class_neighbors(X_tr, y_tr, x_ref, true_c, k_nearest=100)
    mr.add_R5_label_conditioned_box(net, iv, Xn)
    post(net, ov)
    regions.append(("R5_LabelBox", solve(net, "R5")))

    # R6 TV
    net, iv, ov = fresh()
    mr.add_R1_class_box(net, iv, lo, hi)
    mr.add_R6_TV(net, iv, tau=tv_p90)
    post(net, ov)
    regions.append(("R6_TV_p90", solve(net, "R6TV")))

    # R6 Fourier
    net, iv, ov = fresh()
    mr.add_R1_class_box(net, iv, lo, hi)
    mr.add_R6_Fourier(net, iv, tau=hf_p90)
    post(net, ov)
    regions.append(("R6_Fourier_p90", solve(net, "R6F")))

    print(f"\n{'Region':25s} {'Verdict':10s} {'Time':10s}")
    for name, (status, t, _) in regions:
        print(f"{name:25s} {status:10s} {t:.2f}s")


if __name__ == "__main__":
    main()
