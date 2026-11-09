"""
Six-region constructors for the MNIST MLP, mirroring the TinyTaxiNet R1-R6
taxonomy. Each constructor takes a fresh Marabou network (loaded from
mnist_mlp.onnx) plus context and adds the region constraints to its input
variables. The output is a Marabou query ready for solve().

Mapping from the TinyTaxiNet taxonomy:
  R1: Raw per-pixel dataset box, calibrated on a class subset (training images
      of class k). Analog of TaxiNet "centered subset box".
  R2: R1 + total-pixel-mass invariant calibrated on class k
      (sum(x) in [mu_k - 2*sigma_k, mu_k + 2*sigma_k]). MNIST has no
      mean-center preprocessing so this is the closest single-linear-invariant
      analog of TaxiNet's mean(x)=0.5.
  R3: AE decoder-box composition. A small ReLU AE trained per class, decoder
      composed with the classifier. Verification region: training-set latent
      box. Implementation note: composition done at Marabou level by introducing
      auxiliary latent variables and 784 linear equalities, since we cannot
      rebuild a single composed ONNX without re-exporting.
  R4: Local PCA per-image, same-class neighbors. Verification region is the
      affine span around x0 in the top-d directions of the same-class deviation
      matrix, scaled by alpha * sigma_j.
  R5: Label-conditioned per-pixel box. Per-pixel [min, max] over same-class
      training neighbors of x0, intersected with [0,1].
  R6: Math-encoded constraints recalibrated on MNIST class images:
      - TV bound on the 28x28 adjacency set (|E|=1512: 28*27 horizontal +
        27*28 vertical)
      - Fourier high-frequency L1 bound on cells (k>=14 or l>=14)
      - Photometric basis residual with two MNIST-friendly bases (uniform,
        radial-distance)

All regions also enforce 0 <= x_i <= 1.
"""
import math
import numpy as np


N_PIX = 784
ROWS, COLS = 28, 28


def _adj_pairs():
    pairs = []
    for i in range(ROWS):
        for j in range(COLS):
            if j + 1 < COLS:
                pairs.append((i * COLS + j, i * COLS + j + 1))
            if i + 1 < ROWS:
                pairs.append((i * COLS + j, (i + 1) * COLS + j))
    return pairs


PAIRS = _adj_pairs()
NUM_PAIRS = len(PAIRS)


def calibrate_class_box(X_train, y_train, class_k):
    """Per-pixel [min, max] over training images of class_k."""
    mask = y_train == class_k
    Xk = X_train[mask].astype(np.float64)
    return Xk.min(axis=0), Xk.max(axis=0)


def calibrate_class_sum(X_train, y_train, class_k):
    """Mean and std of pixel sum across training images of class_k."""
    mask = y_train == class_k
    Xk = X_train[mask].astype(np.float64)
    sums = Xk.sum(axis=1)
    return float(sums.mean()), float(sums.std())


def add_R1_class_box(net, ivars, lo, hi):
    """R1: raw per-pixel dataset box over a class subset, clipped to [0,1]."""
    for i in range(N_PIX):
        net.setLowerBound(int(ivars[i]), float(max(0.0, lo[i])))
        net.setUpperBound(int(ivars[i]), float(min(1.0, hi[i])))


def add_R2_invariant(net, ivars, mu, sigma, n_std=2.0):
    """R2: R1 + sum(x) in [mu - n_std*sigma, mu + n_std*sigma]."""
    sum_var = int(net.getNewVariable())
    net.setLowerBound(sum_var, mu - n_std * sigma)
    net.setUpperBound(sum_var, mu + n_std * sigma)
    coeffs = [1.0] * N_PIX + [-1.0]
    vars_ = [int(ivars[i]) for i in range(N_PIX)] + [sum_var]
    net.addEquality(vars_, coeffs, 0.0)


def add_R3_ae_decoder(net, ivars, decoder_W1, decoder_b1, decoder_W2, decoder_b2, latent_lo, latent_hi):
    """R3: AE decoder-box composition (in-Marabou).

    Decoder architecture: latent (d) -> ReLU -> (h) -> linear -> 784 pixels.
    Adds:
      - d latent variables with bounds [latent_lo, latent_hi]
      - h hidden-pre-activation variables
      - h hidden-post-ReLU variables
      - 784 linear equalities tying decoded pixels to the hidden activations
    The classifier 784 inputs become a function of the latent box.

    decoder_W1: (h, d), decoder_b1: (h,)
    decoder_W2: (784, h), decoder_b2: (784,)
    """
    d = decoder_W1.shape[1]
    h = decoder_W1.shape[0]
    assert decoder_W2.shape == (N_PIX, h)
    assert latent_lo.shape == (d,) and latent_hi.shape == (d,)

    z_vars = []
    for j in range(d):
        v = int(net.getNewVariable())
        net.setLowerBound(v, float(latent_lo[j]))
        net.setUpperBound(v, float(latent_hi[j]))
        z_vars.append(v)

    pre_vars = []
    post_vars = []
    for k in range(h):
        pre = int(net.getNewVariable())
        post = int(net.getNewVariable())
        net.setLowerBound(post, 0.0)
        coeffs = list(decoder_W1[k]) + [-1.0]
        vars_ = z_vars + [pre]
        net.addEquality(vars_, coeffs, float(-decoder_b1[k]))
        net.addRelu(pre, post)
        pre_vars.append(pre)
        post_vars.append(post)

    for i in range(N_PIX):
        coeffs = list(decoder_W2[i]) + [-1.0]
        vars_ = post_vars + [int(ivars[i])]
        net.addEquality(vars_, coeffs, float(-decoder_b2[i]))

    for i in range(N_PIX):
        net.setLowerBound(int(ivars[i]), 0.0)
        net.setUpperBound(int(ivars[i]), 1.0)


def add_R4_local_pca(net, ivars, x0, V_top, sigmas, alpha):
    """R4: x = x0 + V_top^T z, |z_j| <= alpha * sigma_j, x clipped to [0,1].

    V_top: (d, 784) array of top-d principal directions
    sigmas: (d,) per-direction std of same-class-neighbor projections
    """
    d = V_top.shape[0]
    z_vars = []
    for j in range(d):
        v = int(net.getNewVariable())
        bound = alpha * sigmas[j]
        net.setLowerBound(v, float(-bound))
        net.setUpperBound(v, float(bound))
        z_vars.append(v)

    for i in range(N_PIX):
        coeffs = list(V_top[:, i]) + [-1.0]
        vars_ = z_vars + [int(ivars[i])]
        net.addEquality(vars_, coeffs, float(-x0[i]))

    for i in range(N_PIX):
        net.setLowerBound(int(ivars[i]), 0.0)
        net.setUpperBound(int(ivars[i]), 1.0)


def add_R5_label_conditioned_box(net, ivars, X_neighbors):
    """R5: per-pixel [min, max] over same-class neighbors, clipped to [0,1]."""
    lo = X_neighbors.min(axis=0)
    hi = X_neighbors.max(axis=0)
    for i in range(N_PIX):
        net.setLowerBound(int(ivars[i]), float(max(0.0, lo[i])))
        net.setUpperBound(int(ivars[i]), float(min(1.0, hi[i])))


def add_R6_TV(net, ivars, tau):
    """R6 (TV): sum_{(a,b) in adj} |x_a - x_b| <= tau."""
    d_vars = []
    for a, b in PAIRS:
        d = int(net.getNewVariable())
        net.setLowerBound(d, 0.0)
        d_vars.append(d)
        net.addInequality([int(ivars[a]), int(ivars[b]), d], [1.0, -1.0, -1.0], 0.0)
        net.addInequality([int(ivars[b]), int(ivars[a]), d], [1.0, -1.0, -1.0], 0.0)
    net.addInequality(d_vars, [1.0] * len(d_vars), float(tau))


def add_R6_Fourier(net, ivars, tau, dft_norm=None):
    """R6 (Fourier high-freq L1): sum over high-freq cells of |Re| + |Im| <= tau."""
    if dft_norm is None:
        dft_norm = 1.0 / math.sqrt(N_PIX)
    abs_vars = []
    for k in range(ROWS):
        for l in range(COLS):
            if not (k >= ROWS // 2 or l >= COLS // 2):
                continue
            for trig_name in ["cos", "sin"]:
                trig = math.cos if trig_name == "cos" else math.sin
                sgn = 1.0 if trig_name == "cos" else -1.0
                coeffs = []
                vars_ = []
                for i in range(ROWS):
                    for j in range(COLS):
                        phase = 2.0 * math.pi * (k * i / ROWS + l * j / COLS)
                        coeffs.append(dft_norm * trig(phase) * sgn)
                        vars_.append(int(ivars[i * COLS + j]))
                a = int(net.getNewVariable())
                net.setLowerBound(a, 0.0)
                abs_vars.append(a)
                net.addInequality(vars_ + [a], coeffs + [-1.0], 0.0)
                net.addInequality(vars_ + [a], [-c for c in coeffs] + [-1.0], 0.0)
    net.addInequality(abs_vars, [1.0] * len(abs_vars), float(tau))


def add_R6_photometric(net, ivars, eps, basis_2=None):
    """R6 (photometric): ||x - alpha*b1 - beta*b2||_inf <= eps.

    b1 = uniform / sqrt(N); b2 = radial-distance pattern (MNIST friendly,
    captures central-vs-peripheral brightness), unit-norm.
    """
    b1 = np.ones(N_PIX) / math.sqrt(N_PIX)
    if basis_2 is None:
        cy, cx = (ROWS - 1) / 2.0, (COLS - 1) / 2.0
        radial = np.zeros((ROWS, COLS))
        for i in range(ROWS):
            for j in range(COLS):
                radial[i, j] = math.sqrt((i - cy) ** 2 + (j - cx) ** 2)
        b2 = radial.flatten()
        b2 = b2 - b2.mean()
        b2 = b2 / np.linalg.norm(b2)
    else:
        b2 = basis_2

    alpha = int(net.getNewVariable()); net.setLowerBound(alpha, -1e3); net.setUpperBound(alpha, 1e3)
    beta = int(net.getNewVariable()); net.setLowerBound(beta, -1e3); net.setUpperBound(beta, 1e3)
    for i in range(N_PIX):
        net.addInequality([int(ivars[i]), alpha, beta],
                          [1.0, -float(b1[i]), -float(b2[i])], float(eps))
        net.addInequality([int(ivars[i]), alpha, beta],
                          [-1.0, float(b1[i]), float(b2[i])], float(eps))


def compute_local_pca(X_train, y_train, x0, class_k, d=8, return_neighbors=False):
    """For x0 of class_k, build the same-class-neighbor PCA region."""
    mask = y_train == class_k
    Xk = X_train[mask].astype(np.float64)
    diffs = Xk - x0[None, :]
    dists = np.linalg.norm(diffs, axis=1)
    n = min(100, mask.sum())
    nearest = np.argsort(dists)[1:n + 1]
    D = diffs[nearest]
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    V_top = Vt[:d]
    projections = D @ V_top.T
    sigmas = projections.std(axis=0)
    if return_neighbors:
        return V_top, sigmas, Xk[nearest]
    return V_top, sigmas


def get_same_class_neighbors(X_train, y_train, x0, class_k, k_nearest=100):
    """Return the k_nearest same-class images closest to x0 in L2."""
    mask = y_train == class_k
    Xk = X_train[mask].astype(np.float64)
    diffs = Xk - x0[None, :]
    dists = np.linalg.norm(diffs, axis=1)
    nearest = np.argsort(dists)[1:k_nearest + 1]
    return Xk[nearest]
