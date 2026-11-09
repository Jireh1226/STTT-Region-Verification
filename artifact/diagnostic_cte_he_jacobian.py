"""
CTE vs HE input-Jacobian diagnostic.

For each centered image (|CTE| < 1 m) in the released TinyTaxiNet dataset,
compute the exact input-output Jacobian J in R^{2 x 128} of the trained ReLU MLP
at that point (closed form: product of weight matrices with diagonal ReLU masks).

Row 0 of J: d(CTE_pred) / dx,  row 1: d(HE_pred) / dx.

Aggregate per-output:
  - ||row||_2 over images (median, mean, std)
  - max|row| over images (median, mean)
  - participation ratio  PR(row) = (sum|J_i|)^2 / (n * sum J_i^2 )
    PR = 1 means all magnitude in one pixel; PR = 128 means perfectly uniform
  - effective sparsity  S(row) = (sum|J_i|)^2 / sum(J_i^2)   (between 1 and 128)
  - mean |J| spatial heatmap (8 x 16 grid)

Sanity-check: finite-difference (J approx (f(x+he) - f(x-he)) / 2h) on the
first few images for both outputs against the closed-form Jacobian.

Saved to: diagnostic_cte_he_jacobian.json
"""
import json
import math
from datetime import datetime, timezone

import h5py
import numpy as np
from maraboupy import Marabou


NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "diagnostic_cte_he_jacobian.json"

ROWS, COLS = 8, 16
N_PIX = ROWS * COLS


def load_net():
    net = Marabou.read_nnet(NNET_PATH)
    Ws = [np.asarray(w, dtype=np.float64) for w in net.weights]
    bs = [np.asarray(b, dtype=np.float64) for b in net.biases]
    return Ws, bs


def forward(Ws, bs, x):
    """Run forward pass, return list of (pre, post) per layer plus final output."""
    a = x
    pres, posts = [], []
    for k, (W, b) in enumerate(zip(Ws, bs)):
        z = W @ a + b
        pres.append(z)
        if k < len(Ws) - 1:
            a = np.maximum(z, 0.0)
        else:
            a = z
        posts.append(a)
    return pres, posts, a


def jacobian(Ws, bs, x):
    """Closed-form Jacobian d(output)/d(input) at x.

    For a piecewise-linear ReLU MLP, the local Jacobian is
    W_L @ D_{L-1} @ W_{L-1} @ ... @ D_1 @ W_1
    where D_k = diag(1[z_k > 0]).
    """
    pres, _, _ = forward(Ws, bs, x)
    n_in = Ws[0].shape[1]
    J = np.eye(n_in)
    for k, W in enumerate(Ws):
        J = W @ J
        if k < len(Ws) - 1:
            mask = (pres[k] > 0).astype(np.float64)
            J = mask[:, None] * J
    return J


def participation_ratio(v):
    """PR = (sum|v|)^2 / (n * sum v^2). 1 means concentrated, n means uniform."""
    n = v.size
    s1 = float(np.sum(np.abs(v)))
    s2 = float(np.sum(v ** 2))
    if s2 == 0:
        return 0.0
    return (s1 ** 2) / (n * s2)


def effective_sparsity(v):
    """(sum|v|)^2 / sum(v^2). Equals 1 if one nonzero entry, n if uniform."""
    s1 = float(np.sum(np.abs(v)))
    s2 = float(np.sum(v ** 2))
    if s2 == 0:
        return 0.0
    return (s1 ** 2) / s2


def fd_check(Ws, bs, x, h=1e-4):
    """Finite difference Jacobian as sanity check."""
    n_in = x.size
    n_out = Ws[-1].shape[0]
    J_fd = np.zeros((n_out, n_in))
    for i in range(n_in):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        _, _, fp = forward(Ws, bs, xp)
        _, _, fm = forward(Ws, bs, xm)
        J_fd[:, i] = (fp - fm) / (2 * h)
    return J_fd


def main():
    print("Loading network...", flush=True)
    Ws, bs = load_net()

    print("Loading dataset...", flush=True)
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].reshape(-1, N_PIX).astype(np.float64)
        labels = f["X_train"][:].astype(np.float64)
    cte = labels[:, 0]

    mask_centered = np.abs(cte) < 1.0
    centered = images[mask_centered]
    n = centered.shape[0]
    print(f"Centered images: n={n}", flush=True)

    print("Running closed-form vs finite-difference sanity check on 5 images...",
          flush=True)
    fd_rel_err = []
    for i in range(min(5, n)):
        x = centered[i]
        J = jacobian(Ws, bs, x)
        J_fd = fd_check(Ws, bs, x, h=1e-4)
        err = np.linalg.norm(J - J_fd) / (np.linalg.norm(J_fd) + 1e-12)
        fd_rel_err.append(float(err))
    print(f"  finite-diff relative errors: {fd_rel_err}", flush=True)

    print("Computing Jacobians for all centered images...", flush=True)
    J_cte = np.zeros((n, N_PIX))
    J_he = np.zeros((n, N_PIX))
    for i, x in enumerate(centered):
        J = jacobian(Ws, bs, x)
        J_cte[i] = J[0]
        J_he[i] = J[1]
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n}", flush=True)

    def stats(M, name):
        l2 = np.linalg.norm(M, axis=1)
        linf = np.max(np.abs(M), axis=1)
        l1 = np.sum(np.abs(M), axis=1)
        pr = np.array([participation_ratio(M[i]) for i in range(M.shape[0])])
        es = np.array([effective_sparsity(M[i]) for i in range(M.shape[0])])
        return {
            "name": name,
            "n": int(M.shape[0]),
            "l1":   {"mean": float(l1.mean()),   "median": float(np.median(l1)),   "std": float(l1.std())},
            "l2":   {"mean": float(l2.mean()),   "median": float(np.median(l2)),   "std": float(l2.std())},
            "linf": {"mean": float(linf.mean()), "median": float(np.median(linf)), "std": float(linf.std())},
            "participation_ratio": {"mean": float(pr.mean()), "median": float(np.median(pr)), "std": float(pr.std())},
            "effective_sparsity":  {"mean": float(es.mean()), "median": float(np.median(es)), "std": float(es.std())},
        }

    out = {
        "provenance": {
            "script": "diagnostic_cte_he_jacobian.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "nnet": NNET_PATH,
            "data": DATA_PATH,
            "subset": "centered: |CTE| < 1 m",
            "n_centered": int(n),
            "fd_sanity_rel_errors_first5": fd_rel_err,
        },
        "cte": stats(J_cte, "d(CTE)/dx"),
        "he":  stats(J_he,  "d(HE)/dx"),
        "ratio_cte_over_he": {
            "l1_median":   float(np.median(np.sum(np.abs(J_cte), axis=1)) /
                                np.median(np.sum(np.abs(J_he),  axis=1))),
            "l2_median":   float(np.median(np.linalg.norm(J_cte, axis=1)) /
                                np.median(np.linalg.norm(J_he,  axis=1))),
            "linf_median": float(np.median(np.max(np.abs(J_cte), axis=1)) /
                                np.median(np.max(np.abs(J_he),  axis=1))),
        },
        "mean_abs_heatmap_8x16": {
            "cte": np.mean(np.abs(J_cte), axis=0).reshape(ROWS, COLS).tolist(),
            "he":  np.mean(np.abs(J_he),  axis=0).reshape(ROWS, COLS).tolist(),
        },
        "row_marginals_mean_abs": {
            "cte_per_row": np.mean(np.abs(J_cte), axis=0).reshape(ROWS, COLS).sum(axis=1).tolist(),
            "he_per_row":  np.mean(np.abs(J_he),  axis=0).reshape(ROWS, COLS).sum(axis=1).tolist(),
            "cte_per_col": np.mean(np.abs(J_cte), axis=0).reshape(ROWS, COLS).sum(axis=0).tolist(),
            "he_per_col":  np.mean(np.abs(J_he),  axis=0).reshape(ROWS, COLS).sum(axis=0).tolist(),
        },
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    for tag in ("cte", "he"):
        s = out[tag]
        print(f"  {s['name']}: "
              f"L1 med={s['l1']['median']:.3f}, "
              f"L2 med={s['l2']['median']:.4f}, "
              f"Linf med={s['linf']['median']:.4f}, "
              f"PR med={s['participation_ratio']['median']:.3f}, "
              f"effSp med={s['effective_sparsity']['median']:.2f}",
              flush=True)
    r = out["ratio_cte_over_he"]
    print(f"  ratio (CTE / HE) median: L1={r['l1_median']:.2f}x, "
          f"L2={r['l2_median']:.2f}x, Linf={r['linf_median']:.2f}x",
          flush=True)


if __name__ == "__main__":
    main()
