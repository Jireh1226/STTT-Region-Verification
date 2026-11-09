"""
Postcondition-headroom diagnostic.

Hypothesis from the Jacobian sweep: HE is verified more often than CTE not because
the network is less sensitive to pixel perturbations on the heading output (it is
in fact 2.5x-4.9x MORE sensitive), but because the HE postcondition bound is
wider (|y_1| < 90 deg vs |y_0| < 10 m) relative to the operating point.

Compute, on the released dataset:
  1. Forward-pass output distribution on the centered subset and the full set:
       summary stats of CTE_pred and HE_pred.
  2. Headroom to the verification bound at each image:
       h_CTE(x) = 10  - |CTE_pred(x)|
       h_HE (x) = 90 - |HE_pred(x)|
     and the ratio h / (Jacobian L_inf), which is the local first-order pixel
     perturbation magnitude that would saturate the bound.
  3. Per-output worst-case linear envelope on the dataset box:
       for each output o, with Jacobian row J_o(x_ref) at the centered mean,
       max_{x in [lo,hi]} J_o^T (x - x_ref)
     vs the bound margin.

Saved to: diagnostic_cte_he_headroom.json
"""
import json
from datetime import datetime, timezone

import h5py
import numpy as np
from maraboupy import Marabou


NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "diagnostic_cte_he_headroom.json"

ROWS, COLS = 8, 16
N_PIX = ROWS * COLS

BOUND_CTE = 10.0
BOUND_HE  = 90.0


def load_net():
    net = Marabou.read_nnet(NNET_PATH)
    Ws = [np.asarray(w, dtype=np.float64) for w in net.weights]
    bs = [np.asarray(b, dtype=np.float64) for b in net.biases]
    return Ws, bs


def forward_batch(Ws, bs, X):
    A = X
    for k, (W, b) in enumerate(zip(Ws, bs)):
        Z = A @ W.T + b
        A = np.maximum(Z, 0.0) if k < len(Ws) - 1 else Z
    return A


def jacobian_at(Ws, bs, x):
    a = x
    pres = []
    for k, (W, b) in enumerate(zip(Ws, bs)):
        z = W @ a + b
        pres.append(z)
        a = np.maximum(z, 0.0) if k < len(Ws) - 1 else z
    J = np.eye(x.size)
    for k, W in enumerate(Ws):
        J = W @ J
        if k < len(Ws) - 1:
            mask = (pres[k] > 0).astype(np.float64)
            J = mask[:, None] * J
    return J


def main():
    Ws, bs = load_net()

    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].reshape(-1, N_PIX).astype(np.float64)
        labels = f["X_train"][:].astype(np.float64)
    cte = labels[:, 0]

    centered = images[np.abs(cte) < 1.0]
    n_c = centered.shape[0]
    Y_c = forward_batch(Ws, bs, centered)
    Y_all = forward_batch(Ws, bs, images)

    def out_stats(Y, tag):
        cte_pred = Y[:, 0]
        he_pred  = Y[:, 1]
        return {
            "n": int(Y.shape[0]),
            "tag": tag,
            "CTE_pred": {
                "mean": float(cte_pred.mean()),
                "abs_mean": float(np.abs(cte_pred).mean()),
                "abs_median": float(np.median(np.abs(cte_pred))),
                "abs_p95": float(np.percentile(np.abs(cte_pred), 95)),
                "abs_max": float(np.abs(cte_pred).max()),
                "headroom_to_10m_median": float(np.median(BOUND_CTE - np.abs(cte_pred))),
                "headroom_to_10m_p05":    float(np.percentile(BOUND_CTE - np.abs(cte_pred), 5)),
                "frac_inside_bound":      float(np.mean(np.abs(cte_pred) < BOUND_CTE)),
            },
            "HE_pred": {
                "mean": float(he_pred.mean()),
                "abs_mean": float(np.abs(he_pred).mean()),
                "abs_median": float(np.median(np.abs(he_pred))),
                "abs_p95": float(np.percentile(np.abs(he_pred), 95)),
                "abs_max": float(np.abs(he_pred).max()),
                "headroom_to_90deg_median": float(np.median(BOUND_HE - np.abs(he_pred))),
                "headroom_to_90deg_p05":    float(np.percentile(BOUND_HE - np.abs(he_pred), 5)),
                "frac_inside_bound":        float(np.mean(np.abs(he_pred) < BOUND_HE)),
            },
        }

    stats_c = out_stats(Y_c, "centered (|CTE|<1)")
    stats_all = out_stats(Y_all, "all 10000")

    # Pixel-perturbation magnitude that locally saturates the bound at each centered image
    # using the local L_inf norm of the Jacobian row.
    print("Computing per-image saturating-eps via Jacobian L_inf...", flush=True)
    sat_eps_cte, sat_eps_he = [], []
    for i, x in enumerate(centered):
        J = jacobian_at(Ws, bs, x)
        h_cte = BOUND_CTE - abs(float(Y_c[i, 0]))
        h_he  = BOUND_HE - abs(float(Y_c[i, 1]))
        linf_cte = float(np.max(np.abs(J[0])))
        linf_he  = float(np.max(np.abs(J[1])))
        # local linear saturating eps:  |dy| ~ ||J_o||_1 * ||dx||_inf
        l1_cte = float(np.sum(np.abs(J[0])))
        l1_he  = float(np.sum(np.abs(J[1])))
        if l1_cte > 0:
            sat_eps_cte.append(h_cte / l1_cte)
        if l1_he > 0:
            sat_eps_he.append(h_he / l1_he)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n_c}", flush=True)
    sat_eps_cte = np.array(sat_eps_cte)
    sat_eps_he  = np.array(sat_eps_he)

    saturating = {
        "definition": "eps_local_sat(x) = (bound - |y_pred(x)|) / ||J_row||_1   "
                      "(pixel-perturbation L_inf radius that linearly saturates the bound)",
        "CTE": {
            "median": float(np.median(sat_eps_cte)),
            "mean":   float(sat_eps_cte.mean()),
            "p05":    float(np.percentile(sat_eps_cte, 5)),
            "p95":    float(np.percentile(sat_eps_cte, 95)),
        },
        "HE": {
            "median": float(np.median(sat_eps_he)),
            "mean":   float(sat_eps_he.mean()),
            "p05":    float(np.percentile(sat_eps_he, 5)),
            "p95":    float(np.percentile(sat_eps_he, 95)),
        },
        "ratio_CTE_over_HE_median": float(np.median(sat_eps_cte) / np.median(sat_eps_he)),
    }

    # Dataset-box global linear envelope at centered mean
    print("Computing dataset-box linear envelope at centered mean...", flush=True)
    lo, hi = images.min(axis=0), images.max(axis=0)
    x_ref = centered.mean(axis=0)
    J_ref = jacobian_at(Ws, bs, x_ref)
    # max |J^T (x - x_ref)| over x in [lo, hi] using sign rule
    pos = np.maximum(0.0, J_ref)
    neg = np.minimum(0.0, J_ref)
    max_pos = pos @ (hi - x_ref) + neg @ (lo - x_ref)
    max_neg = pos @ (lo - x_ref) + neg @ (hi - x_ref)
    envelope = {
        "x_ref": "centered subset mean",
        "CTE_pred_at_ref": float((forward_batch(Ws, bs, x_ref[None]))[0, 0]),
        "HE_pred_at_ref":  float((forward_batch(Ws, bs, x_ref[None]))[0, 1]),
        "linear_max_dCTE_positive_dir": float(max_pos[0]),
        "linear_max_dCTE_negative_dir": float(max_neg[0]),
        "linear_max_dHE_positive_dir":  float(max_pos[1]),
        "linear_max_dHE_negative_dir":  float(max_neg[1]),
        "note": "first-order envelope at x_ref over the dataset box; valid only in the local linear region",
    }

    out = {
        "provenance": {
            "script": "diagnostic_cte_he_headroom.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "bounds": {"CTE_m": BOUND_CTE, "HE_deg": BOUND_HE},
        },
        "output_distribution_centered": stats_c,
        "output_distribution_all": stats_all,
        "local_saturating_eps_centered": saturating,
        "global_linear_envelope_centered_mean": envelope,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved {OUTPUT_JSON}", flush=True)
    print("\n=== KEY NUMBERS ===", flush=True)
    print(f"  centered subset (n={n_c}):", flush=True)
    sc = stats_c
    print(f"    |CTE_pred|: median={sc['CTE_pred']['abs_median']:.3f} m,  "
          f"p95={sc['CTE_pred']['abs_p95']:.3f} m,  max={sc['CTE_pred']['abs_max']:.3f} m",
          flush=True)
    print(f"    |HE_pred| : median={sc['HE_pred']['abs_median']:.3f} deg, "
          f"p95={sc['HE_pred']['abs_p95']:.3f} deg, max={sc['HE_pred']['abs_max']:.3f} deg",
          flush=True)
    print(f"    headroom to bound: CTE med={sc['CTE_pred']['headroom_to_10m_median']:.2f} m, "
          f"HE med={sc['HE_pred']['headroom_to_90deg_median']:.2f} deg",
          flush=True)
    print(f"  local saturating pixel-eps (L_inf radius to cross bound):",
          flush=True)
    print(f"    CTE: median={saturating['CTE']['median']:.4f}, p05={saturating['CTE']['p05']:.4f}",
          flush=True)
    print(f"    HE : median={saturating['HE']['median']:.4f}, p05={saturating['HE']['p05']:.4f}",
          flush=True)
    print(f"    ratio CTE/HE median = {saturating['ratio_CTE_over_HE_median']:.3f} "
          f"(>1 means HE saturates faster, <1 means CTE saturates faster)",
          flush=True)


if __name__ == "__main__":
    main()
