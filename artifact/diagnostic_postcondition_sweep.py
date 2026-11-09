#!/usr/bin/env python3
"""Controlled postcondition sweep: isolate postcondition headroom as the
causal driver of the CTE-only P3 failure pattern.

The headroom diagnostic (Section 5.2, diagnostic_cte_he_headroom.py) observes
that at a fixed perturbation radius the network's P3 violations are CTE-only,
and explains this in first-order margin terms: CTE has less postcondition
headroom (smaller |bound - |pred||) relative to its pixel sensitivity, giving
a smaller saturating radius than HE. That is a *correlational* explanation:
the two axes differ in sensitivity, headroom, prediction location, and
nonlinearity simultaneously.

This script makes the manipulation the explanation implies. It holds the
input region fixed (an L_inf ball of radius eps around each centered anchor)
and varies *only the postcondition bound*, then asks whether the violation
rate moves where headroom predicts. Two controlled comparisons:

  (A) Bound sweep. For each axis, sweep the postcondition bound b and report
      the first-order violation fraction over all 893 centered anchors. The
      first-order reachable extremum of axis a over the ball is
      |pred_a(x)| + eps * ||J_a(x)||_1, so anchor x violates bound b iff the
      headroom h_a = b - |pred_a(x)| is below eps*||J_a(x)||_1. If headroom
      drives the pattern, the 50% crossing bound should equal the
      headroom-predicted value median(|pred_a|) + eps*median(||J_a||_1).

  (B) Headroom-equalized bounds. The causal claim is that the per-axis
      violation rate at fixed eps depends only on the saturating radius
      eps*_a = headroom / sensitivity. Choose the HE bound so its median
      saturating radius equals CTE's at the paper bound (CTE 10 m), i.e.
      equalize headroom-to-sensitivity across axes, and show the CTE-only
      asymmetry disappears: both axes then violate at matched rates across a
      range of eps. Architecture-specific explanations cannot be removed by
      a postcondition-bound change; a headroom explanation can.

The exact local ReLU Jacobian is used (closed form; the existing diagnostic
verifies it against finite differences below 1e-11). Pure numpy + h5py, no
TensorFlow/ONNX/Marabou, so it runs in the TF-free reproduction tier. An
optional --marabou flag confirms the first-order flips against the exact
solver on the 20-image P3 sample.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
NNET = ROOT / "VerifyGAN/models/TinyTaxiNet.nnet"
OUT = ROOT / "diagnostic_postcondition_sweep.json"

N_PIX = 128
CTE_BOUND = 10.0          # paper P3 CTE safety bound (m)
HE_BOUND = 90.0           # paper P3 HE safety bound (deg)
EPS_PRIMARY = 0.05        # fixed region radius for the headline sweep


def load_nnet(path: Path):
    """Pure-numpy .nnet parser (weights as (out,in) row-major, biases)."""
    lines = path.read_text().splitlines()
    i = 0
    while lines[i].startswith("//"):
        i += 1
    num_layers = int(lines[i].split(",")[0])
    i += 1
    sizes = [int(x) for x in lines[i].split(",") if x.strip()]
    i += 1
    i += 5  # skip symmetric flag, mins, maxes, means, ranges
    Ws, bs = [], []
    for layer in range(num_layers):
        rows, cols = sizes[layer + 1], sizes[layer]
        W = np.zeros((rows, cols))
        for r in range(rows):
            W[r] = [float(x) for x in lines[i].strip().rstrip(",").split(",")][:cols]
            i += 1
        b = np.zeros(rows)
        for r in range(rows):
            b[r] = float(lines[i].strip().rstrip(","))
            i += 1
        Ws.append(W)
        bs.append(b)
    return Ws, bs


def forward_batch(Ws, bs, X):
    h = X
    for k, (W, b) in enumerate(zip(Ws, bs)):
        h = h @ W.T + b
        if k < len(Ws) - 1:
            h = np.maximum(h, 0.0)
    return h


def jacobian_l1(Ws, bs, x):
    """Per-output L1 norm of the exact local Jacobian at x: returns
    (||J_CTE||_1, ||J_HE||_1)."""
    a = x
    pres = []
    for k, (W, b) in enumerate(zip(Ws, bs)):
        z = W @ a + b
        pres.append(z)
        a = np.maximum(z, 0.0) if k < len(Ws) - 1 else z
    J = np.eye(len(x))
    for k, W in enumerate(Ws):
        J = W @ J
        if k < len(Ws) - 1:
            J = (pres[k] > 0)[:, None] * J
    return np.abs(J).sum(axis=1)   # (2,)


def violation_fraction(pred_abs, jl1, eps, bound):
    """First-order: fraction of anchors whose axis output can exceed `bound`
    somewhere in the eps-ball, i.e. |pred| + eps*||J||_1 > bound."""
    return float(np.mean(pred_abs + eps * jl1 > bound))


def crossing_bound(pred_abs, jl1, eps, grid):
    """Smallest bound on the grid at which first-order violation drops to or
    below 50%."""
    for b in grid:
        if violation_fraction(pred_abs, jl1, eps, b) <= 0.5:
            return float(b)
    return float(grid[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marabou", action="store_true",
                    help="confirm first-order flips with the exact solver "
                         "on the 20-image P3 sample (needs maraboupy, no TF)")
    args = ap.parse_args()

    with h5py.File(DATA, "r") as f:
        images = f["y_train"][:].reshape(-1, N_PIX).astype(np.float64)
        labels = f["X_train"][:].astype(np.float64)
    Ws, bs = load_nnet(NNET)

    centered = images[np.abs(labels[:, 0]) < 1.0]
    preds = forward_batch(Ws, bs, centered)               # (n,2)
    pred_abs = np.abs(preds)                               # (n,2)
    jl1 = np.array([jacobian_l1(Ws, bs, x) for x in centered])  # (n,2)
    n = len(centered)

    cte_pred_abs, he_pred_abs = pred_abs[:, 0], pred_abs[:, 1]
    cte_jl1, he_jl1 = jl1[:, 0], jl1[:, 1]

    # Per-anchor saturating radius eps*_a = headroom / sensitivity at the
    # paper bounds (matches diagnostic_cte_he_headroom.py).
    sat_cte = (CTE_BOUND - cte_pred_abs) / cte_jl1
    sat_he = (HE_BOUND - he_pred_abs) / he_jl1

    result = {
        "provenance": {
            "script": "diagnostic_postcondition_sweep.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "method": "exact local ReLU Jacobian, first-order L_inf-ball "
                      "reachable extremum, pure numpy",
            "n_centered_anchors": n,
            "eps_primary": EPS_PRIMARY,
        },
        "sensitivity_and_headroom_medians": {
            "median_abs_pred_CTE_m": float(np.median(cte_pred_abs)),
            "median_abs_pred_HE_deg": float(np.median(he_pred_abs)),
            "median_J1_CTE": float(np.median(cte_jl1)),
            "median_J1_HE": float(np.median(he_jl1)),
            "median_saturating_radius_CTE": float(np.median(sat_cte)),
            "median_saturating_radius_HE": float(np.median(sat_he)),
        },
    }

    # (A) Bound sweep at fixed eps, paper bounds in context.
    cte_grid = np.arange(2.0, 22.0, 1.0)
    he_grid = np.arange(10.0, 140.0, 5.0)
    result["bound_sweep_eps_0.05"] = {
        "CTE": {
            "bounds": cte_grid.tolist(),
            "violation_fraction": [violation_fraction(cte_pred_abs, cte_jl1,
                                                      EPS_PRIMARY, b)
                                   for b in cte_grid],
            "fifty_pct_crossing_bound": crossing_bound(cte_pred_abs, cte_jl1,
                                                       EPS_PRIMARY, cte_grid),
            "headroom_predicted_crossing":
                float(np.median(cte_pred_abs) + EPS_PRIMARY * np.median(cte_jl1)),
            "violation_at_paper_bound_10m":
                violation_fraction(cte_pred_abs, cte_jl1, EPS_PRIMARY, CTE_BOUND),
        },
        "HE": {
            "bounds": he_grid.tolist(),
            "violation_fraction": [violation_fraction(he_pred_abs, he_jl1,
                                                     EPS_PRIMARY, b)
                                  for b in he_grid],
            "fifty_pct_crossing_bound": crossing_bound(he_pred_abs, he_jl1,
                                                      EPS_PRIMARY, he_grid),
            "headroom_predicted_crossing":
                float(np.median(he_pred_abs) + EPS_PRIMARY * np.median(he_jl1)),
            "violation_at_paper_bound_90deg":
                violation_fraction(he_pred_abs, he_jl1, EPS_PRIMARY, HE_BOUND),
        },
    }

    # (B) Headroom-equalized bounds: pick the HE bound whose median saturating
    # radius matches CTE's at the 10 m paper bound, then compare violation
    # rates across eps. Equalizing headroom/sensitivity should erase the
    # CTE-only asymmetry.
    target_sat = float(np.median(sat_cte))   # CTE saturating radius at 10 m
    # eps*_HE = (b_HE - |pred_HE|)/||J_HE||  ->  choose b_HE so median = target
    b_he_eq = float(np.median(he_pred_abs + target_sat * he_jl1))
    sat_he_eq = (b_he_eq - he_pred_abs) / he_jl1

    eps_grid = [0.02, 0.0345, 0.05, 0.08, 0.10]
    equalized = {
        "CTE_bound_m": CTE_BOUND,
        "HE_bound_deg_equalized": b_he_eq,
        "median_saturating_radius_CTE": target_sat,
        "median_saturating_radius_HE_equalized": float(np.median(sat_he_eq)),
        "by_eps": [],
    }
    for eps in eps_grid:
        equalized["by_eps"].append({
            "eps": eps,
            "CTE_violation_frac_at_10m":
                violation_fraction(cte_pred_abs, cte_jl1, eps, CTE_BOUND),
            "HE_violation_frac_at_paper_90deg":
                violation_fraction(he_pred_abs, he_jl1, eps, HE_BOUND),
            "HE_violation_frac_at_equalized_bound":
                violation_fraction(he_pred_abs, he_jl1, eps, b_he_eq),
        })
    result["headroom_equalized"] = equalized

    print(f"Centered anchors: {n}")
    print(f"Saturating radius (median): CTE={target_sat:.4f}, "
          f"HE={np.median(sat_he):.4f}")
    print(f"\n(A) eps={EPS_PRIMARY} bound sweep:")
    print(f"  CTE violation at paper 10 m: "
          f"{result['bound_sweep_eps_0.05']['CTE']['violation_at_paper_bound_10m']:.3f}; "
          f"50% crossing at {result['bound_sweep_eps_0.05']['CTE']['fifty_pct_crossing_bound']:.1f} m "
          f"(headroom predicts "
          f"{result['bound_sweep_eps_0.05']['CTE']['headroom_predicted_crossing']:.1f} m)")
    print(f"  HE  violation at paper 90 deg: "
          f"{result['bound_sweep_eps_0.05']['HE']['violation_at_paper_bound_90deg']:.3f}; "
          f"50% crossing at {result['bound_sweep_eps_0.05']['HE']['fifty_pct_crossing_bound']:.1f} deg "
          f"(headroom predicts "
          f"{result['bound_sweep_eps_0.05']['HE']['headroom_predicted_crossing']:.1f} deg)")
    print(f"\n(B) Headroom-equalized HE bound: {b_he_eq:.1f} deg "
          f"(matches CTE saturating radius {target_sat:.4f}):")
    for row in equalized["by_eps"]:
        print(f"  eps={row['eps']:.4f}: "
              f"CTE@10m={row['CTE_violation_frac_at_10m']:.3f}  "
              f"HE@90deg={row['HE_violation_frac_at_paper_90deg']:.3f}  "
              f"HE@{b_he_eq:.0f}deg={row['HE_violation_frac_at_equalized_bound']:.3f}")

    if args.marabou:
        confirm = marabou_confirm(images, labels, b_he_eq)
        result["marabou_confirmation"] = confirm
        # Also persist the solver block to its own file so a later Tier-1
        # (no --marabou) recompute of the main JSON does not clobber it.
        (ROOT / "diagnostic_postcondition_sweep_marabou.json").write_text(
            json.dumps(confirm, indent=2) + "\n")

    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nSaved {OUT}")


def marabou_confirm(images, labels, b_he_eq):
    """Exact-solver confirmation on the 20-image P3 sample at eps=0.05:
    CTE at 10 m (expect mostly violated) and HE at the headroom-equalized
    bound (expect a comparable violation rate), versus HE at 90 deg (expect
    none). maraboupy reads .nnet directly; no TensorFlow needed."""
    from maraboupy import Marabou
    eps = EPS_PRIMARY
    cte = labels[:, 0]
    idx = np.where(np.abs(cte) < 1.0)[0]
    sample = idx[np.linspace(0, len(idx) - 1, 20, dtype=int)]

    def solve(x0, out_var, direction, bound):
        net = Marabou.read_nnet(str(NNET))
        iv = [int(v) for v in np.asarray(net.inputVars[0]).flatten()]
        for i in range(N_PIX):
            net.setLowerBound(iv[i], float(max(0.0, x0[i] - eps)))
            net.setUpperBound(iv[i], float(min(1.0, x0[i] + eps)))
        net.addEquality(iv, [1.0] * N_PIX, 0.5 * N_PIX)   # mean-constrained ball
        if direction == ">=":
            net.setLowerBound(out_var, float(bound))
        else:
            net.setUpperBound(out_var, float(bound))
        opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=60)
        status, _, _ = net.solve(verbose=False, options=opts)
        return str(status).strip().lower()

    cte_violated = he90_violated = he_eq_violated = 0
    for s in sample:
        x0 = images[s]
        # axis violated iff EITHER one-sided bound is SAT
        if any(solve(x0, 192, d, b) == "sat"
               for d, b in [(">=", CTE_BOUND), ("<=", -CTE_BOUND)]):
            cte_violated += 1
        if any(solve(x0, 193, d, b) == "sat"
               for d, b in [(">=", HE_BOUND), ("<=", -HE_BOUND)]):
            he90_violated += 1
        if any(solve(x0, 193, d, b) == "sat"
               for d, b in [(">=", b_he_eq), ("<=", -b_he_eq)]):
            he_eq_violated += 1
    out = {
        "eps": eps, "n_sample": len(sample),
        "CTE_violated_at_10m": cte_violated,
        "HE_violated_at_90deg": he90_violated,
        "HE_violated_at_equalized_bound": he_eq_violated,
        "equalized_bound_deg": b_he_eq,
        "note": "exact solver; mean-constrained ball matches the P3 region",
    }
    print(f"\nMarabou (n=20, eps={eps}): CTE@10m={cte_violated}/20, "
          f"HE@90deg={he90_violated}/20, "
          f"HE@{b_he_eq:.0f}deg={he_eq_violated}/20")
    return out


if __name__ == "__main__":
    main()
