"""
Object-specific math/physics regions: two new region constructions that go
beyond universal image priors toward runway-specific spatial structure.

(A) Per-cell mean bounds (2x2 and 4x4 patches): for each spatial cell, bound
    the cell mean by its empirical [p01, p99] range on the centered subset.
    This is a CNN-like spatial-layout prior (each cell typically has a
    specific brightness range) without learning.

(B) Per-pair Lipschitz bounds: for each adjacent pixel pair (i, j), bound
    |x_i - x_j| by tau_pair (calibrated to p99 of that specific pair's
    diff over centered images). This is the adaptive analog of the global
    TV bound: instead of Σ |Δ| ≤ τ_global, each pair has its own τ.

Output: verification_object_specific_math.json
"""
import json
import time
from datetime import datetime, timezone

import h5py
import numpy as np
from maraboupy import Marabou


NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "verification_object_specific_math.json"

NUM_PIXELS = 128
SUM_TARGET = 64.0
PIXEL_MIN, PIXEL_MAX = 0.0, 1.0
CTE_OUT_VAR = 192
HE_OUT_VAR = 193

P1_CTE_BOUND = 10.0
P1_HE_BOUND = 90.0
P5_DEADZONE = 0.01
TIMEOUT_S = 120

# Adjacent pixel pairs (TV-style)
_PAIRS = []
for i in range(8):
    for j in range(16):
        flat = i * 16 + j
        if j + 1 < 16:
            _PAIRS.append((flat, i * 16 + (j + 1)))
        if i + 1 < 8:
            _PAIRS.append((flat, (i + 1) * 16 + j))
assert len(_PAIRS) == 232


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        return (f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float64),
                f["X_train"][:].astype(np.float64))


def per_pixel_box(images_subset):
    return images_subset.min(axis=0), images_subset.max(axis=0)


# --- Per-cell mean bound encoding ---

def patch_indices(cell_h, cell_w):
    """Return list of (cell_id, [flat_indices_in_cell]) for cell_h x cell_w patches."""
    cells = []
    n_rows = 8 // cell_h
    n_cols = 16 // cell_w
    for cr in range(n_rows):
        for cc in range(n_cols):
            idxs = []
            for di in range(cell_h):
                for dj in range(cell_w):
                    i = cr * cell_h + di
                    j = cc * cell_w + dj
                    idxs.append(i * 16 + j)
            cells.append((cr * n_cols + cc, idxs))
    return cells, n_rows, n_cols


def calibrate_cell_means(images_subset, cell_h, cell_w, lo_q=1, hi_q=99):
    cells, n_rows, n_cols = patch_indices(cell_h, cell_w)
    means_per_cell = np.zeros((len(images_subset), len(cells)))
    for img_idx, img in enumerate(images_subset):
        for cell_id, idxs in cells:
            means_per_cell[img_idx, cell_id] = img[idxs].mean()
    bounds = []
    for cell_id, idxs in cells:
        lo = float(np.percentile(means_per_cell[:, cell_id], lo_q))
        hi = float(np.percentile(means_per_cell[:, cell_id], hi_q))
        bounds.append((cell_id, idxs, lo, hi, len(idxs)))
    return bounds, n_rows, n_cols


def add_cell_mean_constraints(net, input_vars, bounds):
    """For each cell, constrain (1/N) * sum(x[idxs]) in [lo, hi].
    Equivalent to: sum(x[idxs]) - N*lo >= 0 AND N*hi - sum(x[idxs]) >= 0.
    Use addInequality with sum <= N*hi and -sum <= -N*lo."""
    for cell_id, idxs, lo, hi, n in bounds:
        # sum(x[idxs]) <= n * hi
        net.addInequality([int(input_vars[i]) for i in idxs], [1.0]*n, float(n*hi))
        # -sum(x[idxs]) <= -n * lo  -->  use coefficients -1 and bound -n*lo
        net.addInequality([int(input_vars[i]) for i in idxs], [-1.0]*n, float(-n*lo))


# --- Per-pair Lipschitz bound encoding ---

def calibrate_per_pair_lipschitz(images_subset, q=99):
    """For each adjacent pair, get the q-th percentile of |x_a - x_b| over images."""
    tau_per_pair = []
    for a, b in _PAIRS:
        diffs = np.abs(images_subset[:, a] - images_subset[:, b])
        tau_per_pair.append(float(np.percentile(diffs, q)))
    return tau_per_pair


def add_per_pair_lipschitz(net, input_vars, tau_per_pair):
    """For each adjacent pair (a, b), add 0 <= |x_a - x_b| <= tau_pair via:
       x_a - x_b <= tau_pair  AND  x_b - x_a <= tau_pair."""
    for (a, b), tau in zip(_PAIRS, tau_per_pair):
        net.addInequality([int(input_vars[a]), int(input_vars[b])], [1.0, -1.0], float(tau))
        net.addInequality([int(input_vars[b]), int(input_vars[a])], [1.0, -1.0], float(tau))


# --- Query runners ---

def base_query(pixel_lo, pixel_hi):
    if np.any(pixel_lo > pixel_hi):
        return None, None
    if pixel_lo.sum() > SUM_TARGET + 1e-9 or pixel_hi.sum() < SUM_TARGET - 1e-9:
        return None, None
    net = Marabou.read_nnet(NNET_PATH)
    ivars = [int(v) for v in net.inputVars[0].flatten()]
    for i in range(NUM_PIXELS):
        net.setLowerBound(ivars[i], float(pixel_lo[i]))
        net.setUpperBound(ivars[i], float(pixel_hi[i]))
    net.addEquality(ivars, [1.0]*NUM_PIXELS, SUM_TARGET)
    return net, ivars


def run_single(name, pixel_lo, pixel_hi, region_setter, out_var, direc, bound):
    net, ivars = base_query(pixel_lo, pixel_hi)
    if net is None:
        return {"status": "region_empty", "time_seconds": 0.0}
    region_setter(net, ivars)
    if direc == ">=":
        net.setLowerBound(out_var, float(bound))
    else:
        net.setUpperBound(out_var, float(bound))
    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    status, vals, _ = net.solve(verbose=False, options=opts)
    elapsed = time.time() - t0
    out = {"status": status.strip().lower(), "time_seconds": round(elapsed, 3)}
    if status.strip().lower() == "sat":
        out["CTE"] = round(float(vals[CTE_OUT_VAR]), 4)
        out["HE"] = round(float(vals[HE_OUT_VAR]), 4)
    return out


def run_p5(name, pixel_lo, pixel_hi, region_setter, deadzone):
    net, ivars = base_query(pixel_lo, pixel_hi)
    if net is None:
        return {"status": "region_empty", "time_seconds": 0.0}
    region_setter(net, ivars)
    net.setLowerBound(CTE_OUT_VAR, -float(deadzone))
    net.setUpperBound(CTE_OUT_VAR,  float(deadzone))
    net.setLowerBound(HE_OUT_VAR,  -float(deadzone))
    net.setUpperBound(HE_OUT_VAR,   float(deadzone))
    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    status, vals, _ = net.solve(verbose=False, options=opts)
    elapsed = time.time() - t0
    out = {"status": status.strip().lower(), "time_seconds": round(elapsed, 3)}
    if status.strip().lower() == "sat":
        out["CTE"] = round(float(vals[CTE_OUT_VAR]), 4)
        out["HE"] = round(float(vals[HE_OUT_VAR]), 4)
    return out


def run_property_suite(label, boxes, region_setter):
    """Run P1a, P1b, P4_left, P4_right, P5 with region_setter applied."""
    results = {}
    print(f"  [{label}]")
    results["P1a"] = run_single("P1a", boxes["centered_lo"], boxes["centered_hi"],
                                region_setter, CTE_OUT_VAR, ">=", P1_CTE_BOUND)
    results["P1b"] = run_single("P1b", boxes["centered_lo"], boxes["centered_hi"],
                                region_setter, CTE_OUT_VAR, "<=", -P1_CTE_BOUND)
    results["P4_left"] = run_single("P4_left", boxes["left_lo"], boxes["left_hi"],
                                    region_setter, CTE_OUT_VAR, "<=", 0.0)
    results["P4_right"] = run_single("P4_right", boxes["right_lo"], boxes["right_hi"],
                                     region_setter, CTE_OUT_VAR, ">=", 0.0)
    results["P5"] = run_p5("P5", boxes["off_lo"], boxes["off_hi"],
                           region_setter, P5_DEADZONE)
    for k, v in results.items():
        ce = ", CTE=" + str(v["CTE"]) if v.get("CTE") is not None else ""
        print(f"    {k:9s} {v['status']:8s} t={v['time_seconds']:.2f}s{ce}")
    return results


def main():
    print("Loading data...", flush=True)
    images, labels = load_data()
    cte = labels[:, 0]
    centered_mask = np.abs(cte) < 1.0
    left_mask = cte > 2.0
    right_mask = cte < -2.0
    off_mask = np.abs(cte) > 1.0
    centered = images[centered_mask]
    left = images[left_mask]
    right = images[right_mask]
    off = images[off_mask]
    print(f"  centered={len(centered)} left={len(left)} right={len(right)} off={len(off)}")

    # Per-subset per-pixel boxes
    boxes = {}
    boxes["centered_lo"], boxes["centered_hi"] = per_pixel_box(centered)
    boxes["left_lo"],     boxes["left_hi"]     = per_pixel_box(left)
    boxes["right_lo"],    boxes["right_hi"]    = per_pixel_box(right)
    boxes["off_lo"],      boxes["off_hi"]      = per_pixel_box(off)

    out = {
        "provenance": {
            "script": "verify_object_specific_math.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timeout_seconds": TIMEOUT_S,
        },
    }

    # === Experiment A: per-cell mean bounds (2x2 patches) ===
    print("\n=== Per-cell mean bounds (2x2) ===", flush=True)
    bounds_2x2, _, _ = calibrate_cell_means(centered, 2, 2, lo_q=1, hi_q=99)
    print(f"  {len(bounds_2x2)} cells; example bounds (cell 0): "
          f"[{bounds_2x2[0][2]:.3f}, {bounds_2x2[0][3]:.3f}]", flush=True)
    def setter_2x2(net, ivars, b=bounds_2x2):
        add_cell_mean_constraints(net, ivars, b)
    out["cell_mean_2x2"] = {
        "n_cells": len(bounds_2x2),
        "calibration_quantiles": [1, 99],
        "results": run_property_suite("2x2 cells", boxes, setter_2x2),
    }

    # === Experiment B: per-cell mean bounds (4x4 patches) ===
    print("\n=== Per-cell mean bounds (4x4) ===", flush=True)
    bounds_4x4, _, _ = calibrate_cell_means(centered, 4, 4, lo_q=1, hi_q=99)
    print(f"  {len(bounds_4x4)} cells; example bounds (cell 0): "
          f"[{bounds_4x4[0][2]:.3f}, {bounds_4x4[0][3]:.3f}]", flush=True)
    def setter_4x4(net, ivars, b=bounds_4x4):
        add_cell_mean_constraints(net, ivars, b)
    out["cell_mean_4x4"] = {
        "n_cells": len(bounds_4x4),
        "calibration_quantiles": [1, 99],
        "results": run_property_suite("4x4 cells", boxes, setter_4x4),
    }

    # === Experiment C: per-pair Lipschitz bounds ===
    print("\n=== Per-pair Lipschitz bounds (p99) ===", flush=True)
    tau_per_pair = calibrate_per_pair_lipschitz(centered, q=99)
    tau_arr = np.array(tau_per_pair)
    print(f"  per-pair tau: min={tau_arr.min():.4f}, median={np.median(tau_arr):.4f}, "
          f"p95={np.percentile(tau_arr, 95):.4f}, max={tau_arr.max():.4f}", flush=True)
    def setter_pair_lip(net, ivars, taus=tau_per_pair):
        add_per_pair_lipschitz(net, ivars, taus)
    out["per_pair_lipschitz_p99"] = {
        "n_pairs": len(tau_per_pair),
        "calibration_quantile": 99,
        "tau_summary": {
            "min": float(tau_arr.min()),
            "median": float(np.median(tau_arr)),
            "p95": float(np.percentile(tau_arr, 95)),
            "max": float(tau_arr.max()),
        },
        "results": run_property_suite("per-pair Lipschitz p99", boxes, setter_pair_lip),
    }

    # === Experiment D: per-pair Lipschitz bounds, tighter (p90) ===
    print("\n=== Per-pair Lipschitz bounds (p90, tighter) ===", flush=True)
    tau_per_pair_p90 = calibrate_per_pair_lipschitz(centered, q=90)
    tau_arr_90 = np.array(tau_per_pair_p90)
    print(f"  per-pair tau: min={tau_arr_90.min():.4f}, median={np.median(tau_arr_90):.4f}, "
          f"p95={np.percentile(tau_arr_90, 95):.4f}, max={tau_arr_90.max():.4f}", flush=True)
    def setter_pair_lip_90(net, ivars, taus=tau_per_pair_p90):
        add_per_pair_lipschitz(net, ivars, taus)
    out["per_pair_lipschitz_p90"] = {
        "n_pairs": len(tau_per_pair_p90),
        "calibration_quantile": 90,
        "tau_summary": {
            "min": float(tau_arr_90.min()),
            "median": float(np.median(tau_arr_90)),
            "p95": float(np.percentile(tau_arr_90, 95)),
            "max": float(tau_arr_90.max()),
        },
        "results": run_property_suite("per-pair Lipschitz p90", boxes, setter_pair_lip_90),
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
