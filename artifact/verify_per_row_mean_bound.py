"""
Experiment 2: per-row mean bound as a fourth math region.

Hypothesis
----------
Real runway images have row-wise structure: some rows (sky/horizon) are
bright on average, others (tarmac) are dark. The per-row mean is a 1-D
summary along the vertical axis; calibrating it on the centered subset
gives 8 (min, max) intervals -- one per row -- that any real centered
image must lie inside.

We add this constraint on top of the per-pixel box and the global
mean(x) = 0.5 constraint and ask: does any of P1/P4/P5 flip verdict?
This complements TV/Fourier/photometric (which constrain local
smoothness, frequency, or 2-D photometric basis residual) by
constraining a row-wise linear summary statistic.

Math
----
For each row i in {0..7}, define
   row_mean_i(x) = (1/16) * sum_{j=0..15} x[i*16 + j].
The region is
   row_mean_min[i] <= row_mean_i(x) <= row_mean_max[i],   for all i.
Calibration: row_mean_min[i] := min over centered images of row_mean_i,
             row_mean_max[i] := max over centered images of row_mean_i.

Marabou encoding
----------------
For each row i, add two inequalities on the 16 input variables of row i:
   sum_j x[i,j] >= 16 * row_mean_min[i]
   sum_j x[i,j] <= 16 * row_mean_max[i].
No auxiliary variables needed; 8 rows * 2 = 16 inequalities total.

Properties tested under per-pixel box + mean(x)=0.5 + per-row-mean region:
  P1a:      CTE >= 10   on centered per-pixel box
  P1b:      CTE <= -10  on centered per-pixel box
  P4_left:  CTE <= 0    on left per-pixel box (wrong-sign query)
  P4_right: CTE >= 0    on right per-pixel box (wrong-sign query)
  P5:       |CTE|<=0.01 AND |HE|<=0.01 on off-center per-pixel box

Per-query timeout: 300 s. Output: verification_per_row_mean.json.

Usage:
  ./prophecy_env/bin/python verify_per_row_mean_bound.py
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
import time

import h5py
import numpy as np
from maraboupy import Marabou


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "verification_per_row_mean.json"

NUM_PIXELS = 128
ROWS, COLS = 8, 16
MEAN_TARGET = 0.5
SUM_TARGET = MEAN_TARGET * NUM_PIXELS

CTE_OUT_VAR = 192
HE_OUT_VAR = 193

P1_CTE_BOUND = 10.0
P5_DEADZONE = 0.01

TIMEOUT_S = 300


# -----------------------------------------------------------------------------
# Data + per-row calibration
# -----------------------------------------------------------------------------
def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].astype(np.float64)
        labels = f["X_train"][:].astype(np.float64)
    return images, labels


def compute_box(images: np.ndarray):
    flat = images.reshape(images.shape[0], -1)
    return flat.min(axis=0), flat.max(axis=0)


def row_means(flat_images: np.ndarray) -> np.ndarray:
    """Return an (N, ROWS) array of per-row means."""
    imgs = flat_images.reshape(-1, ROWS, COLS)
    return imgs.mean(axis=2)  # shape (N, ROWS)


def row_mean_bounds(flat_images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rm = row_means(flat_images)
    return rm.min(axis=0), rm.max(axis=0)


# -----------------------------------------------------------------------------
# Marabou encoding
# -----------------------------------------------------------------------------
def add_per_row_mean_bound(net,
                           input_vars: list[int],
                           row_mean_min: np.ndarray,
                           row_mean_max: np.ndarray) -> None:
    """For each row i: COLS * row_mean_min[i] <= sum_j x[i,j] <= COLS * row_mean_max[i]."""
    for i in range(ROWS):
        row_pix = [int(input_vars[i * COLS + j]) for j in range(COLS)]
        sum_lo = float(COLS * row_mean_min[i])
        sum_hi = float(COLS * row_mean_max[i])
        # sum >= sum_lo  <=>  -sum <= -sum_lo
        net.addInequality(row_pix, [-1.0] * COLS, -sum_lo)
        # sum <= sum_hi
        net.addInequality(row_pix, [1.0] * COLS, sum_hi)


def build_query(pixel_mins: np.ndarray,
                pixel_maxs: np.ndarray,
                row_mean_min: np.ndarray,
                row_mean_max: np.ndarray):
    net = Marabou.read_nnet(NNET_PATH)
    input_vars = [int(v) for v in net.inputVars[0].flatten()]
    assert len(input_vars) == NUM_PIXELS
    for i in range(NUM_PIXELS):
        net.setLowerBound(input_vars[i], float(pixel_mins[i]))
        net.setUpperBound(input_vars[i], float(pixel_maxs[i]))
    net.addEquality(input_vars, [1.0] * NUM_PIXELS, SUM_TARGET)
    add_per_row_mean_bound(net, input_vars, row_mean_min, row_mean_max)
    return net, input_vars


def _witness_row_means(flat128: np.ndarray) -> list[float]:
    img = np.asarray(flat128, dtype=np.float64).reshape(ROWS, COLS)
    return [round(float(v), 6) for v in img.mean(axis=1).tolist()]


def _record_sat_witness(vals, input_vars: list[int]) -> dict:
    cx = np.array([vals[v] for v in input_vars], dtype=np.float64)
    return {
        "CTE": round(float(vals[CTE_OUT_VAR]), 6),
        "HE": round(float(vals[HE_OUT_VAR]), 6),
        "witness_mean": round(float(cx.mean()), 6),
        "witness_row_means": _witness_row_means(cx),
        "witness": [round(float(v), 6) for v in cx.tolist()],
    }


def run_single_output_bound(pixel_mins: np.ndarray,
                            pixel_maxs: np.ndarray,
                            row_mean_min: np.ndarray,
                            row_mean_max: np.ndarray,
                            out_var: int,
                            direction: str,
                            bound: float) -> dict:
    net, input_vars = build_query(
        pixel_mins, pixel_maxs, row_mean_min, row_mean_max
    )
    if direction == ">=":
        net.setLowerBound(out_var, float(bound))
    else:
        net.setUpperBound(out_var, float(bound))
    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    try:
        status, vals, _ = net.solve(verbose=False, options=opts)
    except Exception as e:
        return {
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "time_seconds": round(time.time() - t0, 4),
        }
    elapsed = time.time() - t0
    status = (status or "").strip().lower()
    if status == "":
        status = "timeout"
    out: dict = {"status": status, "time_seconds": round(elapsed, 4)}
    if status == "sat":
        out.update(_record_sat_witness(vals, input_vars))
    return out


def run_p5(pixel_mins: np.ndarray,
           pixel_maxs: np.ndarray,
           row_mean_min: np.ndarray,
           row_mean_max: np.ndarray,
           deadzone: float = P5_DEADZONE) -> dict:
    net, input_vars = build_query(
        pixel_mins, pixel_maxs, row_mean_min, row_mean_max
    )
    net.setLowerBound(CTE_OUT_VAR, -deadzone)
    net.setUpperBound(CTE_OUT_VAR, deadzone)
    net.setLowerBound(HE_OUT_VAR, -deadzone)
    net.setUpperBound(HE_OUT_VAR, deadzone)
    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    try:
        status, vals, _ = net.solve(verbose=False, options=opts)
    except Exception as e:
        return {
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "time_seconds": round(time.time() - t0, 4),
        }
    elapsed = time.time() - t0
    status = (status or "").strip().lower()
    if status == "":
        status = "timeout"
    out: dict = {
        "status": status,
        "time_seconds": round(elapsed, 4),
        "deadzone": deadzone,
    }
    if status == "sat":
        out.update(_record_sat_witness(vals, input_vars))
    return out


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    print("Loading data...", flush=True)
    images, labels = load_data()
    cte = labels[:, 0]
    flat_images = images.reshape(images.shape[0], -1)

    centered = flat_images[np.abs(cte) < 1.0]
    left = flat_images[cte > 2.0]
    right = flat_images[cte < -2.0]
    off_center = flat_images[np.abs(cte) > 1.0]
    print(
        f"  N(total)={len(flat_images)}  centered={len(centered)}  "
        f"left={len(left)}  right={len(right)}  off_center={len(off_center)}",
        flush=True,
    )

    pixel_boxes = {
        "P1": compute_box(centered),
        "P4_left": compute_box(left),
        "P4_right": compute_box(right),
        "P5": compute_box(off_center),
    }

    # Per-row mean calibration on the centered subset (per the spec).
    print("Calibrating per-row mean bounds on centered subset...", flush=True)
    rm_min, rm_max = row_mean_bounds(centered)
    print(
        "  row_mean_min = "
        + ", ".join(f"{v:.4f}" for v in rm_min.tolist()),
        flush=True,
    )
    print(
        "  row_mean_max = "
        + ", ".join(f"{v:.4f}" for v in rm_max.tolist()),
        flush=True,
    )

    row_mean_calibration = {
        "subset": "centered (|CTE| < 1 m)",
        "n_images": int(len(centered)),
        "row_mean_min": [round(float(v), 6) for v in rm_min.tolist()],
        "row_mean_max": [round(float(v), 6) for v in rm_max.tolist()],
    }

    results: dict = {}

    print("\n=== P1a (CTE >= 10) on centered box ===", flush=True)
    pmin, pmax = pixel_boxes["P1"]
    results["P1a"] = run_single_output_bound(
        pmin, pmax, rm_min, rm_max, CTE_OUT_VAR, ">=", P1_CTE_BOUND
    )

    print("=== P1b (CTE <= -10) on centered box ===", flush=True)
    results["P1b"] = run_single_output_bound(
        pmin, pmax, rm_min, rm_max, CTE_OUT_VAR, "<=", -P1_CTE_BOUND
    )

    print("=== P4_left (CTE <= 0) on left box ===", flush=True)
    pmin, pmax = pixel_boxes["P4_left"]
    results["P4_left"] = run_single_output_bound(
        pmin, pmax, rm_min, rm_max, CTE_OUT_VAR, "<=", 0.0
    )

    print("=== P4_right (CTE >= 0) on right box ===", flush=True)
    pmin, pmax = pixel_boxes["P4_right"]
    results["P4_right"] = run_single_output_bound(
        pmin, pmax, rm_min, rm_max, CTE_OUT_VAR, ">=", 0.0
    )

    print("=== P5 (|CTE|<=0.01 & |HE|<=0.01) on off-center box ===", flush=True)
    pmin, pmax = pixel_boxes["P5"]
    results["P5"] = run_p5(pmin, pmax, rm_min, rm_max, deadzone=P5_DEADZONE)

    # Print summary
    for prop_name, r in results.items():
        extra = ""
        if r["status"] == "sat":
            extra = (
                f"  CTE={r['CTE']:+.4f}  HE={r['HE']:+.4f}  "
                f"mean={r['witness_mean']:.4f}"
            )
        print(
            f"  {prop_name:<8} {r['status'].upper():<7} "
            f"t={r['time_seconds']:>7.2f}s{extra}",
            flush=True,
        )

    output = {
        "provenance": {
            "script": "verify_per_row_mean_bound.py",
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "git_commit": _git_commit(),
            "nnet": NNET_PATH,
            "data": DATA_PATH,
            "timeout_seconds_per_query": TIMEOUT_S,
            "mean_target": MEAN_TARGET,
            "sum_target": SUM_TARGET,
            "p1_cte_bound": P1_CTE_BOUND,
            "p5_deadzone": P5_DEADZONE,
            "rows": ROWS,
            "cols": COLS,
            "per_row_mean_calibration_subset": "centered (|CTE| < 1 m)",
            "per_row_mean_inequalities_per_query": 2 * ROWS,
            "per_row_mean_aux_vars_per_query": 0,
        },
        "row_mean_calibration": row_mean_calibration,
        "results": results,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
