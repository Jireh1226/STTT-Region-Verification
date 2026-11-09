"""
Total-variation (TV)-bounded verification for P1/P4/P5 on TinyTaxiNet.

Hypothesis being tested
-----------------------
Real runway images are spatially smooth, so their L1-gradient (TV) is small.
Calibration on the centered subset gives TV in [6.03, 12.65] (p99 ~ 12.2);
full dataset goes up to ~15.4; worst-case 0/1 checkerboard on the same 8x16
grid has TV ~ 232. Adding a TV upper bound tau to the verification region
should forbid the solver from picking non-smooth witnesses. If at sufficiently
tight tau some pixel-box SAT verdicts flip to UNSAT, that signals the SAT
witness was a non-smooth artifact of the box rather than a physically
realistic counterexample.

Properties (under per-pixel-box + mean(x)=0.5 + TV(x) <= tau):
  P1a:     CTE >= 10  on centered per-pixel box (|CTE| < 1m)
  P1b:     CTE <= -10 on centered per-pixel box
  P4_left: CTE <= 0   on left per-pixel box (CTE > 2m, "wrong-sign" query)
  P4_right:CTE >= 0   on right per-pixel box (CTE < -2m, "wrong-sign" query)
  P5:      |CTE|<=0.01 AND |HE|<=0.01 on off-center per-pixel box (|CTE|>1m)

TV sweep: tau in {inf, 50, 15, 13, 10}.

TV encoding (Marabou doesn't have abs; standard slack encoding):
  For each pair (a, b), add d >= 0 with d >= a - b and d >= b - a, then
  sum(d) <= tau. With 8x16 grid that is 8*15 + 7*16 = 232 pairs,
  so 232 aux vars and 465 linear inequalities per query.

Per-query timeout: 300 s. Writes JSON to verification_tv_bound.json.

Usage:
  ./prophecy_env/bin/python verify_p1_p4_p5_with_tv_bound.py
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import subprocess
import time
from typing import Iterable

import h5py
import numpy as np
from maraboupy import Marabou


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "verification_tv_bound.json"

NUM_PIXELS = 128
ROWS, COLS = 8, 16
MEAN_TARGET = 0.5
SUM_TARGET = MEAN_TARGET * NUM_PIXELS  # 64.0

CTE_OUT_VAR = 192
HE_OUT_VAR = 193

P1_CTE_BOUND = 10.0
P5_DEADZONE = 0.01

TIMEOUT_S = 300

# tau = math.inf encodes "no TV constraint" (baseline reproduction)
TAU_SWEEP = [math.inf, 50.0, 15.0, 13.0, 10.0]


# -----------------------------------------------------------------------------
# Data + TV calibration
# -----------------------------------------------------------------------------
def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].astype(np.float64)
        labels = f["X_train"][:].astype(np.float64)
    return images, labels


def compute_box(images: np.ndarray):
    flat = images.reshape(images.shape[0], -1)
    return flat.min(axis=0), flat.max(axis=0)


def tv_of_image(flat128: np.ndarray) -> float:
    """TV(x) = sum_{i, j<COLS-1} |x[i,j] - x[i,j+1]|
            + sum_{i<ROWS-1, j} |x[i,j] - x[i+1,j]|"""
    img = np.asarray(flat128, dtype=np.float64).reshape(ROWS, COLS)
    horiz = np.abs(img[:, :-1] - img[:, 1:]).sum()
    vert = np.abs(img[:-1, :] - img[1:, :]).sum()
    return float(horiz + vert)


def tv_quantiles(images_flat: np.ndarray) -> dict:
    tvs = np.array([tv_of_image(x) for x in images_flat], dtype=np.float64)
    return {
        "n": int(len(tvs)),
        "min": round(float(tvs.min()), 6),
        "p01": round(float(np.percentile(tvs, 1)), 6),
        "p50": round(float(np.percentile(tvs, 50)), 6),
        "p90": round(float(np.percentile(tvs, 90)), 6),
        "p99": round(float(np.percentile(tvs, 99)), 6),
        "max": round(float(tvs.max()), 6),
        "mean": round(float(tvs.mean()), 6),
        "std": round(float(tvs.std()), 6),
    }


# -----------------------------------------------------------------------------
# TV encoding into a Marabou query
# -----------------------------------------------------------------------------
def _neighbor_pairs() -> list[tuple[int, int]]:
    """All horizontal and vertical neighbor flat-index pairs on the 8x16 grid.
    Pixel (i,j) lives at flat index i*COLS + j."""
    pairs: list[tuple[int, int]] = []
    for i in range(ROWS):
        for j in range(COLS - 1):
            pairs.append((i * COLS + j, i * COLS + j + 1))
    for i in range(ROWS - 1):
        for j in range(COLS):
            pairs.append((i * COLS + j, (i + 1) * COLS + j))
    return pairs


_NEIGHBOR_PAIRS = _neighbor_pairs()
assert len(_NEIGHBOR_PAIRS) == ROWS * (COLS - 1) + (ROWS - 1) * COLS == 232


def add_tv_bound(net, input_vars: list[int], tau: float) -> None:
    """For each neighbor pair (a, b) introduce d >= 0 with d >= |x[a] - x[b]|
    and add sum(d) <= tau."""
    if not math.isfinite(tau):
        return  # tau = inf -> no constraint added

    d_vars: list[int] = []
    for (a_idx, b_idx) in _NEIGHBOR_PAIRS:
        a = int(input_vars[a_idx])
        b = int(input_vars[b_idx])
        d = net.getNewVariable()
        net.setLowerBound(d, 0.0)
        # d >= a - b  <=>  a - b - d <= 0
        net.addInequality([a, b, d], [1.0, -1.0, -1.0], 0.0)
        # d >= b - a  <=>  b - a - d <= 0
        net.addInequality([b, a, d], [1.0, -1.0, -1.0], 0.0)
        d_vars.append(d)

    # sum(d_pair) <= tau
    net.addInequality(d_vars, [1.0] * len(d_vars), float(tau))


# -----------------------------------------------------------------------------
# Query builder + runner
# -----------------------------------------------------------------------------
def build_query(pixel_mins: np.ndarray, pixel_maxs: np.ndarray, tau: float):
    """Build a fresh Marabou network with per-pixel box + mean=0.5 + TV<=tau.
    Returns (net, input_vars)."""
    net = Marabou.read_nnet(NNET_PATH)
    input_vars = [int(v) for v in net.inputVars[0].flatten()]
    assert len(input_vars) == NUM_PIXELS
    for i in range(NUM_PIXELS):
        net.setLowerBound(input_vars[i], float(pixel_mins[i]))
        net.setUpperBound(input_vars[i], float(pixel_maxs[i]))
    net.addEquality(input_vars, [1.0] * NUM_PIXELS, SUM_TARGET)
    add_tv_bound(net, input_vars, tau)
    return net, input_vars


def _record_sat_witness(vals, input_vars: list[int]) -> dict:
    cx = np.array([vals[v] for v in input_vars], dtype=np.float64)
    return {
        "CTE": round(float(vals[CTE_OUT_VAR]), 6),
        "HE": round(float(vals[HE_OUT_VAR]), 6),
        "witness_mean": round(float(cx.mean()), 6),
        "witness_TV": round(tv_of_image(cx), 6),
        "witness": [round(float(v), 6) for v in cx.tolist()],
    }


def run_single_output_bound(name: str,
                            pixel_mins: np.ndarray,
                            pixel_maxs: np.ndarray,
                            tau: float,
                            out_var: int,
                            direction: str,
                            bound: float) -> dict:
    """For P1a/P1b/P4_left/P4_right: a single output bound constraint."""
    net, input_vars = build_query(pixel_mins, pixel_maxs, tau)
    if direction == ">=":
        net.setLowerBound(out_var, float(bound))
    else:
        net.setUpperBound(out_var, float(bound))
    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    try:
        status, vals, _ = net.solve(verbose=False, options=opts)
    except Exception as e:  # pragma: no cover -- defensive
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
    }
    if status == "sat":
        out.update(_record_sat_witness(vals, input_vars))
    return out


def run_p5(pixel_mins: np.ndarray,
           pixel_maxs: np.ndarray,
           tau: float,
           deadzone: float = P5_DEADZONE) -> dict:
    """|CTE| <= deadzone AND |HE| <= deadzone on off-center box."""
    net, input_vars = build_query(pixel_mins, pixel_maxs, tau)
    net.setLowerBound(CTE_OUT_VAR, -deadzone)
    net.setUpperBound(CTE_OUT_VAR, deadzone)
    net.setLowerBound(HE_OUT_VAR, -deadzone)
    net.setUpperBound(HE_OUT_VAR, deadzone)
    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    try:
        status, vals, _ = net.solve(verbose=False, options=opts)
    except Exception as e:  # pragma: no cover
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
# Main sweep
# -----------------------------------------------------------------------------
def tau_key(tau: float) -> str:
    return "tau_inf" if not math.isfinite(tau) else f"tau_{int(tau)}"


def run_tau(pixel_boxes: dict, tau: float) -> dict:
    """Run all five properties at a single tau."""
    results: dict = {}
    pmin, pmax = pixel_boxes["P1"]
    results["P1a"] = run_single_output_bound(
        "P1a", pmin, pmax, tau, CTE_OUT_VAR, ">=", P1_CTE_BOUND
    )
    results["P1b"] = run_single_output_bound(
        "P1b", pmin, pmax, tau, CTE_OUT_VAR, "<=", -P1_CTE_BOUND
    )
    pmin, pmax = pixel_boxes["P4_left"]
    results["P4_left"] = run_single_output_bound(
        "P4_left", pmin, pmax, tau, CTE_OUT_VAR, "<=", 0.0
    )
    pmin, pmax = pixel_boxes["P4_right"]
    results["P4_right"] = run_single_output_bound(
        "P4_right", pmin, pmax, tau, CTE_OUT_VAR, ">=", 0.0
    )
    pmin, pmax = pixel_boxes["P5"]
    results["P5"] = run_p5(pmin, pmax, tau, deadzone=P5_DEADZONE)
    return results


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

    print("Computing TV calibration quantiles...", flush=True)
    tv_calibration = {
        "centered_quantiles": tv_quantiles(centered),
        "full_dataset_quantiles": tv_quantiles(flat_images),
        "neighbor_pair_count": len(_NEIGHBOR_PAIRS),
        "worst_case_01_checkerboard_TV_upper": float(len(_NEIGHBOR_PAIRS)),
    }
    print(
        f"  centered TV [min={tv_calibration['centered_quantiles']['min']}, "
        f"p99={tv_calibration['centered_quantiles']['p99']}, "
        f"max={tv_calibration['centered_quantiles']['max']}]",
        flush=True,
    )
    print(
        f"  full TV     [min={tv_calibration['full_dataset_quantiles']['min']}, "
        f"p99={tv_calibration['full_dataset_quantiles']['p99']}, "
        f"max={tv_calibration['full_dataset_quantiles']['max']}]",
        flush=True,
    )

    results_by_tau: dict = {}
    for tau in TAU_SWEEP:
        key = tau_key(tau)
        print(f"\n=== Running sweep at {key} (tau={tau}) ===", flush=True)
        t0 = time.time()
        results_by_tau[key] = run_tau(pixel_boxes, tau)
        for prop_name, r in results_by_tau[key].items():
            extra = ""
            if r["status"] == "sat":
                extra = (
                    f"  CTE={r['CTE']:+.4f}  HE={r['HE']:+.4f}  "
                    f"mean={r['witness_mean']:.4f}  TV={r['witness_TV']:.4f}"
                )
            print(
                f"  {prop_name:<8} {r['status'].upper():<7} "
                f"t={r['time_seconds']:>7.2f}s{extra}",
                flush=True,
            )
        print(f"  tau wall: {time.time()-t0:.2f}s", flush=True)

    output = {
        "provenance": {
            "script": "verify_p1_p4_p5_with_tv_bound.py",
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
            "tau_sweep": [
                "inf" if not math.isfinite(t) else t for t in TAU_SWEEP
            ],
            "tv_encoding_pairs": len(_NEIGHBOR_PAIRS),
            "tv_encoding_aux_vars_per_query": len(_NEIGHBOR_PAIRS),
            "tv_encoding_inequalities_per_query": 2 * len(_NEIGHBOR_PAIRS) + 1,
            "rows": ROWS,
            "cols": COLS,
        },
        "tv_calibration": tv_calibration,
        "results": results_by_tau,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
