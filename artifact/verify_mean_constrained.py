"""
Probe whether key counterexamples survive the mean=0.5 preprocessing invariant.

Experiments:
  - P1-global with the additional linear equality sum(x_i) = 64.0
  - P4 directional correctness with the additional linear equality sum(x_i) = 64.0
  - P5 deadzone with the additional linear equality sum(x_i) = 64.0
  - P5 threshold sweep under the same mean constraint

Usage:
  ./prophecy_env/bin/python verify_mean_constrained.py
"""

import json
import time

import h5py
import numpy as np
from maraboupy import Marabou


NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "verification_mean_constrained.json"

MEAN_TARGET = 0.5
NUM_PIXELS = 128
SUM_TARGET = MEAN_TARGET * NUM_PIXELS

P1_CTE_BOUND = 10.0
P1_HE_BOUND = 90.0
P5_DEADZONE = 0.01
P5_THRESHOLDS = [0.01, 0.1, 0.5, 1.0]


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:]
        labels = f["X_train"][:]
    return images, labels


def compute_box(images):
    flat = images.reshape(images.shape[0], -1)
    return flat.min(axis=0), flat.max(axis=0)


def add_mean_constraint(net, input_vars):
    net.addEquality(list(input_vars), [1.0] * len(input_vars), SUM_TARGET)


def solve_p1_mean(centered_images):
    pixel_mins, pixel_maxs = compute_box(centered_images)
    options = Marabou.createOptions(verbosity=0)

    sub_queries = [
        ("P1a_mean", "CTE >= 10", 192, ">=", P1_CTE_BOUND),
        ("P1b_mean", "CTE <= -10", 192, "<=", -P1_CTE_BOUND),
        ("P1c_mean", "HE >= 90", 193, ">=", P1_HE_BOUND),
        ("P1d_mean", "HE <= -90", 193, "<=", -P1_HE_BOUND),
    ]

    results = {}
    for name, desc, out_var, direction, bound in sub_queries:
        net = Marabou.read_nnet(NNET_PATH)
        input_vars = net.inputVars[0].flatten()
        for i in range(NUM_PIXELS):
            net.setLowerBound(input_vars[i], float(pixel_mins[i]))
            net.setUpperBound(input_vars[i], float(pixel_maxs[i]))
        add_mean_constraint(net, input_vars)

        if direction == ">=":
            net.setLowerBound(out_var, bound)
        else:
            net.setUpperBound(out_var, bound)

        t0 = time.time()
        status, vals, _ = net.solve(verbose=False, options=options)
        elapsed = time.time() - t0
        status = status.strip().lower()

        result = {
            "description": desc,
            "status": status,
            "time_seconds": round(elapsed, 4),
        }
        if status == "sat":
            cx = np.array([vals[v] for v in input_vars], dtype=np.float64)
            result["counterexample"] = {
                "CTE": round(float(vals[192]), 6),
                "HE": round(float(vals[193]), 6),
                "input_mean": round(float(cx.mean()), 6),
            }
        results[name] = result
    return results


def solve_p5_mean(off_center_images, threshold=P5_DEADZONE):
    pixel_mins, pixel_maxs = compute_box(off_center_images)
    options = Marabou.createOptions(verbosity=0)

    net = Marabou.read_nnet(NNET_PATH)
    input_vars = net.inputVars[0].flatten()
    for i in range(NUM_PIXELS):
        net.setLowerBound(input_vars[i], float(pixel_mins[i]))
        net.setUpperBound(input_vars[i], float(pixel_maxs[i]))
    add_mean_constraint(net, input_vars)

    net.setLowerBound(192, -threshold)
    net.setUpperBound(192, threshold)
    net.setLowerBound(193, -threshold)
    net.setUpperBound(193, threshold)

    t0 = time.time()
    status, vals, _ = net.solve(verbose=False, options=options)
    elapsed = time.time() - t0
    status = status.strip().lower()

    result = {
        "threshold": threshold,
        "status": status,
        "time_seconds": round(elapsed, 4),
    }
    if status == "sat":
        cx = np.array([vals[v] for v in input_vars], dtype=np.float64)
        result["counterexample"] = {
            "CTE": round(float(vals[192]), 6),
            "HE": round(float(vals[193]), 6),
            "input_mean": round(float(cx.mean()), 6),
        }
    return result


def solve_p4_mean(left_images, right_images):
    options = Marabou.createOptions(verbosity=0)
    results = {}

    configs = [
        ("C_left_mean", left_images, "<="),
        ("C_right_mean", right_images, ">="),
    ]

    for name, region_images, direction in configs:
        pixel_mins, pixel_maxs = compute_box(region_images)
        net = Marabou.read_nnet(NNET_PATH)
        input_vars = net.inputVars[0].flatten()
        for i in range(NUM_PIXELS):
            net.setLowerBound(input_vars[i], float(pixel_mins[i]))
            net.setUpperBound(input_vars[i], float(pixel_maxs[i]))
        add_mean_constraint(net, input_vars)

        if direction == "<=":
            net.setUpperBound(192, 0.0)
        else:
            net.setLowerBound(192, 0.0)

        t0 = time.time()
        status, vals, _ = net.solve(verbose=False, options=options)
        elapsed = time.time() - t0
        status = status.strip().lower()

        result = {
            "status": status,
            "time_seconds": round(elapsed, 4),
        }
        if status == "sat":
            cx = np.array([vals[v] for v in input_vars], dtype=np.float64)
            result["counterexample"] = {
                "CTE": round(float(vals[192]), 6),
                "HE": round(float(vals[193]), 6),
                "input_mean": round(float(cx.mean()), 6),
            }
        results[name] = result
    return results


def solve_p5_threshold_sweep(off_center_images):
    results = {}
    for threshold in P5_THRESHOLDS:
        key = f"threshold_{threshold}"
        results[key] = solve_p5_mean(off_center_images, threshold=threshold)
    return results


if __name__ == "__main__":
    images, labels = load_data()
    cte = labels[:, 0]

    centered_images = images[np.abs(cte) < 1.0]
    left_images = images[cte > 2.0]
    right_images = images[cte < -2.0]
    off_center_images = images[np.abs(cte) > 1.0]

    p1_results = solve_p1_mean(centered_images)
    p4_results = solve_p4_mean(left_images, right_images)
    p5_result = solve_p5_mean(off_center_images, threshold=P5_DEADZONE)
    p5_sweep = solve_p5_threshold_sweep(off_center_images)

    results = {
        "mean_target": MEAN_TARGET,
        "sum_target": SUM_TARGET,
        "P1_global_mean_constrained": p1_results,
        "P4_mean_constrained": p4_results,
        "P5_mean_constrained": p5_result,
        "P5_threshold_sweep_mean_constrained": p5_sweep,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("P1-global with mean=0.5 constraint:")
    for key in ["P1a_mean", "P1b_mean", "P1c_mean", "P1d_mean"]:
        entry = p1_results[key]
        print(f"  {key}: {entry['status'].upper()} ({entry['description']})")
    print("\nP4 with mean=0.5 constraint:")
    for key in ["C_left_mean", "C_right_mean"]:
        entry = p4_results[key]
        print(f"  {key}: {entry['status'].upper()}")
    print("\nP5 with mean=0.5 constraint:")
    print(f"  status: {p5_result['status'].upper()}")
    if p5_result["status"] == "sat":
        print(f"  counterexample: {p5_result['counterexample']}")
    print("\nP5 threshold sweep with mean=0.5 constraint:")
    for key in [f"threshold_{t}" for t in P5_THRESHOLDS]:
        entry = p5_sweep[key]
        print(f"  {key}: {entry['status'].upper()}")

    print(f"\nSaved {OUTPUT_JSON}")
