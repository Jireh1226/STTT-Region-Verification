"""
P3 robustness verification with the mean(x) = 0.5 preprocessing invariant
and explicit clipping to [0, 1] enforced. This brings P3 to the same
methodological standard as the mean-constrained P1/P4/P5 reruns.

Region:  ||x - x0||_inf <= epsilon,  intersected with  [0, 1]^128,
         intersected with  sum(x_i) = 64.0.

Postcondition (negated for SAT search):
  4 sub-queries:  CTE >= 10, CTE <= -10, HE >= 90, HE <= -90.

For each of the same 20 evenly-sampled centered images used in the
unconstrained P3 run, we run all 4 sub-queries at each epsilon and report
verified vs. violated.
"""
import json
import time

import h5py
import numpy as np
from maraboupy import Marabou


NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "verification_p3_mean_constrained.json"

NUM_PIXELS = 128
SUM_TARGET = 0.5 * NUM_PIXELS
PIXEL_MIN = 0.0
PIXEL_MAX = 1.0

EPSILONS = [0.02, 0.05, 0.10]
NUM_IMAGES = 20
P3_CTE_BOUND = 10.0
P3_HE_BOUND = 90.0
P3_CENTERED_THRESHOLD = 1.0
TIMEOUT_S = 300


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:]
        labels = f["X_train"][:]
    return images, labels


def select_centered_sample(images, labels):
    cte = labels[:, 0]
    centered_idx = np.where(np.abs(cte) < P3_CENTERED_THRESHOLD)[0]
    sample = centered_idx[np.linspace(0, len(centered_idx) - 1, NUM_IMAGES, dtype=int)]
    return sample


def solve_one(x0, epsilon, out_var, direction, bound):
    net = Marabou.read_nnet(NNET_PATH)
    input_vars = net.inputVars[0].flatten()

    # L_inf ball clipped to [0, 1]
    for i in range(NUM_PIXELS):
        lo = max(PIXEL_MIN, float(x0[i] - epsilon))
        hi = min(PIXEL_MAX, float(x0[i] + epsilon))
        net.setLowerBound(int(input_vars[i]), lo)
        net.setUpperBound(int(input_vars[i]), hi)

    # mean(x) = 0.5  i.e.  sum(x) = 64
    net.addEquality(list(input_vars), [1.0] * NUM_PIXELS, SUM_TARGET)

    if direction == ">=":
        net.setLowerBound(out_var, float(bound))
    else:
        net.setUpperBound(out_var, float(bound))

    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    status, vals, _ = net.solve(verbose=False, options=opts)
    elapsed = time.time() - t0
    status = status.strip().lower()

    out = {"status": status, "time_seconds": round(elapsed, 4)}
    if status == "sat":
        cx_input = np.array([vals[int(v)] for v in input_vars])
        out["counterexample"] = {
            "CTE": round(float(vals[192]), 6),
            "HE": round(float(vals[193]), 6),
            "input_mean": round(float(cx_input.mean()), 6),
            "input_min": round(float(cx_input.min()), 6),
            "input_max": round(float(cx_input.max()), 6),
            "linf_from_x0": round(float(np.max(np.abs(cx_input - x0))), 6),
        }
    return out


def feasibility_check(x0, epsilon):
    """Return whether the L_inf ball clipped to [0,1] is compatible with
    sum=64. Necessary condition: lower-sum <= 64 <= upper-sum."""
    lo = np.maximum(PIXEL_MIN, x0 - epsilon)
    hi = np.minimum(PIXEL_MAX, x0 + epsilon)
    return float(lo.sum()), float(hi.sum())


def main():
    images, labels = load_data()
    cte = labels[:, 0]
    sample_idx = select_centered_sample(images, labels)
    print(f"Selected {len(sample_idx)} centered images. ε values: {EPSILONS}")
    print(f"Mean constraint: sum(x) = {SUM_TARGET}, clip to [0, 1]\n")

    sub_queries = [
        ("a", 192, ">=",  P3_CTE_BOUND),
        ("b", 192, "<=", -P3_CTE_BOUND),
        ("c", 193, ">=",  P3_HE_BOUND),
        ("d", 193, "<=", -P3_HE_BOUND),
    ]

    results = {"mean_target": 0.5, "epsilons": EPSILONS, "results_by_epsilon": {}}

    for eps in EPSILONS:
        print(f"=== epsilon = {eps} ===")
        verified = 0
        violated = 0
        infeasible = 0
        per_image = {}

        for k, idx in enumerate(sample_idx):
            x0 = images[int(idx)].flatten().astype(np.float64)
            x0_mean = float(x0.mean())
            lo_sum, hi_sum = feasibility_check(x0, eps)

            # If the clipped ball cannot contain any sum=64 point, the region
            # is empty; record and skip.
            region_feasible = (lo_sum <= SUM_TARGET <= hi_sum)

            entry = {
                "dataset_index": int(idx),
                "x0_mean": round(x0_mean, 6),
                "ball_sum_range": [round(lo_sum, 4), round(hi_sum, 4)],
                "region_feasible": region_feasible,
            }

            if not region_feasible:
                infeasible += 1
                entry["sub_queries"] = "skipped (region empty)"
                entry["verified"] = None
                per_image[f"image_{k+1}"] = entry
                print(f"  img{k+1:>2} idx={idx} mean={x0_mean:.4f}  region empty (sum range "
                      f"[{lo_sum:.2f}, {hi_sum:.2f}] excludes 64.0)")
                continue

            sub_results = {}
            all_unsat = True
            for tag, var, direc, b in sub_queries:
                r = solve_one(x0, eps, var, direc, b)
                sub_results[tag] = r
                if r["status"] != "unsat":
                    all_unsat = False

            entry["sub_queries"] = sub_results
            entry["verified"] = bool(all_unsat)
            per_image[f"image_{k+1}"] = entry

            if all_unsat:
                verified += 1
                tag = "VERIFIED"
            else:
                violated += 1
                tag = "VIOLATED"
            statuses = " ".join(sub_results[t]["status"][:3] for t, *_ in sub_queries)
            print(f"  img{k+1:>2} idx={idx} mean={x0_mean:.4f}  {tag:<8}  [{statuses}]")

        results["results_by_epsilon"][f"epsilon_{eps}"] = {
            "verified": verified,
            "violated": violated,
            "infeasible_region": infeasible,
            "total": NUM_IMAGES,
            "per_image": per_image,
        }
        print(f"  => {verified}/{NUM_IMAGES} verified, "
              f"{violated}/{NUM_IMAGES} violated, "
              f"{infeasible}/{NUM_IMAGES} region empty\n")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
