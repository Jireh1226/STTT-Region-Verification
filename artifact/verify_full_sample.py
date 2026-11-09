"""
Repeat the two cheapest-to-medium P2/P3 experiments on the full centered subset
of 893 images. The full-sample rates can then be compared against the 20-image
baseline rates reported elsewhere in the artifact.

What this runs:
  1. P2 at eps=0.02 on all 893 centered images (|CTE| < 1 m), per-image,
     4 sub-queries each.
  2. P3 mean+clip on all 893 centered images at eps in {0.02, 0.05, 0.10},
     4 sub-queries each.

What this does NOT run (cost): per-image certified-radius binary search on all
893, label-conditioned box on all 893, local-PCA on all 893 across three alphas.
Those would be 3+ hours of solver time each. This script makes the per-epsilon
verification rates population-level rather than 20-image.

Per-query timeout is 60 seconds (smaller than the 300 s used for the harder
follow-up experiments) because P2/P3 baseline queries are sub-second in
practice; total wall-clock runtime is ~3 minutes for all ~8,000 sub-queries.
"""
import json
import time
import sys

import h5py
import numpy as np
from maraboupy import Marabou


NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "verification_full_sample_centered.json"

NUM_PIXELS = 128
SUM_TARGET = 0.5 * NUM_PIXELS
PIXEL_MIN, PIXEL_MAX = 0.0, 1.0

P2_CTE_BOUND = 1.0
P2_HE_BOUND = 5.0
P3_CTE_BOUND = 10.0
P3_HE_BOUND = 90.0
TIMEOUT_S = 60   # P2/P3 queries are sub-second normally; 60s is generous


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        return f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float64), \
               f["X_train"][:].astype(np.float64)


def nominal_predict(x):
    net = Marabou.read_nnet(NNET_PATH)
    out = net.evaluate([x])
    flat = np.array(out).flatten()
    return float(flat[0]), float(flat[1])


def solve_one(x0, eps, out_var, direc, bound, enforce_mean_clip):
    net = Marabou.read_nnet(NNET_PATH)
    ivars = [int(v) for v in net.inputVars[0].flatten()]
    for i in range(NUM_PIXELS):
        if enforce_mean_clip:
            lo = max(PIXEL_MIN, float(x0[i] - eps))
            hi = min(PIXEL_MAX, float(x0[i] + eps))
        else:
            lo = float(x0[i] - eps)
            hi = float(x0[i] + eps)
        net.setLowerBound(ivars[i], lo)
        net.setUpperBound(ivars[i], hi)
    if enforce_mean_clip:
        net.addEquality(ivars, [1.0] * NUM_PIXELS, SUM_TARGET)
    if direc == ">=":
        net.setLowerBound(out_var, float(bound))
    else:
        net.setUpperBound(out_var, float(bound))
    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    status, _, _ = net.solve(verbose=False, options=opts)
    return status.strip().lower()


def run_p2_full(images, labels):
    """P2: for each centered image, check if all inputs in eps=0.02 L_inf ball
    around x0 produce CTE within 1m and HE within 5deg of the ground-truth label.
    """
    cte = labels[:, 0]
    he = labels[:, 1]
    centered_idx = np.where(np.abs(cte) < 1.0)[0]
    n = len(centered_idx)
    print(f"P2 (eps=0.02, full centered N={n})", flush=True)

    nominally_wrong = nominally_correct_fragile = verified_robust = 0
    t_start = time.time()
    for i_idx, gi in enumerate(centered_idx):
        x0 = images[int(gi)]
        cte_pred, he_pred = nominal_predict(x0)
        cte_err = abs(cte_pred - cte[int(gi)])
        he_err = abs(he_pred - he[int(gi)])
        nominally_correct = (cte_err < P2_CTE_BOUND) and (he_err < P2_HE_BOUND)
        if not nominally_correct:
            nominally_wrong += 1
            continue
        all_unsat = True
        sub_queries = [
            (192, ">=", float(cte[int(gi)] + P2_CTE_BOUND)),
            (192, "<=", float(cte[int(gi)] - P2_CTE_BOUND)),
            (193, ">=", float(he[int(gi)] + P2_HE_BOUND)),
            (193, "<=", float(he[int(gi)] - P2_HE_BOUND)),
        ]
        for var, direc, bound in sub_queries:
            status = solve_one(x0, 0.02, var, direc, bound,
                               enforce_mean_clip=False)
            if status != "unsat":
                all_unsat = False
                break
        if all_unsat:
            verified_robust += 1
        else:
            nominally_correct_fragile += 1
        if (i_idx + 1) % 50 == 0:
            el = time.time() - t_start
            rate = (i_idx + 1) / el
            eta = (n - i_idx - 1) / rate
            print(f"  P2 progress {i_idx+1}/{n}  wrong={nominally_wrong} "
                  f"fragile={nominally_correct_fragile} robust={verified_robust} "
                  f"({rate:.1f} img/s, ETA {eta:.0f}s)", flush=True)
    elapsed = time.time() - t_start
    print(f"P2 done in {elapsed:.1f}s: wrong={nominally_wrong} "
          f"fragile={nominally_correct_fragile} robust={verified_robust} / N={n}",
          flush=True)
    return {
        "epsilon": 0.02,
        "n_total": int(n),
        "nominally_wrong": int(nominally_wrong),
        "nominally_correct_fragile": int(nominally_correct_fragile),
        "verified_robust": int(verified_robust),
        "elapsed_seconds": round(elapsed, 2),
    }


def run_p3_meanclip_full(images, labels):
    """P3 mean+clip at three eps: for each centered image, check all 4 sub-queries
    UNSAT inside L_inf ball clipped to [0,1] with mean(x)=0.5 enforced."""
    cte = labels[:, 0]
    centered_idx = np.where(np.abs(cte) < 1.0)[0]
    n = len(centered_idx)
    out = {}
    sub_queries = [
        (192, ">=",  P3_CTE_BOUND),
        (192, "<=", -P3_CTE_BOUND),
        (193, ">=",  P3_HE_BOUND),
        (193, "<=", -P3_HE_BOUND),
    ]
    for eps in [0.02, 0.05, 0.10]:
        print(f"P3 mean+clip eps={eps} full centered N={n}", flush=True)
        t_start = time.time()
        verified = violated = empty = 0
        for i_idx, gi in enumerate(centered_idx):
            x0 = images[int(gi)]
            lo = np.maximum(PIXEL_MIN, x0 - eps)
            hi = np.minimum(PIXEL_MAX, x0 + eps)
            if not (lo.sum() <= SUM_TARGET <= hi.sum()):
                empty += 1
                continue
            all_unsat = True
            for var, direc, bound in sub_queries:
                status = solve_one(x0, eps, var, direc, bound,
                                   enforce_mean_clip=True)
                if status != "unsat":
                    all_unsat = False
                    break
            if all_unsat:
                verified += 1
            else:
                violated += 1
            if (i_idx + 1) % 50 == 0:
                el = time.time() - t_start
                rate = (i_idx + 1) / el
                eta = (n - i_idx - 1) / rate
                print(f"  eps={eps} progress {i_idx+1}/{n}  v={verified} "
                      f"x={violated} ({rate:.1f}/s ETA {eta:.0f}s)", flush=True)
        elapsed = time.time() - t_start
        print(f"P3 eps={eps} done in {elapsed:.1f}s: verified={verified} "
              f"violated={violated} empty={empty} / N={n}", flush=True)
        out[f"epsilon_{eps}"] = {
            "epsilon": eps,
            "n_total": int(n),
            "verified": int(verified),
            "violated": int(violated),
            "empty_region": int(empty),
            "elapsed_seconds": round(elapsed, 2),
        }
    return out


def main():
    print("Loading data...", flush=True)
    images, labels = load_data()
    print(f"Loaded {len(images)} images.", flush=True)

    results = {"description": "Full-sample (893 centered images) P2 and P3 expansion."}
    t0 = time.time()
    results["P2"] = run_p2_full(images, labels)
    results["P3_mean_clip"] = run_p3_meanclip_full(images, labels)
    results["total_elapsed_seconds"] = round(time.time() - t0, 2)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
