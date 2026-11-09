"""
Tighter TV sweep on the P3 perturbation queries at epsilon=0.05.

We previously found that Fourier-HF-L1 at tau=7.0 restores 6/20 verifications
(baseline 4/20), while TV at tau in {12, 13, 15} stayed at 4/20. We now test
TV at much tighter values (10, 8, 6, 4, 2) to see whether TV can also restore
verifications at sufficient tightness, or whether Fourier captures a property
TV does not.

Output: verification_p3_tight_tv.json
"""
import json
import time
from datetime import datetime, timezone

import h5py
import numpy as np
from maraboupy import Marabou


NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "verification_p3_tight_tv.json"

NUM_PIXELS = 128
SUM_TARGET = 64.0
PIXEL_MIN, PIXEL_MAX = 0.0, 1.0
P3_CTE_BOUND = 10.0
P3_HE_BOUND = 90.0
P3_CENTERED_THRESHOLD = 1.0
EPSILON = 0.05
TIMEOUT_S = 300
NUM_IMAGES = 20

CTE_OUT_VAR = 192
HE_OUT_VAR = 193

# TV neighbor pairs on 8x16 grid (matching existing scripts)
_PAIRS = []
for i in range(8):
    for j in range(16):
        flat = i * 16 + j
        if j + 1 < 16:
            _PAIRS.append((flat, i * 16 + (j + 1)))
        if i + 1 < 8:
            _PAIRS.append((flat, (i + 1) * 16 + j))
assert len(_PAIRS) == 8 * 15 + 7 * 16 == 232


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        return (f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float64),
                f["X_train"][:].astype(np.float64))


def select_centered_sample(labels):
    cte = labels[:, 0]
    centered_idx = np.where(np.abs(cte) < P3_CENTERED_THRESHOLD)[0]
    return centered_idx[np.linspace(0, len(centered_idx) - 1, NUM_IMAGES, dtype=int)]


def add_tv_bound(net, input_vars, tau):
    """Add total-variation bound: sum_{adj pairs} |x_a - x_b| <= tau."""
    if not np.isfinite(tau):
        return
    d_vars = []
    for a_idx, b_idx in _PAIRS:
        d = int(net.getNewVariable())
        net.setLowerBound(d, 0.0)
        d_vars.append(d)
        a, b = int(input_vars[a_idx]), int(input_vars[b_idx])
        net.addInequality([a, b, d], [1.0, -1.0, -1.0], 0.0)  # a-b-d <= 0
        net.addInequality([b, a, d], [1.0, -1.0, -1.0], 0.0)  # b-a-d <= 0
    net.addInequality(d_vars, [1.0] * len(d_vars), float(tau))


def solve_one(x0, eps, tau_tv, out_var, direc, bound):
    pixel_lo = np.maximum(PIXEL_MIN, x0 - eps)
    pixel_hi = np.minimum(PIXEL_MAX, x0 + eps)
    if pixel_lo.sum() > SUM_TARGET or pixel_hi.sum() < SUM_TARGET:
        return {"status": "region_empty", "time_seconds": 0.0}
    net = Marabou.read_nnet(NNET_PATH)
    ivars = [int(v) for v in net.inputVars[0].flatten()]
    for i in range(NUM_PIXELS):
        net.setLowerBound(ivars[i], float(pixel_lo[i]))
        net.setUpperBound(ivars[i], float(pixel_hi[i]))
    net.addEquality(ivars, [1.0] * NUM_PIXELS, SUM_TARGET)
    add_tv_bound(net, ivars, tau_tv)
    if direc == ">=":
        net.setLowerBound(out_var, float(bound))
    else:
        net.setUpperBound(out_var, float(bound))
    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    status, _, _ = net.solve(verbose=False, options=opts)
    elapsed = time.time() - t0
    return {"status": status.strip().lower(), "time_seconds": round(elapsed, 4)}


def run_p3_image(x0, tau_tv):
    """Run all 4 P3 sub-queries for an image; verified if all 4 UNSAT."""
    sub_queries = [
        ("a", CTE_OUT_VAR, ">=", P3_CTE_BOUND),
        ("b", CTE_OUT_VAR, "<=", -P3_CTE_BOUND),
        ("c", HE_OUT_VAR, ">=", P3_HE_BOUND),
        ("d", HE_OUT_VAR, "<=", -P3_HE_BOUND),
    ]
    results = {}
    verified = True
    for tag, var, direc, bound in sub_queries:
        r = solve_one(x0, EPSILON, tau_tv, var, direc, bound)
        results[tag] = r
        if r["status"] != "unsat":
            verified = False
    return verified, results


def main():
    print("Loading data...", flush=True)
    images, labels = load_data()
    sample = select_centered_sample(labels)
    print(f"  {len(sample)} centered images selected.", flush=True)

    tau_settings = [10.0, 8.0, 6.0, 4.0, 2.0]
    results_by_tau = {}

    for tau in tau_settings:
        key = f"tv_tau_{tau:.0f}".replace(".0", "")
        print(f"\n=== {key} (tau={tau}) ===", flush=True)
        verified = violated = empty = 0
        per_image = {}
        t_alpha = time.time()
        for k, gi in enumerate(sample):
            x0 = images[int(gi)]
            v, sub = run_p3_image(x0, tau)
            empty_flag = any(s["status"] == "region_empty" for s in sub.values())
            if empty_flag:
                empty += 1
                tag = "EMPTY"
            elif v:
                verified += 1
                tag = "VERIFIED"
            else:
                violated += 1
                tag = "VIOLATED"
            per_image[f"image_{k+1}"] = {
                "dataset_index": int(gi),
                "sub_queries": sub,
                "verified": v,
                "region_empty": empty_flag,
            }
            statuses = " ".join(sub[t]["status"][:3] for t, *_ in [("a",), ("b",), ("c",), ("d",)])
            print(f"  img{k+1:>2} idx={gi} {tag:<9} [{statuses}]", flush=True)
        elapsed = time.time() - t_alpha
        results_by_tau[key] = {
            "tau": tau,
            "n_verified": verified,
            "n_violated": violated,
            "n_empty": empty,
            "n_total": NUM_IMAGES,
            "elapsed_seconds": round(elapsed, 2),
            "per_image": per_image,
        }
        print(f"  => {verified}/{NUM_IMAGES} verified, {violated}/{NUM_IMAGES} violated, "
              f"{empty}/{NUM_IMAGES} empty ({elapsed:.1f}s)", flush=True)

    out = {
        "provenance": {
            "script": "verify_p3_tight_tv.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "epsilon_perturbation": EPSILON,
            "timeout_seconds": TIMEOUT_S,
            "num_images": NUM_IMAGES,
        },
        "baseline_reference": "verification_math_regions_p3_eps0p05.json (tv at tau 12/13/15 all gave 4/20)",
        "results_by_tau": results_by_tau,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}", flush=True)
    print("\n=== SUMMARY ===", flush=True)
    for k, v in results_by_tau.items():
        print(f"  {k}: {v['n_verified']}/{v['n_total']} verified", flush=True)


if __name__ == "__main__":
    main()
