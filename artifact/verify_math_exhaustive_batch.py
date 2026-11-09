"""
Exhaustive batch of math/physics-region experiments to map the remaining
methodology surface:

  1. Per-image TV calibration on P3 (each anchor's own TV +/- a fraction)
  2. TV/Fourier on P3 at epsilon=0.10 (baseline 0/893)
  3. Math regions on P2 (correctness, centered subset)
  4. Math regions on P6 axis-aligned bounds
  5. Math regions on full-centered-subset P3 mean+clip at epsilon=0.05
     (compare per-image TV-restoration rate vs the global TV-restoration rate)

Saves the union of results to verification_math_exhaustive.json.
"""
import json
import time
import math
from datetime import datetime, timezone

import h5py
import numpy as np
from maraboupy import Marabou


NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "verification_math_exhaustive.json"

NUM_PIXELS = 128
SUM_TARGET = 64.0
PIXEL_MIN, PIXEL_MAX = 0.0, 1.0
CTE_OUT_VAR = 192
HE_OUT_VAR = 193

P3_CTE_BOUND = 10.0
P3_HE_BOUND = 90.0
P2_CTE_BAND = 1.0   # |CTE - CTE*| < 1m
P2_HE_BAND = 5.0    # |HE - HE*| < 5deg
P3_CENTERED_THRESHOLD = 1.0

TIMEOUT_S = 120
NUM_IMAGES = 20

# TV neighbor pairs
_TV_PAIRS = []
for i in range(8):
    for j in range(16):
        flat = i * 16 + j
        if j + 1 < 16:
            _TV_PAIRS.append((flat, i * 16 + (j + 1)))
        if i + 1 < 8:
            _TV_PAIRS.append((flat, (i + 1) * 16 + j))


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        return (f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float64),
                f["X_train"][:].astype(np.float64))


def tv(img_flat):
    img = img_flat.reshape(8, 16)
    return float(np.abs(np.diff(img, axis=1)).sum() + np.abs(np.diff(img, axis=0)).sum())


def add_tv(net, ivars, tau):
    if not np.isfinite(tau):
        return
    d_vars = []
    for a, b in _TV_PAIRS:
        d = int(net.getNewVariable())
        net.setLowerBound(d, 0.0)
        d_vars.append(d)
        net.addInequality([int(ivars[a]), int(ivars[b]), d], [1.0, -1.0, -1.0], 0.0)
        net.addInequality([int(ivars[b]), int(ivars[a]), d], [1.0, -1.0, -1.0], 0.0)
    net.addInequality(d_vars, [1.0]*len(d_vars), float(tau))


def select_centered_sample(labels):
    cte = labels[:, 0]
    centered_idx = np.where(np.abs(cte) < P3_CENTERED_THRESHOLD)[0]
    return centered_idx[np.linspace(0, len(centered_idx)-1, NUM_IMAGES, dtype=int)]


def per_pixel_box(images_subset):
    """[min, max] per-pixel over a subset."""
    return images_subset.min(axis=0), images_subset.max(axis=0)


def solve_query(pixel_lo, pixel_hi, tau_tv, post_setter):
    """post_setter(net) configures the negated postcondition."""
    if np.any(pixel_lo > pixel_hi):
        return {"status": "region_empty", "time_seconds": 0.0}
    if pixel_lo.sum() > SUM_TARGET + 1e-9 or pixel_hi.sum() < SUM_TARGET - 1e-9:
        return {"status": "region_empty", "time_seconds": 0.0}
    net = Marabou.read_nnet(NNET_PATH)
    ivars = [int(v) for v in net.inputVars[0].flatten()]
    for i in range(NUM_PIXELS):
        net.setLowerBound(ivars[i], float(pixel_lo[i]))
        net.setUpperBound(ivars[i], float(pixel_hi[i]))
    net.addEquality(ivars, [1.0]*NUM_PIXELS, SUM_TARGET)
    add_tv(net, ivars, tau_tv)
    post_setter(net)
    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    status, vals, _ = net.solve(verbose=False, options=opts)
    elapsed = time.time() - t0
    out = {"status": status.strip().lower(), "time_seconds": round(elapsed, 3)}
    if status.strip().lower() == "sat":
        out["CTE"] = round(float(vals[CTE_OUT_VAR]), 4)
        out["HE"] = round(float(vals[HE_OUT_VAR]), 4)
    return out


# -----------------------------------------------------------------------------
# Experiment 1: Per-image TV calibration on P3
# Each anchor x_0 has its own TV; use tau = TV(x_0) + delta as the bound.
# -----------------------------------------------------------------------------
def exp1_per_image_tv_p3(images, sample, eps=0.05, deltas=(0, 1, 2, 4)):
    print(f"\n[EXP1] Per-image TV calibration on P3, eps={eps}", flush=True)
    sub_queries = [
        (CTE_OUT_VAR, ">=", P3_CTE_BOUND),
        (CTE_OUT_VAR, "<=", -P3_CTE_BOUND),
        (HE_OUT_VAR, ">=", P3_HE_BOUND),
        (HE_OUT_VAR, "<=", -P3_HE_BOUND),
    ]
    result = {"epsilon": eps, "deltas": list(deltas), "per_image": {}}
    for delta in deltas:
        result[f"delta_{delta}"] = {"n_verified": 0, "n_violated": 0, "n_empty": 0}
    for k, gi in enumerate(sample):
        x0 = images[int(gi)]
        tv0 = tv(x0)
        result["per_image"][f"image_{k+1}"] = {
            "dataset_index": int(gi), "anchor_tv": round(tv0, 4),
            "deltas": {},
        }
        for delta in deltas:
            tau = tv0 + delta
            pixel_lo = np.maximum(PIXEL_MIN, x0 - eps)
            pixel_hi = np.minimum(PIXEL_MAX, x0 + eps)
            verdicts = []
            verified = True
            empty = False
            for var, direc, bnd in sub_queries:
                def setter(net, var=var, direc=direc, bnd=bnd):
                    if direc == ">=":
                        net.setLowerBound(var, float(bnd))
                    else:
                        net.setUpperBound(var, float(bnd))
                r = solve_query(pixel_lo, pixel_hi, tau, setter)
                verdicts.append(r["status"])
                if r["status"] == "region_empty":
                    empty = True
                if r["status"] != "unsat":
                    verified = False
            result["per_image"][f"image_{k+1}"]["deltas"][f"delta_{delta}"] = {
                "tau": round(tau, 4), "verified": verified, "verdicts": verdicts,
            }
            if empty:
                result[f"delta_{delta}"]["n_empty"] += 1
            elif verified:
                result[f"delta_{delta}"]["n_verified"] += 1
            else:
                result[f"delta_{delta}"]["n_violated"] += 1
        if (k+1) % 5 == 0:
            print(f"  ...{k+1}/{len(sample)} done", flush=True)
    print(f"  Per-image TV results:")
    for delta in deltas:
        d = result[f"delta_{delta}"]
        print(f"    delta=+{delta}: {d['n_verified']}/20 verified, {d['n_empty']} empty")
    return result


# -----------------------------------------------------------------------------
# Experiment 2: TV on P3 at epsilon = 0.10 (baseline 0/893)
# -----------------------------------------------------------------------------
def exp2_tv_p3_eps10(images, sample, taus=(15, 10, 8, 6, 4, 2)):
    print(f"\n[EXP2] TV on P3 at eps=0.10", flush=True)
    eps = 0.10
    sub_queries = [
        (CTE_OUT_VAR, ">=", P3_CTE_BOUND),
        (CTE_OUT_VAR, "<=", -P3_CTE_BOUND),
        (HE_OUT_VAR, ">=", P3_HE_BOUND),
        (HE_OUT_VAR, "<=", -P3_HE_BOUND),
    ]
    result = {"epsilon": eps, "results_by_tau": {}}
    for tau in taus:
        verified = violated = empty = 0
        for k, gi in enumerate(sample):
            x0 = images[int(gi)]
            pixel_lo = np.maximum(PIXEL_MIN, x0 - eps)
            pixel_hi = np.minimum(PIXEL_MAX, x0 + eps)
            v = True
            em = False
            for var, direc, bnd in sub_queries:
                def setter(net, var=var, direc=direc, bnd=bnd):
                    if direc == ">=":
                        net.setLowerBound(var, float(bnd))
                    else:
                        net.setUpperBound(var, float(bnd))
                r = solve_query(pixel_lo, pixel_hi, tau, setter)
                if r["status"] == "region_empty":
                    em = True
                if r["status"] != "unsat":
                    v = False
            if em:
                empty += 1
            elif v:
                verified += 1
            else:
                violated += 1
        result["results_by_tau"][f"tau_{tau}"] = {
            "tau": tau, "n_verified": verified, "n_violated": violated, "n_empty": empty,
        }
        print(f"  tau={tau}: {verified}/20 verified ({empty} empty)", flush=True)
    return result


# -----------------------------------------------------------------------------
# Experiment 3: Math regions on P2 (centered subset, correctness, eps=0.02)
# Baseline: 0/20 verified at 1.0m CTE band.
# Use TV bound on the perturbed image.
# -----------------------------------------------------------------------------
def exp3_math_on_p2(images, labels, sample, taus=(15, 10, 8, 6, 4)):
    print(f"\n[EXP3] TV on P2 at eps=0.02, 1.0m CTE band", flush=True)
    cte = labels[:, 0]; he = labels[:, 1]
    eps = 0.02
    result = {"epsilon": eps, "cte_band": 1.0, "he_band": 5.0, "results_by_tau": {}}
    for tau in taus:
        verified = violated = empty = nominally_wrong = 0
        for k, gi in enumerate(sample):
            x0 = images[int(gi)]
            cte0 = cte[int(gi)]; he0 = he[int(gi)]
            # Check nominal
            net = Marabou.read_nnet(NNET_PATH)
            out = net.evaluate([x0])
            cte_pred = float(np.array(out).flatten()[0])
            he_pred = float(np.array(out).flatten()[1])
            if abs(cte_pred - cte0) >= P2_CTE_BAND or abs(he_pred - he0) >= P2_HE_BAND:
                nominally_wrong += 1
                continue
            pixel_lo = np.maximum(PIXEL_MIN, x0 - eps)
            pixel_hi = np.minimum(PIXEL_MAX, x0 + eps)
            # 4 sub-queries: CTE pred - CTE* > 1, < -1; HE pred - HE* > 5, < -5
            sub_queries = [
                (CTE_OUT_VAR, ">=", float(cte0 + P2_CTE_BAND)),
                (CTE_OUT_VAR, "<=", float(cte0 - P2_CTE_BAND)),
                (HE_OUT_VAR, ">=", float(he0 + P2_HE_BAND)),
                (HE_OUT_VAR, "<=", float(he0 - P2_HE_BAND)),
            ]
            v = True
            em = False
            for var, direc, bnd in sub_queries:
                def setter(net, var=var, direc=direc, bnd=bnd):
                    if direc == ">=":
                        net.setLowerBound(var, float(bnd))
                    else:
                        net.setUpperBound(var, float(bnd))
                r = solve_query(pixel_lo, pixel_hi, tau, setter)
                if r["status"] == "region_empty":
                    em = True
                if r["status"] != "unsat":
                    v = False
            if em:
                empty += 1
            elif v:
                verified += 1
            else:
                violated += 1
        result["results_by_tau"][f"tau_{tau}"] = {
            "tau": tau,
            "n_verified": verified,
            "n_violated": violated,
            "n_empty": empty,
            "n_nominally_wrong": nominally_wrong,
        }
        print(f"  tau={tau}: {verified}/20 verified ({nominally_wrong} nominally wrong, {empty} empty)", flush=True)
    return result


# -----------------------------------------------------------------------------
# Experiment 4: Math regions on P6 axis-aligned bounds (centered subset)
# Without math: y0 in [-24.33, 23.30], y1 in [-90.10, 84.07]
# Compare bounds under TV tau=10
# -----------------------------------------------------------------------------
def exp4_math_on_p6(images, labels, taus=(15, 10, 8)):
    print(f"\n[EXP4] TV on P6 axis-aligned bounds (centered subset)", flush=True)
    cte = labels[:, 0]
    centered_mask = np.abs(cte) < P3_CENTERED_THRESHOLD
    centered = images[centered_mask]
    pixel_lo, pixel_hi = per_pixel_box(centered)
    result = {"centered_n": int(len(centered)), "results_by_tau": {}}

    # For each tau, do a coarse binary search on each axis to find the reachable range
    def find_bound(direc, var, lo, hi, tau, depth=4):
        """Returns the certified bound: largest M such that var >= M is SAT (for direc='lower'),
        or smallest M such that var <= M is SAT (for direc='upper'). Coarse: tolerance 1."""
        # Binary search: we know var is in [-100, 100].
        # find_bound for direc='upper' finds the max reachable value: largest M with var >= M reachable (SAT).
        if direc == 'upper':
            target_lo, target_hi = lo, hi  # search range
            best = target_lo
            for _ in range(depth):
                mid = 0.5*(target_lo + target_hi)
                def setter(net, var=var, mid=mid):
                    net.setLowerBound(var, float(mid))
                r = solve_query(pixel_lo, pixel_hi, tau, setter)
                if r["status"] == "sat":
                    best = mid
                    target_lo = mid
                else:
                    target_hi = mid
            return best
        else:
            target_lo, target_hi = lo, hi
            best = target_hi
            for _ in range(depth):
                mid = 0.5*(target_lo + target_hi)
                def setter(net, var=var, mid=mid):
                    net.setUpperBound(var, float(mid))
                r = solve_query(pixel_lo, pixel_hi, tau, setter)
                if r["status"] == "sat":
                    best = mid
                    target_hi = mid
                else:
                    target_lo = mid
            return best

    for tau in taus:
        # Approximate bounds with coarse binary search
        cte_upper = find_bound('upper', CTE_OUT_VAR, 0, 50, tau)
        cte_lower = find_bound('lower', CTE_OUT_VAR, -50, 0, tau)
        he_upper  = find_bound('upper', HE_OUT_VAR, 0, 100, tau)
        he_lower  = find_bound('lower', HE_OUT_VAR, -100, 0, tau)
        result["results_by_tau"][f"tau_{tau}"] = {
            "tau": tau,
            "cte_bounds_approx": [round(cte_lower, 2), round(cte_upper, 2)],
            "he_bounds_approx":  [round(he_lower, 2), round(he_upper, 2)],
        }
        print(f"  tau={tau}: CTE ~[{cte_lower:.1f}, {cte_upper:.1f}], HE ~[{he_lower:.1f}, {he_upper:.1f}]", flush=True)
    return result


def main():
    print("Loading data...", flush=True)
    images, labels = load_data()
    sample = select_centered_sample(labels)
    print(f"  N={len(images)}, sample={len(sample)} centered anchors", flush=True)

    out = {
        "provenance": {
            "script": "verify_math_exhaustive_batch.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timeout_seconds": TIMEOUT_S,
        },
        "exp1_per_image_tv_p3": None,
        "exp2_tv_p3_eps0p10": None,
        "exp3_math_on_p2": None,
        "exp4_math_on_p6": None,
    }

    out["exp1_per_image_tv_p3"] = exp1_per_image_tv_p3(images, sample)
    out["exp2_tv_p3_eps0p10"]    = exp2_tv_p3_eps10(images, sample)
    out["exp3_math_on_p2"]       = exp3_math_on_p2(images, labels, sample)
    out["exp4_math_on_p6"]       = exp4_math_on_p6(images, labels)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
