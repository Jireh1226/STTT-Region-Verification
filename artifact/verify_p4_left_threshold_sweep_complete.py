"""
Extend the P4-left threshold sweep to k=20 for the regions that were only
tested to k=8 in the original run (photometric, per-pair Lipschitz at p99/p90).
This makes the sweep grid consistent across all reported math regions.

Output: verification_p4_left_threshold_sweep_complete.json
"""
import json, time, math
from datetime import datetime, timezone
import numpy as np
import h5py
from maraboupy import Marabou


NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "verification_p4_left_threshold_sweep_complete.json"

NUM_PIXELS = 128
ROWS, COLS = 8, 16
SUM_TARGET = 64.0
CTE_OUT_VAR = 192
TIMEOUT_S = 120


_PAIRS = []
for i in range(ROWS):
    for j in range(COLS):
        if j + 1 < COLS:
            _PAIRS.append((i * COLS + j, i * COLS + j + 1))
        if i + 1 < ROWS:
            _PAIRS.append((i * COLS + j, (i + 1) * COLS + j))


def safe_solve(net, opts):
    try:
        return net.solve(verbose=False, options=opts)
    except Exception as e:
        return f"error: {str(e)[:80]}", {}, None


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        return (f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float64),
                f["X_train"][:].astype(np.float64))


def base_query_left(left_lo, left_hi):
    net = Marabou.read_nnet(NNET_PATH)
    ivars = [int(v) for v in net.inputVars[0].flatten()]
    for i in range(NUM_PIXELS):
        net.setLowerBound(ivars[i], float(max(0.0, left_lo[i])))
        net.setUpperBound(ivars[i], float(min(1.0, left_hi[i])))
    net.addEquality(ivars, [1.0] * NUM_PIXELS, SUM_TARGET)
    return net, ivars


def add_photometric(net, ivars, eps):
    b1 = np.ones(NUM_PIXELS) / math.sqrt(NUM_PIXELS)
    h = np.tile(np.arange(16) - 7.5, 8).astype(float); h -= h.mean()
    b2 = h / np.linalg.norm(h)
    alpha = int(net.getNewVariable()); net.setLowerBound(alpha, -1e3); net.setUpperBound(alpha, 1e3)
    beta = int(net.getNewVariable()); net.setLowerBound(beta, -1e3); net.setUpperBound(beta, 1e3)
    for i in range(NUM_PIXELS):
        net.addInequality([int(ivars[i]), alpha, beta],
                          [1.0, -float(b1[i]), -float(b2[i])], float(eps))
        net.addInequality([int(ivars[i]), alpha, beta],
                          [-1.0, float(b1[i]), float(b2[i])], float(eps))


def add_per_pair_lip(net, ivars, taus):
    for (a, b), tau in zip(_PAIRS, taus):
        net.addInequality([int(ivars[a]), int(ivars[b])], [1.0, -1.0], float(tau))
        net.addInequality([int(ivars[b]), int(ivars[a])], [1.0, -1.0], float(tau))


def calibrate_lip(centered, q):
    return np.array([
        float(np.percentile(np.abs(centered[:, a] - centered[:, b]), q))
        for a, b in _PAIRS
    ])


def solve_p4_left_at_threshold(left_lo, left_hi, region_apply, k):
    net, ivars = base_query_left(left_lo, left_hi)
    region_apply(net, ivars)
    net.setUpperBound(CTE_OUT_VAR, -float(k))
    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    status, vals, _ = safe_solve(net, opts)
    out = {"k": -k, "status": str(status).strip().lower(),
           "time_seconds": round(time.time() - t0, 2)}
    if "sat" in out["status"] and vals:
        out["CTE"] = round(float(vals.get(CTE_OUT_VAR, float('nan'))), 4)
    return out


def main():
    images, labels = load_data()
    cte = labels[:, 0]
    left = images[cte > 2.0]
    centered = images[np.abs(cte) < 1.0]
    left_lo, left_hi = left.min(axis=0), left.max(axis=0)

    taus_p90 = calibrate_lip(centered, 90)
    taus_p99 = calibrate_lip(centered, 99)

    # Only the regions where the original/extended sweep didn't reach -20
    regions = [
        ("photometric_le_0p184", lambda n, i: add_photometric(n, i, 0.184)),
        ("per_pair_lip_p99",     lambda n, i: add_per_pair_lip(n, i, taus_p99)),
        ("per_pair_lip_p90",     lambda n, i: add_per_pair_lip(n, i, taus_p90)),
    ]

    ks = [10, 12, 15, 18, 20]

    out = {
        "provenance": {
            "script": "verify_p4_left_threshold_sweep_complete.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timeout_seconds": TIMEOUT_S,
            "purpose": "Extend threshold sweep to -20 for regions not previously pushed there",
        },
        "by_region": {},
    }

    for name, apply_fn in regions:
        print(f"\n=== {name} ===", flush=True)
        per_k = {}
        largest_sat = 0
        first_unsat = None
        for k in ks:
            r = solve_p4_left_at_threshold(left_lo, left_hi, apply_fn, k)
            per_k[f"k_{k}"] = r
            tok = "SAT" if "sat" in r["status"] else "UNSAT" if r["status"] == "unsat" else r["status"]
            cte_s = f" CTE={r.get('CTE','-')}" if "CTE" in r else ""
            print(f"  CTE<=-{k}: {tok:7s} t={r['time_seconds']}s{cte_s}", flush=True)
            if "sat" in r["status"]:
                largest_sat = k
            elif r["status"] == "unsat":
                first_unsat = k
                break
        out["by_region"][name] = {"largest_sat_k": -largest_sat,
                                  "first_unsat_k": -first_unsat if first_unsat else None,
                                  "per_k": per_k}

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}", flush=True)

    print("\n=== SUMMARY ===", flush=True)
    for name, _ in regions:
        b = out["by_region"][name]
        unsat = f"UNSAT@{b['first_unsat_k']}" if b['first_unsat_k'] else "no UNSAT"
        print(f"  {name:24s}: largest SAT CTE<={b['largest_sat_k']}, {unsat}", flush=True)


if __name__ == "__main__":
    main()
