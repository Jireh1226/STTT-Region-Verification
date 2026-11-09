"""
P4-left magnitude-floor threshold sweep under each math region.

Marabou returns *a* SAT witness, not the minimum-CTE witness. So a "collapse
from -4.38m to -0.48m under TV<=10" finding is solver-path arbitrary, not a
region fact. To recover the actual magnitude floor, we explicitly ask
`CTE <= -k` for k in {1,2,3,4,5,6,7,8} under each region and report the
largest k that still returns SAT. That is the certified magnitude floor.

Regions tested (all with box + mean=0.5 + clipping):
  - baseline (no math constraint)
  - TV <= 10
  - TV <= 6   (the tightest setting in the failure-hardness curve)
  - Fourier <= 3.34 (orthonormal, matching Section 7.5)
  - Photometric residual eps <= 0.184
  - Per-pair Lipschitz at p_90 (tight runway-calibrated)
  - Per-pair Lipschitz at p_99 (loose runway-calibrated)

Output: verification_p4_left_threshold_sweep.json
"""
import json, time
from datetime import datetime, timezone
import math
import numpy as np
import h5py
from maraboupy import Marabou


NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "verification_p4_left_threshold_sweep.json"

NUM_PIXELS = 128
ROWS, COLS = 8, 16
SUM_TARGET = 64.0
CTE_OUT_VAR = 192
HE_OUT_VAR = 193
TIMEOUT_S = 120
DFT_NORM = 1.0 / math.sqrt(NUM_PIXELS)


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


def add_tv(net, ivars, tau):
    d_vars = []
    for a, b in _PAIRS:
        d = int(net.getNewVariable()); net.setLowerBound(d, 0.0); d_vars.append(d)
        net.addInequality([int(ivars[a]), int(ivars[b]), d], [1.0, -1.0, -1.0], 0.0)
        net.addInequality([int(ivars[b]), int(ivars[a]), d], [1.0, -1.0, -1.0], 0.0)
    net.addInequality(d_vars, [1.0] * len(d_vars), float(tau))


def add_fourier_hf_orthonormal(net, ivars, tau):
    """Section 7.5 convention: orthonormal DFT (1/sqrt(N)) on each coeff."""
    abs_vars = []
    for k in range(ROWS):
        for l in range(COLS):
            if not (k >= ROWS // 2 or l >= COLS // 2):
                continue
            for trig_name, trig in [("cos", math.cos), ("sin", math.sin)]:
                coeffs = []; vars_ = []
                for i in range(ROWS):
                    for j in range(COLS):
                        phase = 2.0 * math.pi * (k * i / ROWS + l * j / COLS)
                        coef = DFT_NORM * trig(phase) * (1.0 if trig_name == "cos" else -1.0)
                        coeffs.append(coef)
                        vars_.append(int(ivars[i * COLS + j]))
                a = int(net.getNewVariable()); net.setLowerBound(a, 0.0); abs_vars.append(a)
                net.addInequality(vars_ + [a], coeffs + [-1.0], 0.0)
                net.addInequality(vars_ + [a], [-c for c in coeffs] + [-1.0], 0.0)
    net.addInequality(abs_vars, [1.0] * len(abs_vars), float(tau))


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
    """Ask: does there exist x in (left_box + mean + region) with CTE <= -k?"""
    net, ivars = base_query_left(left_lo, left_hi)
    if region_apply is not None:
        try:
            region_apply(net, ivars)
        except Exception as e:
            return {"status": f"region_setup_error: {str(e)[:60]}", "k": -k}
    net.setUpperBound(CTE_OUT_VAR, -float(k))
    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    status, vals, _ = safe_solve(net, opts)
    elapsed = time.time() - t0
    out = {"k": -k, "status": str(status).strip().lower(), "time_seconds": round(elapsed, 2)}
    if "sat" in out["status"] and vals:
        out["CTE"] = round(float(vals.get(CTE_OUT_VAR, float('nan'))), 4)
        out["HE"] = round(float(vals.get(HE_OUT_VAR, float('nan'))), 4)
    return out


def magnitude_floor(left_lo, left_hi, region_apply, region_name):
    """Run threshold sweep; report the largest k where CTE <= -k is SAT.
    A robust answer to 'what's the strongest P4-left wrong-sign magnitude
    achievable inside this region.'"""
    print(f"\n=== {region_name} ===", flush=True)
    results = {}
    largest_sat_k = 0
    for k in [1, 2, 3, 4, 5, 6, 7, 8]:
        r = solve_p4_left_at_threshold(left_lo, left_hi, region_apply, k)
        results[f"k_{k}"] = r
        sat_token = "SAT" if "sat" in r["status"] else "UNSAT" if r["status"] == "unsat" else r["status"]
        cte_s = f" CTE={r.get('CTE','-')}" if "CTE" in r else ""
        print(f"  CTE<=-{k}: {sat_token:6s} t={r['time_seconds']}s{cte_s}", flush=True)
        if "sat" in r["status"]:
            largest_sat_k = k
        elif r["status"] == "unsat":
            # Region cannot reach -k; magnitude floor is between k-1 and k
            break
        # On error/timeout: don't update largest_sat_k, but keep going
    print(f"  --> magnitude floor: CTE <= -{largest_sat_k} achievable; CTE <= -{largest_sat_k+1} not",
          flush=True)
    return {"largest_sat_k": -largest_sat_k, "per_k": results}


def main():
    print("Loading data...", flush=True)
    images, labels = load_data()
    cte = labels[:, 0]
    centered = images[np.abs(cte) < 1.0]
    left = images[cte > 2.0]
    left_lo, left_hi = left.min(axis=0), left.max(axis=0)
    print(f"  left subset n={len(left)}, centered for calibration n={len(centered)}", flush=True)

    taus_p90 = calibrate_lip(centered, 90)
    taus_p99 = calibrate_lip(centered, 99)

    regions = [
        ("baseline_box_plus_mean",   None),
        ("tv_le_10",                 lambda n, i: add_tv(n, i, 10.0)),
        ("tv_le_6",                  lambda n, i: add_tv(n, i, 6.0)),
        ("fourier_le_3p34_orthonormal", lambda n, i: add_fourier_hf_orthonormal(n, i, 3.34)),
        ("photometric_le_0p184",     lambda n, i: add_photometric(n, i, 0.184)),
        ("per_pair_lip_p90",         lambda n, i: add_per_pair_lip(n, i, taus_p90)),
        ("per_pair_lip_p99",         lambda n, i: add_per_pair_lip(n, i, taus_p99)),
    ]

    out = {
        "provenance": {
            "script": "verify_p4_left_threshold_sweep.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timeout_seconds": TIMEOUT_S,
            "purpose": "Magnitude floor for P4-left wrong-sign witness under each math region",
        },
        "by_region": {},
    }
    for name, apply_fn in regions:
        out["by_region"][name] = magnitude_floor(left_lo, left_hi, apply_fn, name)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}", flush=True)

    print("\n=== SUMMARY: P4-left magnitude floor by region ===", flush=True)
    for name, _ in regions:
        f = out["by_region"][name]["largest_sat_k"]
        print(f"  {name:38s}: CTE <= {f:+d} achievable", flush=True)


if __name__ == "__main__":
    main()
