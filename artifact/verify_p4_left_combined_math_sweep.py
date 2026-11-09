"""
P4-left threshold sweep under the COMBINED math constraint
(TV <= 12 AND Fourier <= 7.0 AND photometric <= 0.20, all moderate),
to replace the unsupported speculation in §7.5 with data.

Output: verification_p4_left_combined_math_sweep.json
"""
import json, time, math
from datetime import datetime, timezone
import numpy as np
import h5py
from maraboupy import Marabou


NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "verification_p4_left_combined_math_sweep.json"

NUM_PIXELS = 128
ROWS, COLS = 8, 16
SUM_TARGET = 64.0
CTE_OUT_VAR = 192
TIMEOUT_S = 180
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
    abs_vars = []
    for k in range(ROWS):
        for l in range(COLS):
            if not (k >= ROWS // 2 or l >= COLS // 2):
                continue
            for trig_name in ["cos", "sin"]:
                trig = math.cos if trig_name == "cos" else math.sin
                sgn = 1.0 if trig_name == "cos" else -1.0
                coeffs = []; vars_ = []
                for i in range(ROWS):
                    for j in range(COLS):
                        phase = 2.0 * math.pi * (k * i / ROWS + l * j / COLS)
                        coeffs.append(DFT_NORM * trig(phase) * sgn)
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


def add_combined(net, ivars):
    """TV <= 12, Fourier <= 7.0 (orthonormal), photometric <= 0.20 simultaneously."""
    add_tv(net, ivars, 12.0)
    add_fourier_hf_orthonormal(net, ivars, 7.0)
    add_photometric(net, ivars, 0.20)


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
    left_lo, left_hi = left.min(axis=0), left.max(axis=0)
    print(f"left subset n={len(left)}, timeout {TIMEOUT_S}s", flush=True)

    ks = [1, 2, 4, 8, 10, 12, 15, 18, 20]

    out = {
        "provenance": {
            "script": "verify_p4_left_combined_math_sweep.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timeout_seconds": TIMEOUT_S,
            "constraint": "TV<=12 AND Fourier<=7.0 (orthonormal) AND photometric<=0.20",
        },
        "per_k": {},
    }

    print("\n=== combined (TV<=12, Fourier<=7.0, photo<=0.20) ===", flush=True)
    largest_sat = 0
    first_unsat = None
    for k in ks:
        r = solve_p4_left_at_threshold(left_lo, left_hi, add_combined, k)
        out["per_k"][f"k_{k}"] = r
        tok = "SAT" if "sat" in r["status"] else "UNSAT" if r["status"] == "unsat" else r["status"]
        cte_s = f" CTE={r.get('CTE','-')}" if "CTE" in r else ""
        print(f"  CTE<=-{k}: {tok:7s} t={r['time_seconds']}s{cte_s}", flush=True)
        if "sat" in r["status"]:
            largest_sat = k
        elif r["status"] == "unsat":
            first_unsat = k
            break

    out["summary"] = {"largest_sat_k": -largest_sat,
                      "first_unsat_k": -first_unsat if first_unsat else None}

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}", flush=True)
    print(f"\nLargest SAT CTE<={-largest_sat}; first UNSAT {'CTE<='+str(-first_unsat) if first_unsat else 'not found'}",
          flush=True)


if __name__ == "__main__":
    main()
