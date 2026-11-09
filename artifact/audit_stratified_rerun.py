"""
Targeted rerun of the same SAT-producing queries under each of five region
families, with witnesses stored. Lets the stratified audit compare witnesses
on a level playing field.

Queries (negated postcondition, on the centered or off-center subset as appropriate):
  P1a: CTE >= 10  (centered box)
  P1b: CTE <= -10 (centered box)
  P4_left:  CTE <= 0          (left box; expect SAT at CTE=0 or strong magnitude)
  P4_right: CTE >= 0          (right box; expect SAT at CTE=0 or strong magnitude)
  P5_deadzone: |CTE|<=0.01 AND |HE|<=0.01  (off-center box)

Region families:
  (1) raw_pixel_box        no mean constraint
  (2) box_plus_mean        mean(x) = 0.5
  (3) generic_math         box + mean + (TV <= 12  OR  Fourier <= 7  OR  photometric <= 0.20)
  (4) runway_calibrated    box + mean + per-pair Lipschitz at p90
  (5) ae_region            AE_centered decoder + mean (P1a/P1b only because of subset choice;
                           we already have plenty of AE-region witnesses scored elsewhere)

Output: stratified_witnesses_rerun.json
"""
import json, time
from datetime import datetime, timezone
import h5py
import numpy as np
from maraboupy import Marabou


NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "stratified_witnesses_rerun.json"

NUM_PIXELS = 128
SUM_TARGET = 64.0
CTE_OUT_VAR = 192
HE_OUT_VAR = 193
P1_CTE = 10.0
TIMEOUT_S = 120


_PAIRS = []
for i in range(8):
    for j in range(16):
        flat = i * 16 + j
        if j + 1 < 16:
            _PAIRS.append((flat, i * 16 + (j + 1)))
        if i + 1 < 8:
            _PAIRS.append((flat, (i + 1) * 16 + j))


def safe_solve(net, opts):
    try:
        return net.solve(verbose=False, options=opts)
    except Exception as e:
        return f"error: {str(e)[:60]}", {}, None


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        return (f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float64),
                f["X_train"][:].astype(np.float64))


def box_from_subset(images):
    return images.min(axis=0), images.max(axis=0)


def base_net(plo, phi, enforce_mean=False, clip=True):
    net = Marabou.read_nnet(NNET_PATH)
    ivars = [int(v) for v in net.inputVars[0].flatten()]
    for i in range(NUM_PIXELS):
        lo, hi = float(plo[i]), float(phi[i])
        if clip:
            lo, hi = max(0.0, lo), min(1.0, hi)
        net.setLowerBound(ivars[i], lo)
        net.setUpperBound(ivars[i], hi)
    if enforce_mean:
        net.addEquality(ivars, [1.0] * NUM_PIXELS, SUM_TARGET)
    return net, ivars


def add_tv(net, ivars, tau):
    d_vars = []
    for a, b in _PAIRS:
        d = int(net.getNewVariable()); net.setLowerBound(d, 0.0); d_vars.append(d)
        net.addInequality([int(ivars[a]), int(ivars[b]), d], [1.0, -1.0, -1.0], 0.0)
        net.addInequality([int(ivars[b]), int(ivars[a]), d], [1.0, -1.0, -1.0], 0.0)
    net.addInequality(d_vars, [1.0] * len(d_vars), float(tau))


def add_fourier_hf(net, ivars, tau):
    H, W = 8, 16
    K = np.arange(H)[:, None]; L = np.arange(W)[None, :]
    abs_vars = []
    for k in range(H):
        for l in range(W):
            if not (k >= 4 or l >= 8):
                continue
            for trig_name, trig in [("cos", np.cos), ("sin", np.sin)]:
                # Re/Im of c_{k,l} = sum_{i,j} x_{ij} * exp(-2pi i (k*i/H + l*j/W))
                coeffs = []
                vars_ = []
                for i in range(H):
                    for j in range(W):
                        phase = -2.0 * np.pi * (k * i / H + l * j / W)
                        coef = float(trig(phase))
                        coeffs.append(coef)
                        vars_.append(int(ivars[i * W + j]))
                # |sum| <= a   ↔   sum - a <= 0  AND  -sum - a <= 0
                a = int(net.getNewVariable()); net.setLowerBound(a, 0.0); abs_vars.append(a)
                net.addInequality(vars_ + [a], coeffs + [-1.0], 0.0)
                net.addInequality(vars_ + [a], [-c for c in coeffs] + [-1.0], 0.0)
    net.addInequality(abs_vars, [1.0] * len(abs_vars), float(tau))


def add_photometric(net, ivars, eps):
    b1 = np.ones(NUM_PIXELS) / np.sqrt(NUM_PIXELS)
    h = np.tile(np.arange(16) - 7.5, 8).astype(float); h -= h.mean()
    b2 = h / np.linalg.norm(h)
    alpha = int(net.getNewVariable()); net.setLowerBound(alpha, -1e3); net.setUpperBound(alpha, 1e3)
    beta = int(net.getNewVariable()); net.setLowerBound(beta, -1e3); net.setUpperBound(beta, 1e3)
    for i in range(NUM_PIXELS):
        # r_i = x_i - alpha*b1_i - beta*b2_i, constrained |r_i| <= eps
        # x_i - alpha*b1_i - beta*b2_i <= eps
        net.addInequality([int(ivars[i]), alpha, beta],
                          [1.0, -float(b1[i]), -float(b2[i])], float(eps))
        net.addInequality([int(ivars[i]), alpha, beta],
                          [-1.0, float(b1[i]), float(b2[i])], float(eps))


def add_per_pair_lip(net, ivars, taus):
    for (a, b), tau in zip(_PAIRS, taus):
        net.addInequality([int(ivars[a]), int(ivars[b])], [1.0, -1.0], float(tau))
        net.addInequality([int(ivars[b]), int(ivars[a])], [1.0, -1.0], float(tau))


def calibrate_lip_p90(centered):
    return np.array([
        float(np.percentile(np.abs(centered[:, a] - centered[:, b]), 90))
        for a, b in _PAIRS
    ])


def solve_with_witness(net, ivars, subq_var, direc, bnd):
    if direc == ">=":
        net.setLowerBound(subq_var, float(bnd))
    elif direc == "<=":
        net.setUpperBound(subq_var, float(bnd))
    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    status, vals, _ = safe_solve(net, opts)
    elapsed = time.time() - t0
    out = {"status": str(status).strip().lower(), "time_seconds": round(elapsed, 3)}
    if "sat" in out["status"] and vals:
        wit = [float(vals.get(int(ivars[i]), float('nan'))) for i in range(NUM_PIXELS)]
        out["witness"] = wit
        out["CTE"] = round(float(vals.get(CTE_OUT_VAR, float('nan'))), 4)
        out["HE"] = round(float(vals.get(HE_OUT_VAR, float('nan'))), 4)
    return out


def conjunctive_p5(net, ivars):
    """Negated P5: |CTE|<=0.01 AND |HE|<=0.01. We set both as bounds and solve once."""
    net.setLowerBound(CTE_OUT_VAR, -0.01); net.setUpperBound(CTE_OUT_VAR, 0.01)
    net.setLowerBound(HE_OUT_VAR, -0.01); net.setUpperBound(HE_OUT_VAR, 0.01)
    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    status, vals, _ = safe_solve(net, opts)
    elapsed = time.time() - t0
    out = {"status": str(status).strip().lower(), "time_seconds": round(elapsed, 3)}
    if "sat" in out["status"] and vals:
        wit = [float(vals.get(int(ivars[i]), float('nan'))) for i in range(NUM_PIXELS)]
        out["witness"] = wit
        out["CTE"] = round(float(vals.get(CTE_OUT_VAR, float('nan'))), 4)
        out["HE"] = round(float(vals.get(HE_OUT_VAR, float('nan'))), 4)
    return out


def main():
    print("Loading data and subsets...", flush=True)
    images, labels = load_data()
    cte = labels[:, 0]; he = labels[:, 1]
    centered = images[np.abs(cte) < 1.0]
    left = images[cte > 2.0]; right = images[cte < -2.0]
    off_center = images[np.abs(cte) > 1.0]
    print(f"  centered={len(centered)} left={len(left)} right={len(right)} off={len(off_center)}", flush=True)

    cen_lo, cen_hi = box_from_subset(centered)
    left_lo, left_hi = box_from_subset(left)
    right_lo, right_hi = box_from_subset(right)
    off_lo, off_hi = box_from_subset(off_center)

    taus_p90 = calibrate_lip_p90(centered)
    print(f"  per-pair p90: median={np.median(taus_p90):.4f}, max={taus_p90.max():.4f}", flush=True)

    queries = [
        ("P1a", cen_lo,   cen_hi,   CTE_OUT_VAR, ">=",  10.0),
        ("P1b", cen_lo,   cen_hi,   CTE_OUT_VAR, "<=", -10.0),
        ("P4_left",  left_lo,  left_hi,  CTE_OUT_VAR, "<=", 0.0),
        ("P4_right", right_lo, right_hi, CTE_OUT_VAR, ">=", 0.0),
        # P5 is conjunctive; handled separately
    ]

    families = [
        ("raw_pixel_box",          dict(mean=False, math=None)),
        ("box_plus_mean",          dict(mean=True,  math=None)),
        ("generic_math_tv12",      dict(mean=True,  math=("tv", 12.0))),
        ("generic_math_fourier7",  dict(mean=True,  math=("fourier", 7.0))),
        ("generic_math_photo020",  dict(mean=True,  math=("photo", 0.20))),
        ("runway_calibrated_lip90", dict(mean=True, math=("lip", taus_p90))),
    ]

    def construct(plo, phi, cfg):
        net, ivars = base_net(plo, phi, enforce_mean=cfg["mean"])
        if cfg["math"] is None:
            return net, ivars
        kind, param = cfg["math"]
        if kind == "tv":
            add_tv(net, ivars, param)
        elif kind == "fourier":
            add_fourier_hf(net, ivars, param)
        elif kind == "photo":
            add_photometric(net, ivars, param)
        elif kind == "lip":
            add_per_pair_lip(net, ivars, param)
        return net, ivars

    out = {
        "provenance": {
            "script": "audit_stratified_rerun.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timeout_seconds": TIMEOUT_S,
            "per_pair_p90_median_calibration": float(np.median(taus_p90)),
        },
        "by_family": {},
    }

    for fam_name, cfg in families:
        print(f"\n=== {fam_name} ===", flush=True)
        fam_out = {"config": str(cfg), "queries": {}}
        for q_name, plo, phi, var, dirc, bnd in queries:
            try:
                net, ivars = construct(plo, phi, cfg)
            except Exception as e:
                print(f"  {q_name}: setup error: {e}", flush=True)
                fam_out["queries"][q_name] = {"status": f"setup_error: {str(e)[:60]}"}
                continue
            r = solve_with_witness(net, ivars, var, dirc, bnd)
            fam_out["queries"][q_name] = r
            sc = r.get("CTE")
            sct = "" if sc is None else f"  CTE={sc}"
            print(f"  {q_name:10s} {r['status']:8s} t={r['time_seconds']:.2f}s{sct}", flush=True)
        # P5
        try:
            net, ivars = construct(off_lo, off_hi, cfg)
            r = conjunctive_p5(net, ivars)
        except Exception as e:
            r = {"status": f"setup_error: {str(e)[:60]}"}
        fam_out["queries"]["P5_deadzone"] = r
        print(f"  {'P5_deadzone':10s} {r.get('status','?'):8s} t={r.get('time_seconds',0):.2f}s", flush=True)
        out["by_family"][fam_name] = fam_out

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
