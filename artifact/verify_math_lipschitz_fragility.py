"""
Per-pair Lipschitz and P3 fragility reruns for the JAIS submission.

The script covers per-pair Lipschitz on P2, P3, and P6, plus a larger
P3 fragility sample, with try/except around every Marabou solve to handle
MalformedBasisException.

Output: verification_math_lipschitz_fragility.json
"""
import json, time
from datetime import datetime, timezone
import h5py
import numpy as np
from maraboupy import Marabou


NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "verification_math_lipschitz_fragility.json"

NUM_PIXELS = 128
SUM_TARGET = 64.0
PIXEL_MIN, PIXEL_MAX = 0.0, 1.0
CTE_OUT_VAR = 192
HE_OUT_VAR = 193
P1_CTE = 10.0
P1_HE = 90.0
P3_EPS = 0.05
TIMEOUT_S = 60


_PAIRS = []
for i in range(8):
    for j in range(16):
        flat = i * 16 + j
        if j + 1 < 16:
            _PAIRS.append((flat, i * 16 + (j + 1)))
        if i + 1 < 8:
            _PAIRS.append((flat, (i + 1) * 16 + j))


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        return (f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float64),
                f["X_train"][:].astype(np.float64))


def base_query(pixel_lo, pixel_hi):
    if np.any(pixel_lo > pixel_hi):
        return None, None
    if pixel_lo.sum() > SUM_TARGET + 1e-9 or pixel_hi.sum() < SUM_TARGET - 1e-9:
        return None, None
    net = Marabou.read_nnet(NNET_PATH)
    ivars = [int(v) for v in net.inputVars[0].flatten()]
    for i in range(NUM_PIXELS):
        net.setLowerBound(ivars[i], float(pixel_lo[i]))
        net.setUpperBound(ivars[i], float(pixel_hi[i]))
    net.addEquality(ivars, [1.0]*NUM_PIXELS, SUM_TARGET)
    return net, ivars


def add_tv(net, ivars, tau):
    d_vars = []
    for a, b in _PAIRS:
        d = int(net.getNewVariable()); net.setLowerBound(d, 0.0); d_vars.append(d)
        net.addInequality([int(ivars[a]), int(ivars[b]), d], [1.0, -1.0, -1.0], 0.0)
        net.addInequality([int(ivars[b]), int(ivars[a]), d], [1.0, -1.0, -1.0], 0.0)
    net.addInequality(d_vars, [1.0]*len(d_vars), float(tau))


def add_per_pair_lip(net, ivars, taus):
    for (a, b), tau in zip(_PAIRS, taus):
        net.addInequality([int(ivars[a]), int(ivars[b])], [1.0, -1.0], float(tau))
        net.addInequality([int(ivars[b]), int(ivars[a])], [1.0, -1.0], float(tau))


def safe_solve(net, opts):
    try:
        return net.solve(verbose=False, options=opts)
    except Exception as e:
        msg = str(e)[:80]
        return f"error: {msg}", {}, None


def run_subqueries(pixel_lo, pixel_hi, region_apply, subqueries):
    """Returns (verified, empty, per-query)."""
    v = True; em = False; per = []
    for var, direc, bnd in subqueries:
        net, ivars = base_query(pixel_lo, pixel_hi)
        if net is None:
            per.append({"status": "region_empty"})
            em = True; v = False; continue
        try:
            region_apply(net, ivars)
        except Exception as e:
            per.append({"status": f"region_setup_error: {str(e)[:60]}"})
            v = False; continue
        if direc == ">=": net.setLowerBound(var, float(bnd))
        else: net.setUpperBound(var, float(bnd))
        opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
        t0 = time.time()
        status, vals, _ = safe_solve(net, opts)
        elapsed = time.time() - t0
        out = {"status": str(status).strip().lower(), "time_seconds": round(elapsed, 3)}
        if "sat" == out["status"] and vals:
            out["CTE"] = round(float(vals.get(CTE_OUT_VAR, float('nan'))), 4)
            out["HE"] = round(float(vals.get(HE_OUT_VAR, float('nan'))), 4)
        per.append(out)
        if out["status"] != "unsat":
            v = False
        if "error" in out["status"]:
            v = False
    return v, em, per


def select_centered_sample(labels, n):
    cte = labels[:, 0]
    centered_idx = np.where(np.abs(cte) < 1.0)[0]
    return centered_idx[np.linspace(0, len(centered_idx)-1, n, dtype=int)]


def main():
    print("Loading data...", flush=True)
    images, labels = load_data()
    cte = labels[:, 0]; he = labels[:, 1]
    centered = images[np.abs(cte) < 1.0]
    print(f"  centered={len(centered)}", flush=True)

    # Per-pair Lipschitz calibration
    tau_p99 = []; tau_p90 = []
    for a, b in _PAIRS:
        diffs = np.abs(centered[:, a] - centered[:, b])
        tau_p99.append(float(np.percentile(diffs, 99)))
        tau_p90.append(float(np.percentile(diffs, 90)))

    out = {
        "provenance": {
            "script": "verify_math_lipschitz_fragility.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timeout_seconds": TIMEOUT_S,
        }
    }

    # === EXP B: per-pair Lipschitz on P3 and P2 ===
    print("\n[EXP B] per-pair Lipschitz on P3 (eps=0.05) and P2 (eps=0.02)", flush=True)
    sample = select_centered_sample(labels, 20)
    p3_sub = [
        (CTE_OUT_VAR, ">=", P1_CTE),
        (CTE_OUT_VAR, "<=", -P1_CTE),
        (HE_OUT_VAR, ">=", P1_HE),
        (HE_OUT_VAR, "<=", -P1_HE),
    ]
    out["B_p3"] = {}
    for label, taus in [("p99", tau_p99), ("p90", tau_p90)]:
        verified = empty = errors = 0
        for k, gi in enumerate(sample):
            x0 = images[int(gi)]
            plo = np.maximum(PIXEL_MIN, x0 - P3_EPS)
            phi = np.minimum(PIXEL_MAX, x0 + P3_EPS)
            v, em, per = run_subqueries(plo, phi, lambda n, i, t=taus: add_per_pair_lip(n, i, t), p3_sub)
            if em: empty += 1
            if any("error" in str(p.get("status", "")) for p in per): errors += 1
            if v: verified += 1
        out["B_p3"][f"lip_{label}"] = {"n_verified": verified, "n_empty": empty,
                                       "n_errors": errors, "n_total": len(sample)}
        print(f"  P3 lip_{label}: {verified}/{len(sample)} verified (empty={empty}, err={errors})", flush=True)

    print("\n[EXP B] per-pair Lipschitz on P2 (eps=0.02, 1m correctness)", flush=True)
    out["B_p2"] = {}
    for label, taus in [("p99", tau_p99), ("p90", tau_p90)]:
        verified = nom_wrong = empty = errors = 0
        for k, gi in enumerate(sample):
            x0 = images[int(gi)]
            cte0 = cte[int(gi)]; he0 = he[int(gi)]
            net0 = Marabou.read_nnet(NNET_PATH)
            outv = net0.evaluate([x0])
            cpred = float(np.array(outv).flatten()[0])
            hpred = float(np.array(outv).flatten()[1])
            if abs(cpred - cte0) >= 1.0 or abs(hpred - he0) >= 5.0:
                nom_wrong += 1; continue
            plo = np.maximum(PIXEL_MIN, x0 - 0.02)
            phi = np.minimum(PIXEL_MAX, x0 + 0.02)
            p2_sub = [
                (CTE_OUT_VAR, ">=", float(cte0 + 1.0)),
                (CTE_OUT_VAR, "<=", float(cte0 - 1.0)),
                (HE_OUT_VAR, ">=", float(he0 + 5.0)),
                (HE_OUT_VAR, "<=", float(he0 - 5.0)),
            ]
            v, em, per = run_subqueries(plo, phi, lambda n, i, t=taus: add_per_pair_lip(n, i, t), p2_sub)
            if em: empty += 1
            if any("error" in str(p.get("status", "")) for p in per): errors += 1
            if v: verified += 1
        out["B_p2"][f"lip_{label}"] = {
            "n_verified": verified, "n_nominally_wrong": nom_wrong,
            "n_empty": empty, "n_errors": errors, "n_total": len(sample),
        }
        print(f"  P2 lip_{label}: {verified}/{len(sample)} verified (nom_wrong={nom_wrong}, err={errors})",
              flush=True)

    # === EXP C: Larger P3 fragility (50 anchors to keep wall time bounded) ===
    n_anchors = 50
    print(f"\n[EXP C] Larger P3 fragility on {n_anchors} anchors, eps=0.05", flush=True)
    sample_big = select_centered_sample(labels, n_anchors)
    taus_C = [12, 10, 8, 6, 4]
    out["C_p3_fragility"] = {"n_anchors": n_anchors, "taus": taus_C, "results_by_tau": {}}
    verified_by_anchor_tau = {}
    for tau in taus_C:
        verified = empty = errors = 0
        t0 = time.time()
        for k, gi in enumerate(sample_big):
            x0 = images[int(gi)]
            plo = np.maximum(PIXEL_MIN, x0 - P3_EPS)
            phi = np.minimum(PIXEL_MAX, x0 + P3_EPS)
            v, em, per = run_subqueries(plo, phi, lambda n, i, t=tau: add_tv(n, i, t), p3_sub)
            verified_by_anchor_tau[(int(gi), tau)] = v
            if em: empty += 1
            if any("error" in str(p.get("status", "")) for p in per): errors += 1
            if v: verified += 1
        elapsed = time.time() - t0
        out["C_p3_fragility"]["results_by_tau"][f"tau_{tau}"] = {
            "n_verified": verified, "n_empty": empty, "n_errors": errors,
            "wall_s": round(elapsed, 1), "n_total": n_anchors,
        }
        print(f"  tau={tau}: {verified}/{n_anchors} verified ({elapsed:.1f}s, err={errors})", flush=True)

    # Threshold per anchor
    thresholds = {}
    for gi in sample_big:
        gi = int(gi); thr = 0
        for tau in taus_C:
            if verified_by_anchor_tau.get((gi, tau)):
                if tau > thr: thr = tau
        thresholds[str(gi)] = thr
    from collections import Counter
    dist = Counter(thresholds.values())
    out["C_p3_fragility"]["threshold_distribution"] = dict(sorted(dist.items(), key=lambda x: -x[0]))
    print(f"  Threshold distribution: {dict(sorted(dist.items(), key=lambda x: -x[0]))}", flush=True)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
