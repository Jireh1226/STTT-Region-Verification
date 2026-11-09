"""
Math-region sensitivity experiments for the JAIS submission:

  A. Quantile sensitivity sweep (TV, Fourier, photometric at p90/p95/p99/max)
     on P1a, P1b, P4_left, P4_right, P5.
  B. Per-pair Lipschitz on P2, P3 (eps=0.05), and P6 axis-aligned bounds.
  C. Larger P3 fragility sweep on 100 centered anchors instead of 20.

Output: verification_math_sensitivity.json
"""
import json, time
from datetime import datetime, timezone
import h5py
import numpy as np
from maraboupy import Marabou


NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "verification_math_sensitivity.json"

NUM_PIXELS = 128
SUM_TARGET = 64.0
PIXEL_MIN, PIXEL_MAX = 0.0, 1.0
CTE_OUT_VAR = 192
HE_OUT_VAR = 193
P1_CTE = 10.0
P1_HE = 90.0
P5_DZ = 0.01
P3_EPS = 0.05
TIMEOUT_S = 120


_PAIRS = []
for i in range(8):
    for j in range(16):
        flat = i * 16 + j
        if j + 1 < 16:
            _PAIRS.append((flat, i * 16 + (j + 1)))
        if i + 1 < 8:
            _PAIRS.append((flat, (i + 1) * 16 + j))

# Fourier high-freq cells
HF_CELLS = [(k, l) for k in range(8) for l in range(16) if (k >= 4 or l >= 8)]


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        return (f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float64),
                f["X_train"][:].astype(np.float64))


def per_pixel_box(imgs):
    return imgs.min(axis=0), imgs.max(axis=0)


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
    if not np.isfinite(tau):
        return
    d_vars = []
    for a, b in _PAIRS:
        d = int(net.getNewVariable())
        net.setLowerBound(d, 0.0)
        d_vars.append(d)
        net.addInequality([int(ivars[a]), int(ivars[b]), d], [1.0, -1.0, -1.0], 0.0)
        net.addInequality([int(ivars[b]), int(ivars[a]), d], [1.0, -1.0, -1.0], 0.0)
    net.addInequality(d_vars, [1.0]*len(d_vars), float(tau))


def add_fourier(net, ivars, tau):
    if not np.isfinite(tau):
        return
    d_list = []
    for k, l in HF_CELLS:
        # cos and sin coefficients for (k, l) over 8x16 with 1/sqrt(128) normalisation
        coeffs_re = np.zeros(128); coeffs_im = np.zeros(128)
        for i in range(8):
            for j in range(16):
                ph = -2*np.pi*(k*i/8 + l*j/16)
                coeffs_re[i*16+j] = np.cos(ph) / np.sqrt(128.0)
                coeffs_im[i*16+j] = np.sin(ph) / np.sqrt(128.0)
        cre = int(net.getNewVariable())
        cim = int(net.getNewVariable())
        net.addEquality([int(ivars[i]) for i in range(NUM_PIXELS)] + [cre],
                        list(coeffs_re) + [-1.0], 0.0)
        net.addEquality([int(ivars[i]) for i in range(NUM_PIXELS)] + [cim],
                        list(coeffs_im) + [-1.0], 0.0)
        dre = int(net.getNewVariable()); net.setLowerBound(dre, 0.0)
        dim = int(net.getNewVariable()); net.setLowerBound(dim, 0.0)
        net.addInequality([cre, dre], [1.0, -1.0], 0.0)
        net.addInequality([cre, dre], [-1.0, -1.0], 0.0)
        net.addInequality([cim, dim], [1.0, -1.0], 0.0)
        net.addInequality([cim, dim], [-1.0, -1.0], 0.0)
        d_list.extend([dre, dim])
    net.addInequality(d_list, [1.0]*len(d_list), float(tau))


def add_photometric(net, ivars, eps):
    if not np.isfinite(eps):
        return
    B1 = np.ones(128) / np.sqrt(128.0)
    B2_raw = np.array([(j - 7.5) for i in range(8) for j in range(16)], dtype=float)
    B2 = B2_raw / np.linalg.norm(B2_raw)
    alpha = int(net.getNewVariable())
    beta = int(net.getNewVariable())
    net.setLowerBound(alpha, -1e6); net.setUpperBound(alpha, 1e6)
    net.setLowerBound(beta, -1e6); net.setUpperBound(beta, 1e6)
    for i in range(NUM_PIXELS):
        r_i = int(net.getNewVariable())
        net.setLowerBound(r_i, -float(eps)); net.setUpperBound(r_i, float(eps))
        net.addEquality([int(ivars[i]), alpha, beta, r_i],
                        [1.0, -float(B1[i]), -float(B2[i]), -1.0], 0.0)


def add_per_pair_lipschitz(net, ivars, tau_per_pair):
    for (a, b), tau in zip(_PAIRS, tau_per_pair):
        net.addInequality([int(ivars[a]), int(ivars[b])], [1.0, -1.0], float(tau))
        net.addInequality([int(ivars[b]), int(ivars[a])], [1.0, -1.0], float(tau))


def solve(net, ivars, out_var, direc, bound):
    if direc == ">=":
        net.setLowerBound(out_var, float(bound))
    else:
        net.setUpperBound(out_var, float(bound))
    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    status, vals, _ = net.solve(verbose=False, options=opts)
    elapsed = time.time() - t0
    out = {"status": status.strip().lower(), "time_seconds": round(elapsed, 3)}
    if status.strip().lower() == "sat":
        out["CTE"] = round(float(vals[CTE_OUT_VAR]), 4)
        out["HE"] = round(float(vals[HE_OUT_VAR]), 4)
    return out


def solve_p5_deadzone(net, ivars):
    net.setLowerBound(CTE_OUT_VAR, -P5_DZ); net.setUpperBound(CTE_OUT_VAR, P5_DZ)
    net.setLowerBound(HE_OUT_VAR, -P5_DZ); net.setUpperBound(HE_OUT_VAR, P5_DZ)
    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    status, vals, _ = net.solve(verbose=False, options=opts)
    elapsed = time.time() - t0
    out = {"status": status.strip().lower(), "time_seconds": round(elapsed, 3)}
    if status.strip().lower() == "sat":
        out["CTE"] = round(float(vals[CTE_OUT_VAR]), 4)
        out["HE"] = round(float(vals[HE_OUT_VAR]), 4)
    return out


def run_suite(boxes, region_apply, suite_name):
    print(f"  [{suite_name}]", flush=True)
    out = {}
    queries = [
        ("P1a", "centered", CTE_OUT_VAR, ">=", P1_CTE),
        ("P1b", "centered", CTE_OUT_VAR, "<=", -P1_CTE),
        ("P4_left", "left", CTE_OUT_VAR, "<=", 0.0),
        ("P4_right", "right", CTE_OUT_VAR, ">=", 0.0),
    ]
    for name, subset, var, direc, bnd in queries:
        lo, hi = boxes[f"{subset}_lo"], boxes[f"{subset}_hi"]
        net, ivars = base_query(lo, hi)
        if net is None:
            out[name] = {"status": "region_empty"}; continue
        region_apply(net, ivars)
        r = solve(net, ivars, var, direc, bnd)
        out[name] = r
        print(f"    {name:9s} {r['status']:6s} t={r['time_seconds']:.2f}s "
              + (f" CTE={r.get('CTE')}" if r.get('CTE') is not None else ""), flush=True)
    # P5
    lo, hi = boxes["off_lo"], boxes["off_hi"]
    net, ivars = base_query(lo, hi)
    if net is not None:
        region_apply(net, ivars)
        r = solve_p5_deadzone(net, ivars)
        out["P5"] = r
        print(f"    P5        {r['status']:6s} t={r['time_seconds']:.2f}s "
              + (f" CTE={r.get('CTE')}" if r.get('CTE') is not None else ""), flush=True)
    else:
        out["P5"] = {"status": "region_empty"}
    return out


# -----------------------------------------------------------------------------
# EXPERIMENT A: Quantile sensitivity sweep
# -----------------------------------------------------------------------------
def calibrate_tv(centered, q):
    tvs = []
    for x in centered:
        img = x.reshape(8, 16)
        tvs.append(np.abs(np.diff(img, axis=1)).sum() + np.abs(np.diff(img, axis=0)).sum())
    return float(np.percentile(tvs, q)) if q < 100 else float(max(tvs))


def calibrate_fourier(centered, q):
    vals = []
    for x in centered:
        img = x.reshape(8, 16)
        C = np.fft.fft2(img) / np.sqrt(128.0)
        s = 0.0
        for k, l in HF_CELLS:
            s += abs(C[k, l].real) + abs(C[k, l].imag)
        vals.append(s)
    return float(np.percentile(vals, q)) if q < 100 else float(max(vals))


def calibrate_photometric(centered, q):
    B1 = np.ones(128) / np.sqrt(128.0)
    B2_raw = np.array([(j - 7.5) for i in range(8) for j in range(16)], dtype=float)
    B2 = B2_raw / np.linalg.norm(B2_raw)
    rs = []
    for x in centered:
        a = x @ B1; b = x @ B2
        r = x - a*B1 - b*B2
        rs.append(float(np.abs(r).max()))
    return float(np.percentile(rs, q)) if q < 100 else float(max(rs))


def exp_A_quantile_sensitivity(centered, boxes):
    print("\n[EXP A] Quantile sensitivity sweep", flush=True)
    results = {}
    quantiles = [90, 95, 99, 100]
    for q in quantiles:
        tau_tv = calibrate_tv(centered, q)
        tau_f  = calibrate_fourier(centered, q)
        eps_p  = calibrate_photometric(centered, q)
        print(f"  q={q}: TV={tau_tv:.3f} Fourier={tau_f:.3f} Photometric={eps_p:.4f}", flush=True)
        results[f"q{q}"] = {
            "tv_tau": tau_tv, "fourier_tau": tau_f, "photometric_eps": eps_p,
            "tv":   run_suite(boxes, lambda n, i, t=tau_tv: add_tv(n, i, t), f"TV q={q}"),
            "fourier": run_suite(boxes, lambda n, i, t=tau_f: add_fourier(n, i, t), f"Fourier q={q}"),
            "photometric": run_suite(boxes, lambda n, i, e=eps_p: add_photometric(n, i, e), f"Photometric q={q}"),
        }
    return results


# -----------------------------------------------------------------------------
# EXPERIMENT B: Per-pair Lipschitz on P2, P3, P6
# -----------------------------------------------------------------------------
def calibrate_per_pair_lipschitz(centered, q):
    tau = []
    for a, b in _PAIRS:
        diffs = np.abs(centered[:, a] - centered[:, b])
        tau.append(float(np.percentile(diffs, q)))
    return tau


def select_centered_sample(labels, n=20):
    cte = labels[:, 0]
    centered_idx = np.where(np.abs(cte) < 1.0)[0]
    return centered_idx[np.linspace(0, len(centered_idx)-1, n, dtype=int)]


def exp_B_per_pair_lipschitz_extension(images, labels, centered, boxes):
    print("\n[EXP B] Per-pair Lipschitz on P2, P3, P6", flush=True)
    out = {}
    tau_p99 = calibrate_per_pair_lipschitz(centered, 99)
    tau_p90 = calibrate_per_pair_lipschitz(centered, 90)

    # P3 at eps=0.05 (20 anchor sample) under per-pair Lipschitz
    print("  P3 under per-pair Lipschitz...", flush=True)
    sample = select_centered_sample(labels, 20)
    p3_subqueries = [
        (CTE_OUT_VAR, ">=", P1_CTE),
        (CTE_OUT_VAR, "<=", -P1_CTE),
        (HE_OUT_VAR, ">=", P1_HE),
        (HE_OUT_VAR, "<=", -P1_HE),
    ]
    for label, tau_pp in [("p99", tau_p99), ("p90", tau_p90)]:
        verified = violated = empty = 0
        for k, gi in enumerate(sample):
            x0 = images[int(gi)]
            pixel_lo = np.maximum(PIXEL_MIN, x0 - P3_EPS)
            pixel_hi = np.minimum(PIXEL_MAX, x0 + P3_EPS)
            v = True; em = False
            for var, direc, bnd in p3_subqueries:
                net, ivars = base_query(pixel_lo, pixel_hi)
                if net is None: em = True; v = False; break
                add_per_pair_lipschitz(net, ivars, tau_pp)
                r = solve(net, ivars, var, direc, bnd)
                if r["status"] == "region_empty": em = True
                if r["status"] != "unsat": v = False
            if em: empty += 1
            elif v: verified += 1
            else: violated += 1
        out[f"p3_lip_{label}"] = {"n_verified": verified, "n_violated": violated, "n_empty": empty, "n_total": len(sample)}
        print(f"    P3 lip {label}: {verified}/{len(sample)} verified", flush=True)

    # P2 (correctness band 1m) at eps=0.02 under per-pair Lipschitz
    print("  P2 under per-pair Lipschitz...", flush=True)
    cte = labels[:, 0]; he = labels[:, 1]
    for label, tau_pp in [("p99", tau_p99), ("p90", tau_p90)]:
        verified = violated = empty = nom_wrong = 0
        for k, gi in enumerate(sample):
            x0 = images[int(gi)]
            cte0 = cte[int(gi)]; he0 = he[int(gi)]
            net0 = Marabou.read_nnet(NNET_PATH)
            outv = net0.evaluate([x0])
            cpred = float(np.array(outv).flatten()[0])
            hpred = float(np.array(outv).flatten()[1])
            if abs(cpred - cte0) >= 1.0 or abs(hpred - he0) >= 5.0:
                nom_wrong += 1; continue
            pixel_lo = np.maximum(PIXEL_MIN, x0 - 0.02)
            pixel_hi = np.minimum(PIXEL_MAX, x0 + 0.02)
            p2_subqueries = [
                (CTE_OUT_VAR, ">=", float(cte0 + 1.0)),
                (CTE_OUT_VAR, "<=", float(cte0 - 1.0)),
                (HE_OUT_VAR, ">=", float(he0 + 5.0)),
                (HE_OUT_VAR, "<=", float(he0 - 5.0)),
            ]
            v = True; em = False
            for var, direc, bnd in p2_subqueries:
                net, ivars = base_query(pixel_lo, pixel_hi)
                if net is None: em = True; v = False; break
                add_per_pair_lipschitz(net, ivars, tau_pp)
                r = solve(net, ivars, var, direc, bnd)
                if r["status"] == "region_empty": em = True
                if r["status"] != "unsat": v = False
            if em: empty += 1
            elif v: verified += 1
            else: violated += 1
        out[f"p2_lip_{label}"] = {
            "n_verified": verified, "n_violated": violated, "n_empty": empty,
            "n_nominally_wrong": nom_wrong, "n_total": len(sample),
        }
        print(f"    P2 lip {label}: {verified}/{len(sample)} verified ({nom_wrong} nom wrong)", flush=True)

    # P6 axis-aligned bounds under per-pair Lipschitz
    print("  P6 axis-aligned bounds under per-pair Lipschitz...", flush=True)
    pixel_lo, pixel_hi = boxes["centered_lo"], boxes["centered_hi"]
    for label, tau_pp in [("p99", tau_p99)]:
        # Coarse binary search per axis
        def find_bound(direc, var, lo, hi, depth=4):
            best = lo if direc == 'upper' else hi
            for _ in range(depth):
                mid = 0.5*(lo + hi)
                net, ivars = base_query(pixel_lo, pixel_hi)
                add_per_pair_lipschitz(net, ivars, tau_pp)
                if direc == 'upper':
                    net.setLowerBound(var, float(mid))
                else:
                    net.setUpperBound(var, float(mid))
                opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
                status, _, _ = net.solve(verbose=False, options=opts)
                if status.strip().lower() == 'sat':
                    best = mid
                    if direc == 'upper': lo = mid
                    else: hi = mid
                else:
                    if direc == 'upper': hi = mid
                    else: lo = mid
            return best
        cte_u = find_bound('upper', CTE_OUT_VAR, 0, 50)
        cte_l = find_bound('lower', CTE_OUT_VAR, -50, 0)
        he_u  = find_bound('upper', HE_OUT_VAR, 0, 100)
        he_l  = find_bound('lower', HE_OUT_VAR, -100, 0)
        out[f"p6_lip_{label}"] = {
            "cte_bounds_approx": [round(cte_l, 2), round(cte_u, 2)],
            "he_bounds_approx":  [round(he_l, 2), round(he_u, 2)],
        }
        print(f"    P6 lip {label}: CTE~[{cte_l:.1f}, {cte_u:.1f}], HE~[{he_l:.1f}, {he_u:.1f}]", flush=True)

    return out


# -----------------------------------------------------------------------------
# EXPERIMENT C: Larger P3 fragility sweep
# -----------------------------------------------------------------------------
def exp_C_larger_p3_fragility(images, labels, n_anchors=100):
    print(f"\n[EXP C] Larger P3 fragility sweep on {n_anchors} anchors", flush=True)
    sample = select_centered_sample(labels, n_anchors)
    taus = [12, 10, 8, 6, 4]
    p3_subqueries = [
        (CTE_OUT_VAR, ">=", P1_CTE),
        (CTE_OUT_VAR, "<=", -P1_CTE),
        (HE_OUT_VAR, ">=", P1_HE),
        (HE_OUT_VAR, "<=", -P1_HE),
    ]
    results_by_tau = {}
    threshold_distribution = {}
    for tau in taus:
        verified_set = set()
        verified = violated = empty = 0
        t0 = time.time()
        for k, gi in enumerate(sample):
            x0 = images[int(gi)]
            pixel_lo = np.maximum(PIXEL_MIN, x0 - P3_EPS)
            pixel_hi = np.minimum(PIXEL_MAX, x0 + P3_EPS)
            v = True; em = False
            for var, direc, bnd in p3_subqueries:
                net, ivars = base_query(pixel_lo, pixel_hi)
                if net is None: em = True; v = False; break
                add_tv(net, ivars, tau)
                r = solve(net, ivars, var, direc, bnd)
                if r["status"] == "region_empty": em = True
                if r["status"] != "unsat": v = False
            if em: empty += 1
            elif v: verified += 1; verified_set.add(int(gi))
            else: violated += 1
        elapsed = time.time() - t0
        results_by_tau[f"tau_{tau}"] = {
            "tau": tau, "n_verified": verified, "n_violated": violated,
            "n_empty": empty, "n_total": n_anchors, "wall_s": round(elapsed, 1),
            "verified_indices": sorted(verified_set),
        }
        print(f"  tau={tau}: {verified}/{n_anchors} verified ({elapsed:.1f}s)", flush=True)

    # Compute threshold per anchor
    thresholds = {}
    for k, gi in enumerate(sample):
        gi = int(gi)
        thr = None
        for tau in taus:
            if gi in results_by_tau[f"tau_{tau}"]["verified_indices"]:
                if thr is None or tau > thr:
                    thr = tau
        thresholds[gi] = thr if thr is not None else 0
    # Distribution
    from collections import Counter
    dist = Counter(thresholds.values())
    out = {
        "n_anchors": n_anchors,
        "taus": taus,
        "results_by_tau": {k: {kk: vv for kk, vv in v.items() if kk != "verified_indices"}
                           for k, v in results_by_tau.items()},
        "threshold_per_anchor": thresholds,
        "threshold_distribution": dict(sorted(dist.items(), key=lambda x: -x[0])),
    }
    print(f"  Threshold distribution: {dict(sorted(dist.items(), key=lambda x: -x[0]))}", flush=True)
    return out


def main():
    print("Loading data...", flush=True)
    images, labels = load_data()
    cte = labels[:, 0]
    centered = images[np.abs(cte) < 1.0]
    left     = images[cte > 2.0]
    right    = images[cte < -2.0]
    off      = images[np.abs(cte) > 1.0]
    boxes = {
        "centered_lo": centered.min(axis=0), "centered_hi": centered.max(axis=0),
        "left_lo":     left.min(axis=0),     "left_hi":     left.max(axis=0),
        "right_lo":    right.min(axis=0),    "right_hi":    right.max(axis=0),
        "off_lo":      off.min(axis=0),      "off_hi":      off.max(axis=0),
    }
    print(f"  centered={len(centered)} left={len(left)} right={len(right)} off={len(off)}", flush=True)

    out = {
        "provenance": {
            "script": "verify_math_sensitivity_batch.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timeout_seconds": TIMEOUT_S,
        }
    }
    out["exp_A_quantile_sensitivity"] = exp_A_quantile_sensitivity(centered, boxes)
    out["exp_B_per_pair_lipschitz_extension"] = exp_B_per_pair_lipschitz_extension(images, labels, centered, boxes)
    out["exp_C_larger_p3_fragility"] = exp_C_larger_p3_fragility(images, labels, n_anchors=100)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
