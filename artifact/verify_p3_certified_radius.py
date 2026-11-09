"""
Per-image certified safety radius via binary search on epsilon.

For each of the 20 sampled centered images, we find the maximum epsilon such
that the |CTE| < 10 AND |HE| < 90 safety property verifies inside the L_inf
ball of radius epsilon, intersected with [0,1] (clip) and the linear invariant
sum(x_i) = 64 (mean = 0.5).

The certified radius eps* is the largest tested epsilon at which all four
sub-queries return UNSAT, to a tolerance of 0.005.

Search interval: [eps_lo, eps_hi]. We start with a coarse grid {0.005, 0.01,
0.02, 0.05, 0.10, 0.15, 0.20} to find the bracket, then bisect within it.
"""
import json
import time

import h5py
import numpy as np
from maraboupy import Marabou


NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "verification_p3_certified_radius.json"

NUM_PIXELS = 128
SUM_TARGET = 0.5 * NUM_PIXELS
PIXEL_MIN = 0.0
PIXEL_MAX = 1.0
P3_CTE_BOUND = 10.0
P3_HE_BOUND = 90.0
P3_CENTERED_THRESHOLD = 1.0
NUM_IMAGES = 20

GRID = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
TOL = 0.005
TIMEOUT_S = 120


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:]
        labels = f["X_train"][:]
    return images, labels


def is_verified_at(x0, eps):
    """All four safety sub-queries UNSAT inside the constrained ball?"""
    sub_queries = [
        (192, ">=",  P3_CTE_BOUND),
        (192, "<=", -P3_CTE_BOUND),
        (193, ">=",  P3_HE_BOUND),
        (193, "<=", -P3_HE_BOUND),
    ]
    for var, direc, bound in sub_queries:
        net = Marabou.read_nnet(NNET_PATH)
        input_vars = net.inputVars[0].flatten()
        for i in range(NUM_PIXELS):
            lo = max(PIXEL_MIN, float(x0[i] - eps))
            hi = min(PIXEL_MAX, float(x0[i] + eps))
            net.setLowerBound(int(input_vars[i]), lo)
            net.setUpperBound(int(input_vars[i]), hi)
        net.addEquality(list(input_vars), [1.0] * NUM_PIXELS, SUM_TARGET)
        if direc == ">=":
            net.setLowerBound(var, float(bound))
        else:
            net.setUpperBound(var, float(bound))
        opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
        status, _, _ = net.solve(verbose=False, options=opts)
        status = status.strip().lower()
        if status != "unsat":
            return False, status
    return True, "unsat"


def certified_radius(x0):
    """Return the largest epsilon at which all 4 sub-queries are UNSAT, plus
    log of the search."""
    log = []

    # Coarse grid: find the largest eps in GRID that verifies, and the smallest
    # that does not. Stop early as soon as we cross the boundary going up.
    last_pass = 0.0
    first_fail = None
    for eps in GRID:
        ok, last_status = is_verified_at(x0, eps)
        log.append({"eps": eps, "verified": ok, "status_last_subquery": last_status})
        if ok:
            last_pass = eps
        else:
            first_fail = eps
            break

    # If first_fail is None, the largest grid point still verified.
    if first_fail is None:
        return {"radius": last_pass, "lower_bracket": last_pass,
                "upper_bracket": None, "log": log,
                "note": f"verified at all grid points up to {last_pass}"}

    # Bisect between last_pass and first_fail to tolerance.
    lo, hi = last_pass, first_fail
    while (hi - lo) > TOL:
        mid = 0.5 * (lo + hi)
        ok, last_status = is_verified_at(x0, mid)
        log.append({"eps": round(mid, 5), "verified": ok,
                    "status_last_subquery": last_status})
        if ok:
            lo = mid
        else:
            hi = mid

    return {"radius": round(lo, 5), "lower_bracket": round(lo, 5),
            "upper_bracket": round(hi, 5), "log": log,
            "note": f"bisected to tolerance {TOL}"}


def main():
    images, labels = load_data()
    cte = labels[:, 0]
    centered_idx = np.where(np.abs(cte) < P3_CENTERED_THRESHOLD)[0]
    sample = centered_idx[np.linspace(0, len(centered_idx) - 1, NUM_IMAGES, dtype=int)]

    out = {
        "spec": "|CTE| < 10 AND |HE| < 90 inside L_inf eps ball + clip + mean=0.5",
        "tolerance": TOL,
        "grid": GRID,
        "per_image": [],
    }

    print(f"Per-image certified radius (mean-constrained, clipped). tol={TOL}\n")
    print(f"{'#':>3} {'idx':>5} {'true_CTE':>9} {'cert_eps':>9} {'bracket':>16} {'queries':>8}  time")
    t_total = time.time()
    for k, idx in enumerate(sample):
        x0 = images[int(idx)].flatten().astype(np.float64)
        t0 = time.time()
        r = certified_radius(x0)
        elapsed = time.time() - t0
        bracket = f"[{r['lower_bracket']}, {r['upper_bracket']}]"
        n_queries = len(r["log"])
        print(f"{k+1:>3} {idx:>5} {float(cte[int(idx)]):>+9.3f} "
              f"{r['radius']:>9.4f} {bracket:>16} {n_queries:>8}  {elapsed:>5.1f}s")
        r["dataset_index"] = int(idx)
        r["true_CTE"] = float(cte[int(idx)])
        out["per_image"].append(r)

    radii = [r["radius"] for r in out["per_image"]]
    out["summary"] = {
        "n_images": NUM_IMAGES,
        "min_radius": float(min(radii)),
        "max_radius": float(max(radii)),
        "median_radius": float(np.median(radii)),
        "mean_radius": float(np.mean(radii)),
        "fraction_above_0.02": float(sum(r >= 0.02 for r in radii)) / NUM_IMAGES,
        "fraction_above_0.05": float(sum(r >= 0.05 for r in radii)) / NUM_IMAGES,
    }
    print(f"\nElapsed total: {time.time() - t_total:.1f}s")
    print(f"\nMin / median / max certified eps: "
          f"{out['summary']['min_radius']:.4f} / "
          f"{out['summary']['median_radius']:.4f} / "
          f"{out['summary']['max_radius']:.4f}")
    print(f"Fraction with eps* >= 0.02: {out['summary']['fraction_above_0.02']:.0%}")
    print(f"Fraction with eps* >= 0.05: {out['summary']['fraction_above_0.05']:.0%}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
