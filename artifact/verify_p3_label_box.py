"""
P3 verification under per-image label-conditioned per-pixel boxes.

For each sampled centered image x0:
  1. Find same-state neighbors:  |dCTE| <= 0.5,  |dHE| <= 5,  including x0.
  2. Per-pixel box: [min, max] over the neighbor pixel values.
  3. Verify safety property over that box, with sum(x_i) = 64 and clipping.

This is a simpler alternative to the local-PCA region: it uses the per-pixel
envelope of same-state real images instead of a low-dim subspace. It does
not enforce that x lies in a low-dim subspace, so it is a strict superset
of the local-PCA region.
"""
import json
import time

import h5py
import numpy as np
from maraboupy import Marabou


NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "verification_p3_label_box.json"

NUM_PIXELS = 128
SUM_TARGET = 0.5 * NUM_PIXELS
PIXEL_MIN = 0.0
PIXEL_MAX = 1.0
P3_CTE_BOUND = 10.0
P3_HE_BOUND = 90.0
P3_CENTERED_THRESHOLD = 1.0
NUM_IMAGES = 20
CTE_BIN_HALF = 0.5
HE_BIN_HALF = 5.0
TIMEOUT_S = 300


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        return f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float64), \
               f["X_train"][:].astype(np.float64)


def select_centered_sample(labels):
    cte = labels[:, 0]
    centered_idx = np.where(np.abs(cte) < P3_CENTERED_THRESHOLD)[0]
    return centered_idx[np.linspace(0, len(centered_idx) - 1, NUM_IMAGES, dtype=int)]


def label_neighbor_box(images, labels, idx):
    cte = labels[:, 0]; he = labels[:, 1]
    cte_q, he_q = cte[idx], he[idx]
    mask = (np.abs(cte - cte_q) <= CTE_BIN_HALF) & \
           (np.abs(he - he_q) <= HE_BIN_HALF)
    block = images[mask]   # includes x0
    pmin = np.maximum(PIXEL_MIN, block.min(axis=0))
    pmax = np.minimum(PIXEL_MAX, block.max(axis=0))
    return block.shape[0], pmin, pmax


def solve_one(pmin, pmax, out_var, direc, bound):
    if np.any(pmin > pmax):
        return {"status": "region_empty"}
    if pmin.sum() > SUM_TARGET or pmax.sum() < SUM_TARGET:
        return {"status": "region_empty_under_mean"}

    net = Marabou.read_nnet(NNET_PATH)
    input_vars = [int(v) for v in net.inputVars[0].flatten()]
    for i in range(NUM_PIXELS):
        net.setLowerBound(input_vars[i], float(pmin[i]))
        net.setUpperBound(input_vars[i], float(pmax[i]))
    net.addEquality(input_vars, [1.0] * NUM_PIXELS, SUM_TARGET)
    if direc == ">=":
        net.setLowerBound(out_var, float(bound))
    else:
        net.setUpperBound(out_var, float(bound))

    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    status, vals, _ = net.solve(verbose=False, options=opts)
    elapsed = time.time() - t0
    status = status.strip().lower()
    out = {"status": status, "time_seconds": round(elapsed, 4)}
    if status == "sat":
        cx = np.array([vals[v] for v in input_vars])
        out["counterexample"] = {
            "CTE": round(float(vals[192]), 6),
            "HE": round(float(vals[193]), 6),
            "input_mean": round(float(cx.mean()), 6),
        }
    return out


def main():
    images, labels = load_data()
    sample = select_centered_sample(labels)

    sub_queries = [
        ("a", 192, ">=",  P3_CTE_BOUND),
        ("b", 192, "<=", -P3_CTE_BOUND),
        ("c", 193, ">=",  P3_HE_BOUND),
        ("d", 193, "<=", -P3_HE_BOUND),
    ]

    out = {"results": {}}
    verified = violated = empty = 0
    print(f"P3 with label-conditioned per-pixel box "
          f"(|dCTE|<={CTE_BIN_HALF}, |dHE|<={HE_BIN_HALF}, mean=0.5, clip)\n")
    t0 = time.time()
    for k, idx in enumerate(sample):
        n_neigh, pmin, pmax = label_neighbor_box(images, labels, int(idx))
        median_hw = float(np.median((pmax - pmin) / 2))
        max_hw = float(np.max((pmax - pmin) / 2))
        sub_results = {}
        all_unsat = True
        for tag, var, direc, b in sub_queries:
            r = solve_one(pmin, pmax, var, direc, b)
            sub_results[tag] = r
            if r["status"] != "unsat":
                all_unsat = False
        entry = {
            "dataset_index": int(idx),
            "n_label_neighbors": int(n_neigh),
            "median_pixel_half_width": round(median_hw, 5),
            "max_pixel_half_width": round(max_hw, 5),
            "sub_queries": sub_results,
            "verified": bool(all_unsat),
        }
        out["results"][f"image_{k+1}"] = entry
        if all_unsat:
            verified += 1
            tag = "VERIFIED"
        else:
            violated += 1
            tag = "VIOLATED"
        statuses = " ".join(sub_results[t]["status"][:3] for t, *_ in sub_queries)
        print(f"  img{k+1:>2} idx={idx} m={n_neigh:>3} "
              f"hw_med={median_hw:.4f} hw_max={max_hw:.4f}  {tag:<8} [{statuses}]")

    out["summary"] = {
        "verified": verified, "violated": violated, "empty": empty,
        "total": NUM_IMAGES,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    print(f"\n=> {verified}/{NUM_IMAGES} verified, "
          f"{violated}/{NUM_IMAGES} violated  ({time.time() - t0:.1f}s)")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
