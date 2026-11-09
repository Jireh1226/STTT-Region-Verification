"""
Witness audit for the TaxiNet R5 (label-conditioned per-pixel box) region.

The original verify_p3_label_box.py run stored only output values (CTE, HE,
input_mean) for its SAT witnesses, so the R5 family could not be included in
the stratified witness audit. This script re-runs the same 20 label-box P3
queries (identical region construction: same-state neighbor per-pixel box,
mean(x)=0.5, clipping), stores the full 128-pixel witness vector for every
SAT sub-query, and scores each witness with the same three audit signals used
in audit_stratified_witnesses.py:

  - plausibility-classifier probability
  - L2 distance to nearest real centered image
  - per-pair Lipschitz violation fraction vs the centered p99 calibration

Outputs:
  r5_label_box_witness_audit.json
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import time
from datetime import datetime, timezone

import h5py
import numpy as np
from maraboupy import Marabou


NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
CLF_WEIGHTS = "plausibility_classifier.npz"
OUTPUT_JSON = "r5_label_box_witness_audit.json"

NUM_PIXELS = 128
SUM_TARGET = 0.5 * NUM_PIXELS
PIXEL_MIN, PIXEL_MAX = 0.0, 1.0
P3_CTE_BOUND, P3_HE_BOUND = 10.0, 90.0
P3_CENTERED_THRESHOLD = 1.0
NUM_IMAGES = 20
CTE_BIN_HALF, HE_BIN_HALF = 0.5, 5.0
TIMEOUT_S = 300


def load_classifier():
    w = np.load(CLF_WEIGHTS)
    return [(w["c1_W"], w["c1_b"]),
            (w["c2_W"], w["c2_b"]),
            (w["score_W"], w["score_b"])]


def score_clf(x, layers):
    h = x[None, :]
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    return float(h.flatten()[0])


def prob_from_logit(logit):
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))


def pair_list():
    pairs = []
    for i in range(8):
        for j in range(16):
            if j + 1 < 16:
                pairs.append((i * 16 + j, i * 16 + j + 1))
            if i + 1 < 8:
                pairs.append((i * 16 + j, (i + 1) * 16 + j))
    return pairs


def main():
    print("Loading dataset, classifier...", flush=True)
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float64)
        labels = f["X_train"][:].astype(np.float64)
    cte, he = labels[:, 0], labels[:, 1]
    centered = images[np.abs(cte) < P3_CENTERED_THRESHOLD]
    clf = load_classifier()

    pairs = pair_list()
    taus_p99 = np.array([
        float(np.percentile(np.abs(centered[:, a] - centered[:, b]), 99))
        for a, b in pairs
    ])

    centered_idx = np.where(np.abs(cte) < P3_CENTERED_THRESHOLD)[0]
    sample = centered_idx[np.linspace(0, len(centered_idx) - 1, NUM_IMAGES, dtype=int)]

    sub_queries = [
        ("a", 192, ">=",  P3_CTE_BOUND),
        ("b", 192, "<=", -P3_CTE_BOUND),
        ("c", 193, ">=",  P3_HE_BOUND),
        ("d", 193, "<=", -P3_HE_BOUND),
    ]

    witnesses = []
    verified = 0
    t0 = time.time()
    for k, idx in enumerate(sample):
        mask = (np.abs(cte - cte[idx]) <= CTE_BIN_HALF) & \
               (np.abs(he - he[idx]) <= HE_BIN_HALF)
        block = images[mask]
        pmin = np.maximum(PIXEL_MIN, block.min(axis=0))
        pmax = np.minimum(PIXEL_MAX, block.max(axis=0))
        all_unsat = True
        for tag, var, direc, bound in sub_queries:
            net = Marabou.read_nnet(NNET_PATH)
            ivars = [int(v) for v in net.inputVars[0].flatten()]
            for i in range(NUM_PIXELS):
                net.setLowerBound(ivars[i], float(pmin[i]))
                net.setUpperBound(ivars[i], float(pmax[i]))
            net.addEquality(ivars, [1.0] * NUM_PIXELS, SUM_TARGET)
            if direc == ">=":
                net.setLowerBound(var, float(bound))
            else:
                net.setUpperBound(var, float(bound))
            opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
            status, vals, _ = net.solve(verbose=False, options=opts)
            status = status.strip().lower()
            if status == "sat":
                all_unsat = False
                x = np.array([vals[v] for v in ivars])
                logit = score_clf(x, clf)
                d_real = float(np.min(np.linalg.norm(centered - x, axis=1)))
                viol = int(sum(abs(x[a] - x[b]) > t
                               for (a, b), t in zip(pairs, taus_p99)))
                witnesses.append({
                    "image": k + 1,
                    "dataset_index": int(idx),
                    "sub_query": tag,
                    "CTE": round(float(vals[192]), 4),
                    "HE": round(float(vals[193]), 4),
                    "classifier_prob": prob_from_logit(logit),
                    "classifier_logit": logit,
                    "nearest_real_centered_L2": d_real,
                    "per_pair_violation_frac": viol / len(pairs),
                    "witness": [round(float(v), 6) for v in x],
                })
        if all_unsat:
            verified += 1
        print(f"  img{k+1:>2} idx={idx}  "
              f"{'VERIFIED' if all_unsat else 'VIOLATED'}", flush=True)

    probs = np.array([w["classifier_prob"] for w in witnesses])
    l2s = np.array([w["nearest_real_centered_L2"] for w in witnesses])
    viols = np.array([w["per_pair_violation_frac"] for w in witnesses])
    agg = {
        "n_witnesses": len(witnesses),
        "verified_images": verified,
        "total_images": NUM_IMAGES,
        "classifier_prob_mean": float(probs.mean()),
        "classifier_prob_median": float(np.median(probs)),
        "classifier_above_0_5": int((probs > 0.5).sum()),
        "classifier_below_0_1": int((probs < 0.1).sum()),
        "nearest_real_L2_median": float(np.median(l2s)),
        "per_pair_violation_frac_mean": float(viols.mean()),
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    out = {
        "provenance": {
            "script": "audit_r5_label_box_witnesses.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "region": "R5 label-conditioned per-pixel box + mean=0.5 + clip",
            "calibration": "per-pair p99 over centered subset (matches audit_stratified_witnesses.py)",
        },
        "aggregate": agg,
        "witnesses": witnesses,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nAggregate: {json.dumps(agg, indent=2)}")
    print(f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
