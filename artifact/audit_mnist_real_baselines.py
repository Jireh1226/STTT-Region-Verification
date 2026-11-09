"""
Real-image baselines for the MNIST witness-audit table, plus the local-PCA
variance-explained statistic for the MNIST R4 regions.

Fills the two missing real-image reference signals in the MNIST audit:
  1. Plausibility-classifier probability on real MNIST test digits.
  2. Nearest-real L2 (each test digit to its nearest same-class test digit,
     excluding itself), matching the reference set used for witnesses in
     audit_mnist_witnesses.py.
Also computes, for the 20 P3 anchors and 10 representative anchors of
mnist_protocol_results.json, the fraction of same-class-neighbor deviation
variance captured by the top-8 local-PCA directions (the MNIST analogue of
the TaxiNet 92% statistic).

Outputs:
  mnist_real_baselines.json
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf

import mnist_regions as mr

RESULTS_PATH = "mnist_protocol_results.json"
CLASSIFIER_PATH = "mnist_plausibility_classifier_weights.npz"
DATA_PATH = "mnist_test_data.npz"
OUTPUT_JSON = "mnist_real_baselines.json"

N_PIX = 784


def classifier_prob_batch(X, w):
    h1 = np.maximum(X @ w["W0"].T + w["b0"], 0)
    h2 = np.maximum(h1 @ w["W1"].T + w["b1"], 0)
    logit = np.clip((h2 @ w["W2"].T + w["b2"]).flatten(), -30, 30)
    return 1.0 / (1.0 + np.exp(-logit))


def main():
    w = np.load(CLASSIFIER_PATH)
    data = np.load(DATA_PATH)
    X_te = data["X_test"].astype(np.float64)
    y_te = data["y_test"]
    (X_tr_raw, y_tr), _ = tf.keras.datasets.mnist.load_data()
    X_tr = X_tr_raw.reshape(-1, N_PIX).astype(np.float64) / 255.0

    # 1. Classifier probability on all real test digits
    probs = classifier_prob_batch(X_te, w)
    clf_baseline = {
        "n": int(len(probs)),
        "mean": float(probs.mean()),
        "median": float(np.median(probs)),
        "frac_above_0.5": float((probs > 0.5).mean()),
        "frac_below_0.1": float((probs < 0.1).mean()),
    }
    print(f"classifier on real test digits: mean={clf_baseline['mean']:.3f}, "
          f"median={clf_baseline['median']:.3f}, "
          f">.5: {clf_baseline['frac_above_0.5']:.3f}")

    # 2. Nearest same-class test-digit L2 (excluding self), per class
    nn_l2 = []
    for c in range(10):
        Xc = X_te[y_te == c]
        # chunked pairwise distances within the class slice
        for i0 in range(0, len(Xc), 200):
            chunk = Xc[i0:i0 + 200]
            d2 = (np.sum(chunk**2, axis=1)[:, None]
                  + np.sum(Xc**2, axis=1)[None, :]
                  - 2.0 * chunk @ Xc.T)
            d2 = np.maximum(d2, 0.0)
            for r in range(len(chunk)):
                d2[r, i0 + r] = np.inf   # exclude self
            nn_l2.extend(np.sqrt(d2.min(axis=1)).tolist())
    nn_l2 = np.array(nn_l2)
    l2_baseline = {
        "n": int(len(nn_l2)),
        "median": float(np.median(nn_l2)),
        "mean": float(nn_l2.mean()),
        "p90": float(np.percentile(nn_l2, 90)),
    }
    print(f"nearest same-class test L2: median={l2_baseline['median']:.2f}, "
          f"mean={l2_baseline['mean']:.2f}")

    # 3. Local-PCA top-8 variance explained at the protocol anchors
    res = json.load(open(RESULTS_PATH))
    anchors = sorted({(q["anchor_idx"], q["true_class"])
                      for q in res["per_anchor"] if "anchor_idx" in q})
    rep = res["provenance"].get("rep_anchors")
    rep_anchors = ([(int(v), int(c)) for c, v in rep.items()]
                   if isinstance(rep, dict) else [])
    var_expl = {}
    for tag, pairs in [("p3_anchors", anchors), ("rep_anchors", rep_anchors)]:
        fracs = []
        for a_idx, c in pairs:
            x0 = X_te[a_idx].astype(np.float64)
            mask = y_tr == c
            Xk = X_tr[mask]
            diffs = Xk - x0[None, :]
            dists = np.linalg.norm(diffs, axis=1)
            n = min(100, mask.sum())
            nearest = np.argsort(dists)[1:n + 1]
            D = diffs[nearest]
            _, S, _ = np.linalg.svd(D, full_matrices=False)
            fracs.append(float((S[:8]**2).sum() / (S**2).sum()))
        if fracs:
            var_expl[tag] = {
                "n_anchors": len(fracs),
                "mean": float(np.mean(fracs)),
                "min": float(np.min(fracs)),
                "max": float(np.max(fracs)),
            }
            print(f"{tag}: top-8 variance explained "
                  f"mean={np.mean(fracs):.3f} range "
                  f"[{np.min(fracs):.3f}, {np.max(fracs):.3f}]")

    out = {
        "provenance": {
            "script": "audit_mnist_real_baselines.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "classifier": CLASSIFIER_PATH,
            "l2_reference": "nearest same-class MNIST test digit, self excluded",
            "pca": "top-8 directions over 100 nearest same-class train "
                   "neighbors (matches mnist_regions.compute_local_pca)",
        },
        "real_test_classifier_baseline": clf_baseline,
        "real_test_nearest_same_class_L2": l2_baseline,
        "local_pca_top8_variance_explained": var_expl,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
