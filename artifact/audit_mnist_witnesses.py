"""
Triangulated witness audit on MNIST verification results.

For each SAT witness recovered by verify_mnist_protocol.py:
  1. Classifier probability under the MNIST plausibility classifier
  2. Nearest-real-image L_2 distance over the held-out test set,
     conditioned on the witness's true class
  3. Per-pair Lipschitz violation fraction on the 28x28 adjacency set

Reports aggregates per (region, property) and a per-witness CSV-like JSON list
with all three signals, mirroring the TinyTaxiNet audit_stratified_witnesses.py
structure.

Outputs:
  mnist_witness_audit.json - per-witness scores + aggregates
"""
import json
import math
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import tensorflow as tf

import mnist_regions as mr


RESULTS_PATH = "mnist_protocol_results.json"
CLASSIFIER_PATH = "mnist_plausibility_classifier_weights.npz"
DATA_PATH = "mnist_test_data.npz"
OUTPUT_JSON = "mnist_witness_audit.json"

ROWS, COLS = 28, 28
N_PIX = 784


def classifier_prob_batch(X, w):
    """Forward through 784->64(ReLU)->32(ReLU)->1 sigmoid."""
    h1 = np.maximum(X @ w["W0"].T + w["b0"], 0)
    h2 = np.maximum(h1 @ w["W1"].T + w["b1"], 0)
    logit = h2 @ w["W2"].T + w["b2"]
    return 1.0 / (1.0 + np.exp(-logit.flatten()))


def per_pair_lipschitz_violation(x, taus):
    """Fraction of adjacent pairs (a,b) where |x_a - x_b| > tau_{a,b}."""
    diffs = np.array([abs(x[a] - x[b]) for a, b in mr.PAIRS])
    return float(np.mean(diffs > taus))


def calibrate_per_pair_lip(X_class, q=90):
    """Per-pair tau at quantile q over a class subset."""
    taus = np.zeros(len(mr.PAIRS))
    for k, (a, b) in enumerate(mr.PAIRS):
        taus[k] = np.percentile(np.abs(X_class[:, a] - X_class[:, b]), q)
    return taus


def baseline_per_pair_violation_rate(X_class, taus):
    rates = np.zeros(len(X_class))
    for i, x in enumerate(X_class):
        rates[i] = per_pair_lipschitz_violation(x, taus)
    return float(rates.mean())


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)
    classifier_w = np.load(CLASSIFIER_PATH)
    data = np.load(DATA_PATH)
    X_te = data["X_test"].astype(np.float64)
    y_te = data["y_test"]

    (X_tr_raw, y_tr), _ = tf.keras.datasets.mnist.load_data()
    X_tr = X_tr_raw.reshape(-1, 784).astype(np.float64) / 255.0

    print("Calibrating per-class per-pair Lipschitz thresholds...", flush=True)
    class_taus = {}
    baseline_rates = {}
    for c in range(10):
        Xk = X_tr[y_tr == c]
        class_taus[c] = calibrate_per_pair_lip(Xk, q=90)
        baseline_rates[c] = baseline_per_pair_violation_rate(Xk[:500], class_taus[c])
        print(f"  class {c}: pp-baseline rate = {baseline_rates[c]:.4f}", flush=True)

    test_by_class = {c: X_te[y_te == c] for c in range(10)}

    all_records = []
    by_region = defaultdict(list)
    by_property_region = defaultdict(list)

    queries = results["per_anchor"] + sum(results["per_class"].values(), [])
    sat_count = 0
    for q in queries:
        if q.get("status") != "sat" or "witness_pixels" not in q:
            continue
        sat_count += 1
        x_w = np.array(q["witness_pixels"], dtype=np.float64)
        true_c = q["true_class"]

        prob = float(classifier_prob_batch(x_w[None, :], classifier_w)[0])
        Xtc = test_by_class[true_c]
        l2 = float(np.min(np.linalg.norm(Xtc - x_w[None, :], axis=1)))
        ppv = per_pair_lipschitz_violation(x_w, class_taus[true_c])

        rec = {
            "property": q.get("property"),
            "region": q.get("region"),
            "anchor_idx": q.get("anchor_idx"),
            "true_class": true_c,
            "target_class": q.get("target_class"),
            "eps": q.get("eps"),
            "classifier_prob": prob,
            "nearest_real_L2_same_class": l2,
            "pp_lip_violation_fraction": ppv,
            "pp_lip_baseline_rate": baseline_rates[true_c],
            "witness_linf_from_anchor": q.get("witness_linf_from_anchor"),
        }
        all_records.append(rec)
        by_region[q["region"]].append(rec)
        by_property_region[(q["property"], q["region"])].append(rec)

    print(f"\n{sat_count} SAT witnesses audited.\n", flush=True)

    def agg(rs):
        if not rs:
            return {}
        probs = np.array([r["classifier_prob"] for r in rs])
        l2s = np.array([r["nearest_real_L2_same_class"] for r in rs])
        ppvs = np.array([r["pp_lip_violation_fraction"] for r in rs])
        return {
            "n": len(rs),
            "classifier_prob_mean": float(probs.mean()),
            "classifier_prob_median": float(np.median(probs)),
            "classifier_above_0_5": int((probs > 0.5).sum()),
            "classifier_below_0_1": int((probs < 0.1).sum()),
            "nearest_real_L2_median": float(np.median(l2s)),
            "nearest_real_L2_mean": float(l2s.mean()),
            "pp_lip_violation_mean": float(ppvs.mean()),
            "pp_lip_violation_median": float(np.median(ppvs)),
        }

    by_region_agg = {r: agg(rs) for r, rs in by_region.items()}
    by_property_region_agg = {f"{p}::{r}": agg(rs) for (p, r), rs in by_property_region.items()}

    print("=== AGGREGATES BY REGION ===")
    for r, a in sorted(by_region_agg.items()):
        print(f"  {r:18s} n={a['n']:3d}  cls_mean={a['classifier_prob_mean']:.3f}  "
              f"L2_med={a['nearest_real_L2_median']:.2f}  pp_viol_mean={a['pp_lip_violation_mean']:.3f}",
              flush=True)

    out = {
        "provenance": {
            "script": "audit_mnist_witnesses.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "results_source": RESULTS_PATH,
        },
        "per_class_pp_baseline_rates": {str(k): v for k, v in baseline_rates.items()},
        "by_region": by_region_agg,
        "by_property_region": by_property_region_agg,
        "per_witness": all_records,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
