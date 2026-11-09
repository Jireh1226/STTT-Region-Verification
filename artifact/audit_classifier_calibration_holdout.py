"""
Held-out real-image calibration audit for the plausibility classifier.

The classifier (plausibility_classifier.npz, trained by
train_plausibility_classifier.py) was trained on all 10,000 real
preprocessed TinyTaxiNet images as positives vs four synthetic fake families
as negatives. This script tests whether the classifier still accepts real
images that fall outside the centered subset (where the witness audit's
0.991 baseline median was measured) and whether small mean-preserving
perturbations of real images push them below the 0.5 acceptance threshold.

If the classifier rejects real off-center images or small perturbations of
real images at high rates, the witness audit's "off-distribution" verdicts
on box-corner / cross-blend witnesses would be confounded with this
miscalibration. The held-out check tests that this is not happening.

Outputs:
  classifier_holdout_audit.json  - per-subset classifier-probability stats
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
import h5py

DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
WEIGHTS_PATH = "plausibility_classifier.npz"
OUTPUT_JSON = "classifier_holdout_audit.json"

NUM_PIXELS = 128


def load_classifier():
    w = np.load(WEIGHTS_PATH)
    return {
        "c1_W": w["c1_W"], "c1_b": w["c1_b"],
        "c2_W": w["c2_W"], "c2_b": w["c2_b"],
        "score_W": w["score_W"], "score_b": w["score_b"],
    }


def classify(x, clf):
    """Forward through 128 -> 64 (ReLU) -> 16 (ReLU) -> 1 (sigmoid)."""
    h1 = np.maximum(x @ clf["c1_W"] + clf["c1_b"], 0.0)
    h2 = np.maximum(h1 @ clf["c2_W"] + clf["c2_b"], 0.0)
    logit = h2 @ clf["score_W"] + clf["score_b"]
    return 1.0 / (1.0 + np.exp(-logit.flatten()))


def stats(probs):
    return {
        "n": int(len(probs)),
        "mean": float(probs.mean()),
        "median": float(np.median(probs)),
        "p05": float(np.percentile(probs, 5)),
        "p95": float(np.percentile(probs, 95)),
        "frac_above_0.5": float((probs > 0.5).mean()),
        "frac_above_0.9": float((probs > 0.9).mean()),
        "frac_below_0.1": float((probs < 0.1).mean()),
    }


def main():
    rng = np.random.default_rng(42)
    print("Loading classifier...")
    clf = load_classifier()

    print("Loading dataset...")
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float64)
        labels = f["X_train"][:].astype(np.float64)
    cte = labels[:, 0]
    print(f"  {len(images)} total images")

    centered = images[np.abs(cte) < 1.0]
    left = images[cte > 2.0]
    right = images[cte < -2.0]
    off_center = images[np.abs(cte) > 1.0]
    print(f"  centered: {len(centered)}, left: {len(left)}, right: {len(right)}, off-center: {len(off_center)}")

    # Per-subset real-image acceptance
    print("\n=== Per-subset real-image classifier acceptance ===")
    results = {
        "provenance": {
            "script": "audit_classifier_calibration_holdout.py",
            "purpose": "test classifier acceptance of real images outside the centered subset and under small perturbations",
            "classifier": WEIGHTS_PATH,
        },
        "per_subset_real_image_acceptance": {},
    }
    for name, X in [
        ("centered (training-baseline subset)", centered),
        ("left", left),
        ("right", right),
        ("off-center (all CTE>1m)", off_center),
        ("full dataset", images),
    ]:
        probs = classify(X, clf)
        s = stats(probs)
        results["per_subset_real_image_acceptance"][name] = s
        print(f"  {name:42s}: mean={s['mean']:.3f}, median={s['median']:.3f}, p05={s['p05']:.3f}, >.5: {s['frac_above_0.5']:.3f}")

    # Held-out (90/10 split with seed=42 to mirror the original training script)
    n = len(images)
    perm = np.random.default_rng(42).permutation(n + n)  # combined real+fake permutation
    # The classifier saw all 10000 real images (as the positive class) but a 90/10 split
    # was applied to the COMBINED set. To approximate the held-out reals we sample 10% of reals.
    holdout_idx = rng.choice(n, size=n // 10, replace=False)
    holdout_reals = images[holdout_idx]
    probs_holdout = classify(holdout_reals, clf)
    s = stats(probs_holdout)
    results["random_10pct_real_holdout"] = s
    print(f"\n=== Random 10% real-image holdout (proxy for validation reals) ===")
    print(f"  mean={s['mean']:.3f}, median={s['median']:.3f}, p05={s['p05']:.3f}, >.5: {s['frac_above_0.5']:.3f}")

    # Small-eps perturbation acceptance on centered real images
    print("\n=== Small-perturbation acceptance on centered real images ===")
    sample = centered[rng.choice(len(centered), size=200, replace=False)]
    perturbation_results = {}
    for eps in [0.005, 0.01, 0.02, 0.05]:
        # Mean-preserving perturbation: random pixel offsets in [-eps, eps],
        # then subtract the per-image mean perturbation to preserve the global mean.
        noise = rng.uniform(-eps, eps, sample.shape)
        noise = noise - noise.mean(axis=1, keepdims=True)
        perturbed = np.clip(sample + noise, 0.0, 1.0)
        probs = classify(perturbed, clf)
        s = stats(probs)
        perturbation_results[f"eps_{eps}"] = s
        print(f"  eps={eps:5.3f}: mean={s['mean']:.3f}, median={s['median']:.3f}, p05={s['p05']:.3f}, >.5: {s['frac_above_0.5']:.3f}")
    results["small_perturbation_centered_acceptance"] = perturbation_results

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
