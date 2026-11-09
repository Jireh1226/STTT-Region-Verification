"""
True held-out calibration audit for the plausibility classifier.

Uses plausibility_classifier_holdout.npz (trained by
train_plausibility_classifier_holdout.py on a 90% real-image split, fakes
derived from training reals only) and scores:

  1. Acceptance on the 1,000 genuinely unseen real images, overall and per
     CTE subset (centered / left / right / off-center within the holdout).
  2. Mean-preserving-noise acceptance on unseen centered reals.
  3. Re-score of every stored math-region SAT witness (the 91-witness
     population of math_witnesses_classifier_audit.json) with the retrained
     classifier, to confirm the rejection rate is not an artifact of
     training-set memorization in the original classifier.

Outputs:
  classifier_true_holdout_audit.json
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
from datetime import datetime, timezone

import h5py
import numpy as np

DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
WEIGHTS_PATH = "plausibility_classifier_holdout.npz"
OUTPUT_JSON = "classifier_true_holdout_audit.json"

MATH_WITNESS_FILES = [
    "verification_tv_bound.json",
    "verification_fourier_bound.json",
    "verification_photometric_bound.json",
    "verification_per_row_mean.json",
    "verification_math_combined.json",
    "verification_p4_right_strong_math.json",
    "verification_ae_plus_math.json",
    "verification_object_specific_math.json",
]

NUM_PIXELS = 128


def load_classifier():
    w = np.load(WEIGHTS_PATH)
    layers = [(w["c1_W"], w["c1_b"]),
              (w["c2_W"], w["c2_b"]),
              (w["score_W"], w["score_b"])]
    return layers, w["holdout_idx"]


def classify(X, layers):
    h = X
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    logit = np.clip(h.flatten(), -30, 30)
    return 1.0 / (1.0 + np.exp(-logit))


def stats(probs):
    return {
        "n": int(len(probs)),
        "mean": float(probs.mean()),
        "median": float(np.median(probs)),
        "p05": float(np.percentile(probs, 5)),
        "frac_above_0.5": float((probs > 0.5).mean()),
        "frac_above_0.9": float((probs > 0.9).mean()),
        "frac_below_0.1": float((probs < 0.1).mean()),
    }


def collect_witnesses(json_path):
    found = []

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            if (obj.get('status') == 'sat'
                    and 'witness' in obj
                    and isinstance(obj['witness'], list)
                    and len(obj['witness']) == NUM_PIXELS):
                found.append((json_path + ":" + prefix.strip(' .'),
                              np.array(obj['witness'], dtype=np.float64)))
            else:
                for k, v in obj.items():
                    walk(v, prefix + "." + str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, prefix + f"[{i}]")

    if os.path.exists(json_path):
        walk(json.load(open(json_path)))
    return found


def main():
    rng = np.random.default_rng(42)
    layers, holdout_idx = load_classifier()
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float64)
        labels = f["X_train"][:].astype(np.float64)
    cte = labels[:, 0]

    hold_imgs = images[holdout_idx]
    hold_cte = cte[holdout_idx]
    results = {
        "provenance": {
            "script": "audit_classifier_true_holdout.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "classifier": WEIGHTS_PATH,
            "note": "holdout reals never seen in training, directly or via fakes",
        },
        "unseen_real_acceptance": {},
    }

    subsets = [
        ("all holdout reals", np.ones(len(holdout_idx), dtype=bool)),
        ("centered (|CTE|<1)", np.abs(hold_cte) < 1.0),
        ("left (CTE>2)", hold_cte > 2.0),
        ("right (CTE<-2)", hold_cte < -2.0),
        ("off-center (|CTE|>1)", np.abs(hold_cte) > 1.0),
    ]
    for name, mask in subsets:
        probs = classify(hold_imgs[mask], layers)
        s = stats(probs)
        results["unseen_real_acceptance"][name] = s
        print(f"  {name:24s}: n={s['n']:4d}  >.5: {s['frac_above_0.5']:.3f}  "
              f"median={s['median']:.3f}")

    # Mean-preserving noise on unseen centered reals
    centered_hold = hold_imgs[np.abs(hold_cte) < 1.0]
    n_pert = min(200, len(centered_hold))
    sample = centered_hold[rng.choice(len(centered_hold), size=n_pert,
                                      replace=False)]
    pert = {}
    for eps in [0.005, 0.01, 0.02, 0.05]:
        noise = rng.uniform(-eps, eps, sample.shape)
        noise = noise - noise.mean(axis=1, keepdims=True)
        probs = classify(np.clip(sample + noise, 0.0, 1.0), layers)
        s = stats(probs)
        pert[f"eps_{eps}"] = s
        print(f"  noise eps={eps:5.3f}: >.5: {s['frac_above_0.5']:.3f}")
    results["unseen_centered_perturbation_acceptance"] = pert

    # Re-score the stored math-region witnesses with the retrained classifier
    witnesses = []
    for jf in MATH_WITNESS_FILES:
        witnesses.extend(collect_witnesses(jf))
    if witnesses:
        W = np.stack([w for _, w in witnesses])
        probs = classify(W, layers)
        results["math_witness_rescoring"] = {
            "n_witnesses": int(len(probs)),
            "median": float(np.median(probs)),
            "mean": float(probs.mean()),
            "n_below_0.1": int((probs < 0.1).sum()),
            "n_above_0.5": int((probs > 0.5).sum()),
            "n_above_0.9": int((probs > 0.9).sum()),
        }
        print(f"  math witnesses: n={len(probs)}, "
              f"below 0.1: {(probs < 0.1).sum()}/{len(probs)}, "
              f"above 0.5: {(probs > 0.5).sum()}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
