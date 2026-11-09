"""
Final auditing pass on every math-region SAT witness:
  - Plausibility-classifier probability
  - L2 distance to nearest real centered image
  - Math-region property post-hoc (so we can verify what each witness satisfies)

Scans every released math-region JSON, extracts SAT witnesses that carry a
full 128-vector witness, scores each, and reports the distribution alongside
the existing PCA-witness and real-image baselines.

Output: math_witnesses_classifier_audit.json
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import glob
import json
from datetime import datetime, timezone

import h5py
import numpy as np


DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
CLF_WEIGHTS = "plausibility_classifier.npz"
NUM_PIXELS = 128


def load_classifier():
    w = np.load(CLF_WEIGHTS)
    return [(w["c1_W"], w["c1_b"]), (w["c2_W"], w["c2_b"]), (w["score_W"], w["score_b"])]


def score_clf(x, layers):
    if x.ndim == 1:
        x = x[None, :]
    h = x
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    return float(h.flatten()[0])


def prob_from_logit(logit):
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))


def collect_witnesses(json_path):
    """Walk a results JSON and yield (label, witness_vec) for each SAT witness with
    a stored full 128-vector."""
    with open(json_path) as f:
        d = json.load(f)
    found = []

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            if (obj.get('status') == 'sat'
                    and 'witness' in obj
                    and isinstance(obj['witness'], list)
                    and len(obj['witness']) == NUM_PIXELS):
                found.append((prefix.strip(' .'), np.array(obj['witness'])))
                return
            for k, v in obj.items():
                walk(v, prefix + "." + str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, prefix + f"[{i}]")
    walk(d)
    return found


def tv(img_flat):
    img = img_flat.reshape(8, 16)
    return float(np.abs(np.diff(img, axis=1)).sum() + np.abs(np.diff(img, axis=0)).sum())


def main():
    print("Loading classifier, dataset...", flush=True)
    clf = load_classifier()
    with h5py.File(DATA_PATH, "r") as f:
        all_images = f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float64)
        labels = f["X_train"][:]
    cte_true = labels[:, 0]
    centered_idx = np.where(np.abs(cte_true) < 1.0)[0]
    centered = all_images[centered_idx]
    print(f"  N_total={len(all_images)} N_centered={len(centered)}", flush=True)

    # Real-image baseline (centered subset)
    real_logits = np.array([score_clf(x, clf) for x in centered[::20]])  # sample
    real_probs = 1.0 / (1.0 + np.exp(-np.clip(real_logits, -30, 30)))
    print(f"  Real-centered baseline (n={len(real_probs)}): "
          f"prob median={np.median(real_probs):.3f}, mean={real_probs.mean():.3f}",
          flush=True)

    # Walk each known math-region JSON
    json_files = [
        "verification_tv_bound.json",
        "verification_fourier_bound.json",
        "verification_photometric_bound.json",
        "verification_per_row_mean.json",
        "verification_math_combined.json",
        "verification_p4_right_strong_math.json",
        "verification_ae_plus_math.json",
        "verification_object_specific_math.json",
    ]

    out = {
        "provenance": {
            "script": "audit_math_witnesses.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "classifier_weights": CLF_WEIGHTS,
        },
        "real_image_baseline": {
            "n_sampled": int(len(real_probs)),
            "prob_median": round(float(np.median(real_probs)), 4),
            "prob_mean": round(float(real_probs.mean()), 4),
        },
        "per_file": {},
        "all_witnesses": [],
    }

    grand_probs = []
    grand_l2s = []
    for jf in json_files:
        if not os.path.exists(jf):
            print(f"  [skip] {jf} not found", flush=True)
            continue
        witnesses = collect_witnesses(jf)
        if not witnesses:
            print(f"  [skip] {jf}: no SAT witness with full pixel vector", flush=True)
            out["per_file"][jf] = {"n_witnesses": 0}
            continue
        print(f"\n  === {jf} ({len(witnesses)} witnesses) ===", flush=True)
        per_file_probs = []
        per_file_l2 = []
        per_file_records = []
        for label, w in witnesses:
            logit = score_clf(w, clf)
            prob = prob_from_logit(logit)
            diffs = centered - w
            l2 = np.linalg.norm(diffs, axis=1)
            nn = int(np.argmin(l2))
            l2_nn = float(l2[nn])
            tvw = tv(w)
            rec = {
                "label": label[:80],
                "classifier_prob": round(prob, 4),
                "classifier_logit": round(logit, 4),
                "l2_to_nearest_centered": round(l2_nn, 4),
                "witness_tv": round(tvw, 4),
                "witness_mean": round(float(w.mean()), 4),
            }
            per_file_probs.append(prob)
            per_file_l2.append(l2_nn)
            per_file_records.append(rec)
            grand_probs.append(prob)
            grand_l2s.append(l2_nn)
            out["all_witnesses"].append({"source_file": jf, **rec})
        per_file_probs = np.array(per_file_probs)
        per_file_l2 = np.array(per_file_l2)
        print(f"    classifier prob:  median={np.median(per_file_probs):.3f} mean={per_file_probs.mean():.3f} "
              f"min={per_file_probs.min():.3f} max={per_file_probs.max():.3f}", flush=True)
        print(f"    L2 to nearest:    median={np.median(per_file_l2):.3f} mean={per_file_l2.mean():.3f}", flush=True)
        out["per_file"][jf] = {
            "n_witnesses": len(witnesses),
            "classifier_prob": {
                "median": round(float(np.median(per_file_probs)), 4),
                "mean": round(float(per_file_probs.mean()), 4),
                "min": round(float(per_file_probs.min()), 4),
                "max": round(float(per_file_probs.max()), 4),
            },
            "l2_to_nearest_centered": {
                "median": round(float(np.median(per_file_l2)), 4),
                "mean": round(float(per_file_l2.mean()), 4),
            },
            "witnesses": per_file_records,
        }

    # Grand summary
    if grand_probs:
        gp = np.array(grand_probs)
        gl = np.array(grand_l2s)
        n_below_05 = int((gp < 0.5).sum())
        n_below_01 = int((gp < 0.1).sum())
        n_above_09 = int((gp > 0.9).sum())
        out["aggregate"] = {
            "n_total_witnesses_audited": int(len(gp)),
            "classifier_prob": {
                "median": round(float(np.median(gp)), 4),
                "mean": round(float(gp.mean()), 4),
                "min": round(float(gp.min()), 4),
                "max": round(float(gp.max()), 4),
                "n_below_0.5": n_below_05,
                "n_below_0.1": n_below_01,
                "n_above_0.9": n_above_09,
            },
            "l2_to_nearest_centered": {
                "median": round(float(np.median(gl)), 4),
                "mean": round(float(gl.mean()), 4),
            },
        }
        print("\n=== AGGREGATE SUMMARY ===", flush=True)
        print(f"  total math-region witnesses scored: {len(gp)}", flush=True)
        print(f"  classifier prob (real): median={np.median(gp):.3f}, "
              f"mean={gp.mean():.3f}, range [{gp.min():.3f}, {gp.max():.3f}]", flush=True)
        print(f"  L2 to nearest real centered image: median={np.median(gl):.3f}, "
              f"mean={gl.mean():.3f}", flush=True)
        print(f"  witnesses scored 'classifier-real' (prob > 0.5): {gp.size - n_below_05}/{gp.size}", flush=True)
        print(f"  witnesses scored 'classifier-fake' (prob < 0.1): {n_below_01}/{gp.size}", flush=True)
        print(f"  witnesses scored 'classifier-confident-real' (prob > 0.9): {n_above_09}/{gp.size}", flush=True)
        print(f"  real-centered baseline (n={len(real_probs)}): median {np.median(real_probs):.3f}, "
              f"mean {real_probs.mean():.3f}", flush=True)

    with open("math_witnesses_classifier_audit.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved math_witnesses_classifier_audit.json", flush=True)


if __name__ == "__main__":
    main()
