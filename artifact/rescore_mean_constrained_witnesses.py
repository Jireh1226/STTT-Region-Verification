"""
Score the mean-constrained threshold-sweep witnesses with the plausibility
classifier (and include the unconstrained AE witnesses for comparison).

The earlier score_counterexamples.py run only looked at the no-mean AE
witnesses. This script scores the mean-constrained threshold-sweep witnesses
that appear in the P4 table.
"""
import json
import numpy as np


def load_classifier_layers():
    w = np.load("plausibility_classifier.npz")
    return [(w["c1_W"], w["c1_b"]),
            (w["c2_W"], w["c2_b"]),
            (w["score_W"], w["score_b"])]


def score_logits(x, layers):
    if x.ndim == 1:
        x = x[None, :]
    h = x
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    return h.flatten()


def decode_latent(latent, subset):
    data = np.load(f"autoencoder_{subset}_weights.npz")
    layers = [(data[f"{n}_W"], data[f"{n}_b"]) for n in ["dec1", "dec2", "pixels_out"]]
    h = np.asarray(latent, dtype=np.float64)
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0.0)
    return h


def main():
    clf = load_classifier_layers()

    # Threshold-sweep witnesses (mean=0.5 enforced, the ones that made it to the P4 table)
    sweep = json.load(open("verification_strong_wrong_sign_sweep.json"))

    targets = [
        ("P4_C_right_thr=3.0_mean=0.5", "right",
         sweep["with_mean_0.5"]["P4_C_right (CTE > 0 req)"]["threshold_3.0"]),
        ("P4_C_left_thr=3.0_mean=0.5",  "left",
         sweep["with_mean_0.5"]["P4_C_left  (CTE < 0 req)"]["threshold_3.0"]),
        ("P4_C_right_thr=2.0_mean=0.5", "right",
         sweep["with_mean_0.5"]["P4_C_right (CTE > 0 req)"]["threshold_2.0"]),
    ]

    # Also include the boundary witnesses (just CTE >= 0 / <= 0 with mean=0.5)
    base = json.load(open("verification_ae_with_mean_constraint.json"))
    for name, rec in base.items():
        subset_key = rec.get("ae_subset")
        if subset_key is None or "ae_with_mean" not in rec:
            continue
        q = rec["ae_with_mean"]
        if q["status"] != "sat":
            continue
        targets.append((f"{name}_base_mean=0.5", subset_key, q))

    # Score each
    print(f"{'Witness':<40} {'logit':>8} {'pix_mean':>10} {'prob':>7}")
    print("-" * 70)
    results = []
    for name, subset, r in targets:
        cx = r["counterexample"]
        latent = np.array(cx["latent"])
        pix = decode_latent(latent, subset)
        logit = float(score_logits(pix, clf)[0])
        prob = 1.0 / (1.0 + np.exp(-max(min(logit, 30), -30)))
        pix_mean = float(pix.mean())
        print(f"{name:<40} {logit:+8.2f} {pix_mean:10.4f} {prob:7.3f}")
        results.append({
            "witness": name,
            "subset": subset,
            "CTE": cx.get("CTE"),
            "pixel_mean_decoded": round(pix_mean, 4),
            "logit": round(logit, 3),
            "prob": round(prob, 3),
        })

    with open("mean_constrained_witness_scores.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: mean_constrained_witness_scores.json")


if __name__ == "__main__":
    main()
