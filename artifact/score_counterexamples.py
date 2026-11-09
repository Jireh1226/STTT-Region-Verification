"""
Score counterexamples (pixel-box, mean-constrained, and AE-composed) with the
trained plausibility classifier, and compare to real-image scores as baseline.

Output: a table showing classifier logit / probability for each counterexample,
giving a "does this look real?" number.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import numpy as np
import h5py
from maraboupy import Marabou

NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
CLF_WEIGHTS = "plausibility_classifier.npz"
OUTPUT_JSON = "counterexample_plausibility_scores.json"
NUM_PIXELS = 128


# ----- Load classifier and helpers -----

def load_classifier_weights():
    w = np.load(CLF_WEIGHTS)
    # Keras weights: (in, out)
    W1, b1 = w["c1_W"], w["c1_b"]
    W2, b2 = w["c2_W"], w["c2_b"]
    W3, b3 = w["score_W"], w["score_b"]
    return [(W1, b1), (W2, b2), (W3, b3)]


def score(x, clf_layers):
    """x: shape (N, 128) or (128,). Returns (N,) logits."""
    if x.ndim == 1:
        x = x[None, :]
    h = x
    for i, (W, b) in enumerate(clf_layers):
        h = h @ W + b
        if i < len(clf_layers) - 1:
            h = np.maximum(h, 0)   # ReLU
    return h.flatten()


def prob_from_logit(logit):
    # Avoid overflow for very negative logits
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))


# ----- Re-extract pixel counterexamples via Marabou -----

def compute_box(imgs):
    flat = imgs.reshape(imgs.shape[0], -1)
    return flat.min(axis=0), flat.max(axis=0)


def run_pixel_box_query(kind, pmins, pmaxs, mean_constraint=False):
    net = Marabou.read_nnet(NNET_PATH)
    input_vars = net.inputVars[0].flatten()
    for i in range(NUM_PIXELS):
        net.setLowerBound(int(input_vars[i]), float(pmins[i]))
        net.setUpperBound(int(input_vars[i]), float(pmaxs[i]))
    if mean_constraint:
        net.addEquality([int(v) for v in input_vars], [1.0] * NUM_PIXELS, 64.0)
    if kind == "P1a":
        net.setLowerBound(192, 10.0)
    elif kind == "P1b":
        net.setUpperBound(192, -10.0)
    elif kind == "P4_C_left":
        net.setUpperBound(192, 0.0)
    elif kind == "P4_C_right":
        net.setLowerBound(192, 0.0)
    elif kind == "P5":
        net.setLowerBound(192, -0.01); net.setUpperBound(192, 0.01)
        net.setLowerBound(193, -0.01); net.setUpperBound(193, 0.01)

    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=300)
    status, vals, _ = net.solve(verbose=False, options=opts)
    status = status.strip().lower()
    if status != "sat":
        return None
    pix = np.array([vals[int(input_vars[i])] for i in range(NUM_PIXELS)], dtype=np.float64)
    return pix


# ----- Decode AE latent to pixel vector -----

def decode_latent(latent, subset):
    data = np.load(f"autoencoder_{subset}_weights.npz")
    layers = []
    for name in ["dec1", "dec2", "pixels_out"]:
        layers.append((data[f"{name}_W"], data[f"{name}_b"]))
    h = np.asarray(latent)
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    return h


# ----- Main scoring -----

def main():
    print("Loading real images...")
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].reshape(-1, NUM_PIXELS)
        labels = f["X_train"][:]
    cte = labels[:, 0]

    print("Loading classifier...")
    clf = load_classifier_weights()

    # Baseline: score all 10k real images
    all_logits = score(images, clf)
    print(f"\nReal image score distribution (baseline):")
    print(f"  logit mean/median/std: {all_logits.mean():.3f} / {np.median(all_logits):.3f} / {all_logits.std():.3f}")
    print(f"  logit 5th/95th percentile: {np.percentile(all_logits, 5):.3f} / {np.percentile(all_logits, 95):.3f}")
    print(f"  fraction scored as real (prob > 0.5): {(prob_from_logit(all_logits) > 0.5).mean():.3f}")

    # Subsets
    centered_mask = np.abs(cte) < 1.0
    left_mask = cte > 2.0
    right_mask = cte < -2.0
    offctr_mask = np.abs(cte) > 1.0

    centered_pmins, centered_pmaxs = compute_box(images[centered_mask])
    left_pmins, left_pmaxs = compute_box(images[left_mask])
    right_pmins, right_pmaxs = compute_box(images[right_mask])
    offctr_pmins, offctr_pmaxs = compute_box(images[offctr_mask])

    results = {
        "baseline": {
            "source": "all 10,000 real images",
            "logit_mean": float(all_logits.mean()),
            "logit_median": float(np.median(all_logits)),
            "logit_5th_percentile": float(np.percentile(all_logits, 5)),
            "logit_95th_percentile": float(np.percentile(all_logits, 95)),
            "fraction_scored_real": float((prob_from_logit(all_logits) > 0.5).mean()),
        },
        "witnesses": {},
    }

    # Per-subset baseline (real images restricted to the subset)
    for subset_name, mask in [("centered", centered_mask), ("left", left_mask),
                              ("right", right_mask), ("off_center", offctr_mask)]:
        subset_logits = score(images[mask], clf)
        results["baseline"][f"{subset_name}_subset_logit_mean"] = float(subset_logits.mean())
        results["baseline"][f"{subset_name}_subset_logit_median"] = float(np.median(subset_logits))

    # ----- Pixel-box witnesses (re-extract via Marabou) -----
    print("\n--- Re-extracting pixel-box witnesses ---")
    pixel_box_queries = [
        ("P1a_pixel_box",   "P1a",       centered_pmins, centered_pmaxs, False),
        ("P1b_pixel_box",   "P1b",       centered_pmins, centered_pmaxs, False),
        ("P4_left_pixel_box",  "P4_C_left",  left_pmins,  left_pmaxs,  False),
        ("P4_right_pixel_box", "P4_C_right", right_pmins, right_pmaxs, False),
        ("P5_pixel_box",    "P5",        offctr_pmins, offctr_pmaxs, False),
        ("P1a_mean",        "P1a",       centered_pmins, centered_pmaxs, True),
        ("P1b_mean",        "P1b",       centered_pmins, centered_pmaxs, True),
        ("P4_left_mean",    "P4_C_left", left_pmins,  left_pmaxs, True),
        ("P4_right_mean",   "P4_C_right",right_pmins, right_pmaxs, True),
        ("P5_mean",         "P5",        offctr_pmins, offctr_pmaxs, True),
    ]

    for name, kind, pmins, pmaxs, mean_c in pixel_box_queries:
        print(f"  [{name}]", end=" ", flush=True)
        pix = run_pixel_box_query(kind, pmins, pmaxs, mean_c)
        if pix is None:
            print("(no SAT witness)")
            continue
        logit = float(score(pix, clf)[0])
        prob = float(prob_from_logit(logit))
        pix_mean = float(pix.mean())
        print(f"  logit={logit:+.2f}  prob={prob:.3f}  pix_mean={pix_mean:.3f}")
        results["witnesses"][name] = {
            "logit": logit, "prob": prob, "pixel_mean": pix_mean,
        }

    # ----- AE-composed witnesses -----
    # Load stored latent witnesses
    print("\n--- Scoring AE-composed witnesses (decoded) ---")
    try:
        ae_results = json.load(open("verification_subset_autoencoder_onnx.json"))
    except FileNotFoundError:
        ae_results = {}

    ae_mapping = {
        "P4_C_left": "left",
        "P4_C_right": "right",
        "P5_deadzone": "off_center",
    }
    for prop_name, r in ae_results.items():
        if "ae_subset" not in r:
            continue
        ae_r = r["ae_subset"]
        if ae_r["status"] != "sat":
            continue
        latent = np.array(ae_r["counterexample"]["latent"])
        subset = ae_mapping.get(prop_name) or "centered"
        pix = decode_latent(latent, subset)
        logit = float(score(pix, clf)[0])
        prob = float(prob_from_logit(logit))
        pix_mean = float(pix.mean())
        print(f"  [{prop_name}_AE_{subset}]  logit={logit:+.2f}  prob={prob:.3f}  pix_mean={pix_mean:.3f}")
        results["witnesses"][f"{prop_name}_AE_{subset}"] = {
            "logit": logit, "prob": prob, "pixel_mean": pix_mean,
            "latent": latent.tolist(),
        }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    # ----- Summary table -----
    print("\n" + "=" * 88)
    print("PLAUSIBILITY CLASSIFIER SCORES FOR ALL COUNTEREXAMPLES")
    print("=" * 88)
    print(f"Baseline (real images):  logit mean={results['baseline']['logit_mean']:+.2f}, "
          f"5th pct={results['baseline']['logit_5th_percentile']:+.2f}")
    print(f"(Positive logit = classifier thinks it's real; negative = thinks it's fake)")
    print()
    print(f"{'Witness':<28} {'Logit':>8} {'Prob':>7} {'Verdict':<12}")
    print("-" * 70)
    for name, w in results["witnesses"].items():
        verdict = ("plausible" if w["logit"] > 0
                   else "borderline" if w["logit"] > -2
                   else "implausible")
        print(f"{name:<28} {w['logit']:+8.2f} {w['prob']:7.3f} {verdict:<12}")


if __name__ == "__main__":
    main()
