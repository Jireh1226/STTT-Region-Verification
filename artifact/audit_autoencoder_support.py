"""
Audit the subset autoencoders used for learned-support-constrained verification.

This script does not retrain or re-verify. It uses the saved subset AE latents
and decoders to measure:
  1. held-out reconstruction quality under the original 90/10 split,
  2. TinyTaxiNet output preservation under AE reconstruction,
  3. whether held-out latents would fall inside a train-only latent box,
  4. how loose the latent box is under random latent sampling,
  5. where AE SAT witnesses sit inside the latent box.

The goal is to distinguish a useful learned-support diagnostic from a
deployment-realism claim. Output: autoencoder_support_audit.json.
"""

import json
from pathlib import Path

import h5py
import numpy as np


DATA_PATH = Path("VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5")
NNET_PATH = Path("VerifyGAN/models/TinyTaxiNet.nnet")
OUTPUT_JSON = Path("autoencoder_support_audit.json")
NUM_PIXELS = 128
SEED = 42
RANDOM_LATENT_N = 2000


SUBSETS = {
    "centered": lambda cte: np.abs(cte) < 1.0,
    "left": lambda cte: cte > 2.0,
    "right": lambda cte: cte < -2.0,
    "off_center": lambda cte: np.abs(cte) > 1.0,
}


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float64)
        labels = f["X_train"][:].astype(np.float64)
    return images, labels


def load_nnet_weights():
    lines = NNET_PATH.read_text().splitlines()
    i = 0
    while lines[i].startswith("//"):
        i += 1
    header = [int(x) for x in lines[i].split(",") if x.strip()]
    num_layers = header[0]
    i += 1
    layer_sizes = [int(x) for x in lines[i].split(",") if x.strip()]
    i += 1
    i += 5

    weights = []
    biases = []
    for layer in range(num_layers):
        rows = layer_sizes[layer + 1]
        cols = layer_sizes[layer]
        W = np.zeros((rows, cols), dtype=np.float64)
        for r in range(rows):
            vals = [float(x) for x in lines[i].strip().rstrip(",").split(",")]
            W[r] = vals[:cols]
            i += 1
        b = np.zeros(rows, dtype=np.float64)
        for r in range(rows):
            b[r] = float(lines[i].strip().rstrip(","))
            i += 1
        weights.append(W)
        biases.append(b)
    return weights, biases


def forward_tinytaxi_batch(images, weights, biases):
    h = np.asarray(images, dtype=np.float64)
    for i, (W, b) in enumerate(zip(weights, biases)):
        h = h @ W.T + b
        if i < len(weights) - 1:
            h = np.maximum(h, 0.0)
    return h


def load_decoder(subset):
    data = np.load(f"autoencoder_{subset}_weights.npz")
    layers = []
    for name in ["dec1", "dec2", "pixels_out"]:
        layers.append((data[f"{name}_W"].astype(np.float64),
                       data[f"{name}_b"].astype(np.float64)))
    return data, layers


def decode_batch(latents, layers):
    h = np.asarray(latents, dtype=np.float64)
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0.0)
    return h


def load_classifier():
    path = Path("plausibility_classifier.npz")
    if not path.exists():
        return None
    data = np.load(path)
    return [
        (data["c1_W"].astype(np.float64), data["c1_b"].astype(np.float64)),
        (data["c2_W"].astype(np.float64), data["c2_b"].astype(np.float64)),
        (data["score_W"].astype(np.float64), data["score_b"].astype(np.float64)),
    ]


def score_classifier_batch(images, layers):
    if layers is None:
        return None
    h = np.asarray(images, dtype=np.float64)
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0.0)
    logits = h.reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
    return logits, probs


def stats(arr):
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def signed_rate(true_cte, pred_cte, side):
    if side == "left":
        wrong = pred_cte <= 0.0
    elif side == "right":
        wrong = pred_cte >= 0.0
    else:
        return None
    return float(np.mean(wrong))


def split_indices(n):
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    n_val = max(10, int(0.1 * n))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def fraction_inside_box(points, lo, hi):
    inside = np.all((points >= lo) & (points <= hi), axis=1)
    return float(np.mean(inside)), int(np.sum(inside)), int(len(points))


def latent_box_position(z, lo, hi):
    denom = np.maximum(hi - lo, 1e-12)
    pos = (np.asarray(z, dtype=np.float64) - lo) / denom
    edge_dist = np.minimum(pos, 1.0 - pos)
    return {
        "min_normalized_edge_distance": float(np.min(edge_dist)),
        "dims_within_1pct_edge": int(np.sum(edge_dist <= 0.01)),
        "dims_on_edge_1e-5": int(np.sum(edge_dist <= 1e-5)),
        "normalized_position": [float(x) for x in pos],
    }


def collect_witnesses():
    witnesses = []

    mean_path = Path("verification_ae_with_mean_constraint.json")
    if mean_path.exists():
        data = json.loads(mean_path.read_text())
        for name, record in data.items():
            subset = record.get("ae_subset")
            for mode in ["ae_with_mean", "ae_no_mean"]:
                cx = record.get(mode, {}).get("counterexample")
                if subset and cx and "latent" in cx:
                    witnesses.append({
                        "source": mean_path.name,
                        "name": f"{name}:{mode}",
                        "subset": subset,
                        "latent": cx["latent"],
                        "outputs": {"CTE": cx.get("CTE"), "HE": cx.get("HE")},
                        "pixel_sum": cx.get("pixel_sum"),
                    })

    sweep_path = Path("verification_strong_wrong_sign_sweep.json")
    if sweep_path.exists():
        data = json.loads(sweep_path.read_text())
        for group_name, group in data.get("with_mean_0.5", {}).items():
            subset = "right" if "right" in group_name else "left" if "left" in group_name else None
            for threshold, record in group.items():
                cx = record.get("counterexample")
                if subset and cx and "latent" in cx:
                    witnesses.append({
                        "source": sweep_path.name,
                        "name": f"{group_name}:{threshold}",
                        "subset": subset,
                        "latent": cx["latent"],
                        "outputs": {"CTE": cx.get("CTE"), "HE": cx.get("HE")},
                        "pixel_sum": cx.get("pixel_sum"),
                    })

    return witnesses


def main():
    images, labels = load_data()
    cte = labels[:, 0]
    weights, biases = load_nnet_weights()
    classifier = load_classifier()
    rng = np.random.default_rng(SEED)

    audit = {
        "script": "audit_autoencoder_support.py",
        "purpose": "AE support audit: reconstruction, output preservation, latent-box coverage, random-latent controls, witness boundary positions",
        "seed": SEED,
        "random_latent_n": RANDOM_LATENT_N,
        "subsets": {},
        "witnesses": [],
    }

    witnesses = collect_witnesses()

    for subset, mask_fn in SUBSETS.items():
        mask = mask_fn(cte)
        subset_images = images[mask]
        subset_labels = labels[mask]
        ae_data, decoder = load_decoder(subset)
        latents = ae_data["latents"].astype(np.float64)
        z_min_all = ae_data["latent_min"].astype(np.float64)
        z_max_all = ae_data["latent_max"].astype(np.float64)

        recon = decode_batch(latents, decoder)
        err = subset_images - recon
        pixel_linf = np.max(np.abs(err), axis=1)
        pixel_mse = np.mean(err ** 2, axis=1)

        train_idx, val_idx = split_indices(len(subset_images))
        z_min_train = latents[train_idx].min(axis=0)
        z_max_train = latents[train_idx].max(axis=0)
        val_inside_frac, val_inside_count, val_n = fraction_inside_box(
            latents[val_idx], z_min_train, z_max_train
        )

        out_real = forward_tinytaxi_batch(subset_images, weights, biases)
        out_recon = forward_tinytaxi_batch(recon, weights, biases)
        out_abs_delta = np.abs(out_real - out_recon)

        clf_real = score_classifier_batch(subset_images, classifier)
        clf_recon = score_classifier_batch(recon, classifier)

        random_z = rng.uniform(z_min_all, z_max_all, size=(RANDOM_LATENT_N, len(z_min_all)))
        random_decodes = decode_batch(random_z, decoder)
        random_outputs = forward_tinytaxi_batch(random_decodes, weights, biases)
        clf_random = score_classifier_batch(random_decodes, classifier)
        random_means = random_decodes.mean(axis=1)
        random_in_pixel_range = np.all((random_decodes >= 0.0) & (random_decodes <= 1.0), axis=1)

        subset_record = {
            "n_images": int(len(subset_images)),
            "heldout_split": {
                "train_n": int(len(train_idx)),
                "heldout_n": int(len(val_idx)),
                "heldout_recon_linf": stats(pixel_linf[val_idx]),
                "heldout_recon_mse": stats(pixel_mse[val_idx]),
                "heldout_latents_inside_train_only_box": {
                    "fraction": val_inside_frac,
                    "inside": val_inside_count,
                    "total": val_n,
                    "note": "Current verification uses all-subset latent min/max; this reports what would happen if bounds were computed from the train split only.",
                },
            },
            "all_subset_reconstruction": {
                "recon_linf": stats(pixel_linf),
                "recon_mse": stats(pixel_mse),
            },
            "tinytaxinet_output_preservation": {
                "abs_delta_cte": stats(out_abs_delta[:, 0]),
                "abs_delta_he": stats(out_abs_delta[:, 1]),
                "cte_output_sign_changed_fraction_abs_real_cte_gt_0.5": float(
                    np.mean(
                        np.sign(out_real[np.abs(out_real[:, 0]) > 0.5, 0])
                        != np.sign(out_recon[np.abs(out_real[:, 0]) > 0.5, 0])
                    )
                ) if np.any(np.abs(out_real[:, 0]) > 0.5) else None,
                "p1_threshold_crossing_counts": {
                    "real_abs_cte_ge_10": int(np.sum(np.abs(out_real[:, 0]) >= 10.0)),
                    "recon_abs_cte_ge_10": int(np.sum(np.abs(out_recon[:, 0]) >= 10.0)),
                    "real_abs_he_ge_90": int(np.sum(np.abs(out_real[:, 1]) >= 90.0)),
                    "recon_abs_he_ge_90": int(np.sum(np.abs(out_recon[:, 1]) >= 90.0)),
                },
                "p4_wrong_sign_rate_real_if_applicable": signed_rate(subset_labels[:, 0], out_real[:, 0], subset),
                "p4_wrong_sign_rate_recon_if_applicable": signed_rate(subset_labels[:, 0], out_recon[:, 0], subset),
            },
            "classifier_scores": {},
            "random_latent_box_controls": {
                "decoded_pixel_mean": stats(random_means),
                "fraction_all_pixels_in_0_1": float(np.mean(random_in_pixel_range)),
                "fraction_mean_within_0.01_of_0.5": float(np.mean(np.abs(random_means - 0.5) <= 0.01)),
                "tinytaxinet_output_cte": stats(random_outputs[:, 0]),
                "tinytaxinet_output_he": stats(random_outputs[:, 1]),
            },
        }

        if clf_real is not None:
            real_logits, real_probs = clf_real
            recon_logits, recon_probs = clf_recon
            random_logits, random_probs = clf_random
            subset_record["classifier_scores"] = {
                "real_prob": stats(real_probs),
                "reconstruction_prob": stats(recon_probs),
                "random_latent_decode_prob": stats(random_probs),
                "real_fraction_prob_gt_0.5": float(np.mean(real_probs > 0.5)),
                "reconstruction_fraction_prob_gt_0.5": float(np.mean(recon_probs > 0.5)),
                "random_latent_decode_fraction_prob_gt_0.5": float(np.mean(random_probs > 0.5)),
            }

        audit["subsets"][subset] = subset_record

        for w in [w for w in witnesses if w["subset"] == subset]:
            z = np.asarray(w["latent"], dtype=np.float64)
            decoded = decode_batch(z.reshape(1, -1), decoder)[0]
            out = forward_tinytaxi_batch(decoded.reshape(1, -1), weights, biases)[0]
            clf = score_classifier_batch(decoded.reshape(1, -1), classifier)
            witness_record = {
                "source": w["source"],
                "name": w["name"],
                "subset": subset,
                "reported_outputs": w["outputs"],
                "decoded_outputs": {"CTE": float(out[0]), "HE": float(out[1])},
                "decoded_pixel_mean": float(decoded.mean()),
                "decoded_pixel_range": [float(decoded.min()), float(decoded.max())],
                "latent_box_position": latent_box_position(z, z_min_all, z_max_all),
            }
            if clf is not None:
                witness_record["classifier_prob_real"] = float(clf[1][0])
                witness_record["classifier_logit"] = float(clf[0][0])
            audit["witnesses"].append(witness_record)

    OUTPUT_JSON.write_text(json.dumps(audit, indent=2))

    print(f"Wrote {OUTPUT_JSON}")
    print("\nKey summary:")
    for subset, rec in audit["subsets"].items():
        heldout = rec["heldout_split"]["heldout_recon_linf"]
        cte_delta = rec["tinytaxinet_output_preservation"]["abs_delta_cte"]
        clf = rec.get("classifier_scores", {})
        random_frac = rec["random_latent_box_controls"]["fraction_all_pixels_in_0_1"]
        print(
            f"  {subset:<10} heldout Linf median={heldout['median']:.3f}, "
            f"p95={heldout['p95']:.3f}; CTE delta median={cte_delta['median']:.3f}, "
            f"p95={cte_delta['p95']:.3f}; random latent in [0,1]={random_frac:.3f}; "
            f"recon clf>0.5={clf.get('reconstruction_fraction_prob_gt_0.5', float('nan')):.3f}"
        )


if __name__ == "__main__":
    main()
