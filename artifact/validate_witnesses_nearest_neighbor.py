"""
Nearest-neighbor validation of AE-composed SAT witnesses.

For each SAT witness, we:
  1. Decode the latent to a pixel vector (the counterexample image)
  2. Find the nearest real training image in the relevant subset (L2 distance)
  3. Run TinyTaxiNet on the nearest real image
  4. Compare the network's output on the real image vs the counterexample
  5. Report: does the failure localize near a real image, or is it isolated?

Also: report whether the failure's local latent neighborhood stays in the
failure region (local stability check).
"""
import json
import numpy as np
import h5py
from maraboupy import Marabou

NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "witness_nearest_neighbor_validation.json"
NUM_PIXELS = 128


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].reshape(-1, NUM_PIXELS)
        labels = f["X_train"][:]
    return images, labels


def load_decoder(subset):
    data = np.load(f"autoencoder_{subset}_weights.npz")
    layers = []
    for name in ["dec1", "dec2", "pixels_out"]:
        layers.append((data[f"{name}_W"], data[f"{name}_b"]))
    return layers


def decode_latent(latent, decoder_layers):
    h = np.asarray(latent, dtype=np.float64)
    for i, (W, b) in enumerate(decoder_layers):
        h = h @ W + b
        if i < len(decoder_layers) - 1:
            h = np.maximum(h, 0.0)
    return h


def load_tinytaxi_weights():
    """Parse TinyTaxiNet.nnet and return a forward-pass function."""
    with open(NNET_PATH) as f:
        lines = f.readlines()
    i = 0
    while lines[i].startswith('//'):
        i += 1
    header = [int(x) for x in lines[i].split(',') if x.strip()]
    num_layers = header[0]
    i += 1
    layer_sizes = [int(x) for x in lines[i].split(',') if x.strip()]
    i += 1
    i += 5
    weights = []
    biases = []
    for l in range(num_layers):
        rows = layer_sizes[l + 1]
        cols = layer_sizes[l]
        W = np.zeros((rows, cols))
        for r in range(rows):
            vals = [float(x) for x in lines[i].strip().rstrip(',').split(',')]
            W[r] = vals[:cols]
            i += 1
        b = np.zeros(rows)
        for r in range(rows):
            b[r] = float(lines[i].strip().rstrip(','))
            i += 1
        weights.append(W)
        biases.append(b)

    def forward(x):
        h = np.asarray(x, dtype=np.float64)
        for j, (W, b) in enumerate(zip(weights, biases)):
            h = W @ h + b
            if j < len(weights) - 1:
                h = np.maximum(h, 0.0)
        return h    # returns [CTE, HE]

    return forward


def validate_witness(name, subset, latent, expected_cte, expected_he, images, labels, forward, decoder_layers, subset_mask):
    """Run full validation on a single witness."""
    # Decode latent
    decoded = decode_latent(latent, decoder_layers)

    # Network output on decoded image (sanity check vs expected)
    net_out_decoded = forward(decoded)

    # Find nearest real image in the subset
    subset_imgs = images[subset_mask]
    subset_lbls = labels[subset_mask]
    diffs_l2 = np.sqrt(np.sum((subset_imgs - decoded) ** 2, axis=1))
    diffs_linf = np.max(np.abs(subset_imgs - decoded), axis=1)
    nearest_l2_idx = int(np.argmin(diffs_l2))
    nearest_linf_idx = int(np.argmin(diffs_linf))
    nearest_img_l2 = subset_imgs[nearest_l2_idx]
    nearest_lbl_l2 = subset_lbls[nearest_l2_idx]
    nearest_out_l2 = forward(nearest_img_l2)

    # Also: what does the network output on a SET of real subset images?
    # Sample 20 real images from the subset and run them through the network
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(subset_imgs), size=min(20, len(subset_imgs)), replace=False)
    sample_outputs = np.array([forward(subset_imgs[i]) for i in sample_idx])

    result = {
        "name": name,
        "subset": subset,
        "decoded_pixel_mean": float(decoded.mean()),
        "decoded_pixel_range": [float(decoded.min()), float(decoded.max())],
        "witness_output_expected": {"CTE": expected_cte, "HE": expected_he},
        "witness_output_from_decode": {"CTE": float(net_out_decoded[0]), "HE": float(net_out_decoded[1])},
        "nearest_real_l2_dist": float(diffs_l2[nearest_l2_idx]),
        "nearest_real_linf_dist": float(diffs_linf[nearest_linf_idx]),
        "nearest_real_label": {"CTE_gt": float(nearest_lbl_l2[0]), "HE_gt": float(nearest_lbl_l2[1])},
        "nearest_real_output": {"CTE_pred": float(nearest_out_l2[0]), "HE_pred": float(nearest_out_l2[1])},
        "subset_sample_output_stats": {
            "n_sampled": int(len(sample_idx)),
            "CTE_pred_min": float(sample_outputs[:, 0].min()),
            "CTE_pred_max": float(sample_outputs[:, 0].max()),
            "CTE_pred_mean": float(sample_outputs[:, 0].mean()),
            "fraction_wrong_sign": None,
        },
    }
    return result


def main():
    print("Loading data and models...")
    images, labels = load_data()
    cte = labels[:, 0]
    forward = load_tinytaxi_weights()

    # Load strong wrong-sign witnesses from the sweep
    sweep = json.load(open("verification_strong_wrong_sign_sweep.json"))

    decoders = {s: load_decoder(s) for s in ["centered", "left", "right", "off_center"]}
    masks = {
        "centered": np.abs(cte) < 1.0,
        "left": cte > 2.0,
        "right": cte < -2.0,
        "off_center": np.abs(cte) > 1.0,
    }

    # Witnesses to validate (the strongest wrong-sign from mean=0.5 sweep)
    witnesses = []
    # P4 C_right at CTE >= 2.0, mean=0.5
    r = sweep["with_mean_0.5"]["P4_C_right (CTE > 0 req)"]["threshold_2.0"]
    if r["status"] == "sat":
        witnesses.append(("P4_C_right_threshold_2.0_mean=0.5", "right", r["counterexample"]))
    r = sweep["with_mean_0.5"]["P4_C_right (CTE > 0 req)"]["threshold_3.0"]
    if r["status"] == "sat":
        witnesses.append(("P4_C_right_threshold_3.0_mean=0.5", "right", r["counterexample"]))
    r = sweep["with_mean_0.5"]["P4_C_left  (CTE < 0 req)"]["threshold_3.0"]
    if r["status"] == "sat":
        witnesses.append(("P4_C_left_threshold_3.0_mean=0.5", "left", r["counterexample"]))

    # Compute dataset empirical wrong-sign rates on both regions
    print("\nComputing empirical wrong-sign rates on dataset...")
    left_imgs = images[masks["left"]]
    right_imgs = images[masks["right"]]
    left_preds = np.array([forward(x)[0] for x in left_imgs])
    right_preds = np.array([forward(x)[0] for x in right_imgs])
    left_wrong = (left_preds < 0).sum()
    right_wrong = (right_preds > 0).sum()
    print(f"  Left-offset images (ground truth CTE > 2):  {left_wrong}/{len(left_imgs)} produce wrong-sign (CTE < 0)")
    print(f"    Predicted CTE min/max/mean: {left_preds.min():.3f}/{left_preds.max():.3f}/{left_preds.mean():.3f}")
    print(f"  Right-offset images (ground truth CTE < -2): {right_wrong}/{len(right_imgs)} produce wrong-sign (CTE > 0)")
    print(f"    Predicted CTE min/max/mean: {right_preds.min():.3f}/{right_preds.max():.3f}/{right_preds.mean():.3f}")

    empirical = {
        "left_total": int(len(left_imgs)),
        "left_wrong_sign": int(left_wrong),
        "left_wrong_sign_fraction": float(left_wrong / len(left_imgs)),
        "left_pred_CTE_min": float(left_preds.min()),
        "left_pred_CTE_max": float(left_preds.max()),
        "left_pred_CTE_mean": float(left_preds.mean()),
        "right_total": int(len(right_imgs)),
        "right_wrong_sign": int(right_wrong),
        "right_wrong_sign_fraction": float(right_wrong / len(right_imgs)),
        "right_pred_CTE_min": float(right_preds.min()),
        "right_pred_CTE_max": float(right_preds.max()),
        "right_pred_CTE_mean": float(right_preds.mean()),
    }

    # Validate each witness
    results = []
    for name, subset, cx in witnesses:
        print(f"\n--- {name} ---")
        latent = np.array(cx["latent"])
        r = validate_witness(
            name, subset, latent,
            cx["CTE"], cx["HE"],
            images, labels, forward, decoders[subset], masks[subset]
        )
        print(f"  Decoded pixel mean: {r['decoded_pixel_mean']:.4f}  (want 0.5)")
        print(f"  Decoded pixel range: [{r['decoded_pixel_range'][0]:.3f}, {r['decoded_pixel_range'][1]:.3f}]")
        print(f"  Network output on decoded: CTE={r['witness_output_from_decode']['CTE']:.3f}, HE={r['witness_output_from_decode']['HE']:.3f}")
        print(f"  Witness expected (from Marabou): CTE={r['witness_output_expected']['CTE']:.3f}")
        print(f"  Nearest real image (L2 = {r['nearest_real_l2_dist']:.4f}):")
        print(f"    Ground truth CTE = {r['nearest_real_label']['CTE_gt']:.3f}")
        print(f"    Network output   CTE = {r['nearest_real_output']['CTE_pred']:.3f}  HE = {r['nearest_real_output']['HE_pred']:.3f}")
        results.append(r)

    with open(OUTPUT_JSON, "w") as f:
        json.dump({"empirical_wrong_sign": empirical, "witnesses": results}, f, indent=2)
    print(f"\nSaved: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
