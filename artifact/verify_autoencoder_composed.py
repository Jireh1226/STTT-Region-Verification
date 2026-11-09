"""
Plausibility-constrained verification via ReLU autoencoder composition.

Composes the trained decoder (z -> pixel image) with TinyTaxiNet and runs
Marabou 2.0 on the composed network. Marabou searches over the low-dimensional
latent space `z` rather than the 128-D per-pixel box. Every input it considers
is by construction a decoder-generated image, which approximates the training
data manifold nonlinearly (what PCA could not do).

Method (SEVIN-style, per Habeeb et al. ICCPS 2025; Bak et al. 2024):
  input: z in [z_min, z_max]^LATENT_DIM  (bounding box of training latents)
  -> Decoder: z -> dec1 = ReLU(W1 z + b1) -> dec2 = ReLU(W2 dec1 + b2)
            -> pixels = W3 dec2 + b3    (linear output)
  -> TinyTaxiNet: pixels -> (CTE, HE)
  Add negated postcondition on (CTE, HE).

Applied to P1a, P1b, P4 C_left/C_right, P5. For each property, four modes:
  (a) unconstrained:      pixel box only (baseline)
  (b) mean_only:          pixel box + mean(x)=0.5
  (c) ae_full_range:      decoder over the full latent bounding box of training
  (d) ae_per_subset_box:  decoder over the latent box of images in the property's subset
"""
import json
import time
import numpy as np
import h5py
from maraboupy import Marabou

NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
AE_WEIGHTS = "autoencoder_weights.npz"
OUTPUT_JSON = "verification_autoencoder_composed.json"

NUM_PIXELS = 128


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].reshape(-1, NUM_PIXELS)
        labels = f["X_train"][:]
    return images, labels


def compute_box(images):
    flat = images.reshape(images.shape[0], -1)
    return flat.min(axis=0), flat.max(axis=0)


def load_decoder():
    w = np.load(AE_WEIGHTS)
    layers = []
    for name in ["dec1", "dec2", "pixels_out"]:
        W = w[f"{name}_W"]                  # Keras shape: (in, out)
        b = w[f"{name}_b"]
        layers.append((W, b))
    latents = w["latents"]
    latent_min = w["latent_min"]
    latent_max = w["latent_max"]
    return layers, latents, latent_min, latent_max


def add_decoder_and_tie(net, input_vars_net, decoder_layers, latent_dim):
    """Add decoder z -> pixels to Marabou network. Returns the latent variables.
    The decoder's output is tied to TinyTaxiNet's input variables via equality.
    """
    # Create latent variables
    latent_vars = [net.getNewVariable() for _ in range(latent_dim)]

    # Forward through decoder
    curr_vars = latent_vars
    for li, (W, b) in enumerate(decoder_layers):
        in_dim, out_dim = W.shape
        is_last = (li == len(decoder_layers) - 1)
        pre_vars = [net.getNewVariable() for _ in range(out_dim)]

        # Linear: pre[j] = sum_i W[i,j] * curr[i] + b[j]
        # As equality: pre[j] - sum_i W[i,j] * curr[i] = b[j]
        for j in range(out_dim):
            vars_list = [pre_vars[j]] + list(curr_vars)
            coeffs = [1.0] + [-float(W[i, j]) for i in range(in_dim)]
            net.addEquality(vars_list, coeffs, float(b[j]))

        if is_last:
            # Linear output (no ReLU)
            curr_vars = pre_vars
        else:
            # Apply ReLU
            post_vars = [net.getNewVariable() for _ in range(out_dim)]
            for j in range(out_dim):
                net.addRelu(pre_vars[j], post_vars[j])
            curr_vars = post_vars

    # Tie decoder output to TinyTaxiNet input
    assert len(curr_vars) == NUM_PIXELS
    for i in range(NUM_PIXELS):
        # curr_vars[i] == input_vars_net[i]  =>  curr_vars[i] - input_vars_net[i] = 0
        net.addEquality([curr_vars[i], int(input_vars_net[i])], [1.0, -1.0], 0.0)

    return latent_vars


def set_latent_box(net, latent_vars, z_min, z_max):
    for i, v in enumerate(latent_vars):
        net.setLowerBound(v, float(z_min[i]))
        net.setUpperBound(v, float(z_max[i]))


def setup_base_net():
    net = Marabou.read_nnet(NNET_PATH)
    input_vars = net.inputVars[0].flatten()
    return net, input_vars


def apply_output_constraint(net, kind):
    """Apply negated postcondition to output variables (192=CTE, 193=HE)."""
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


def solve_unconstrained(property_kind, pmins, pmaxs, decoder_layers=None, latent_dim=None, z_min=None, z_max=None, mean_constraint=False):
    """Run a single query. If decoder_layers given, compose with decoder."""
    net, input_vars = setup_base_net()

    if decoder_layers is None:
        # Classical pixel-box query
        for i in range(NUM_PIXELS):
            net.setLowerBound(int(input_vars[i]), float(pmins[i]))
            net.setUpperBound(int(input_vars[i]), float(pmaxs[i]))
        if mean_constraint:
            net.addEquality([int(v) for v in input_vars], [1.0] * NUM_PIXELS, 64.0)
    else:
        # Decoder-composed query: input is the latent, pixels are computed by decoder.
        # TinyTaxiNet's pixel inputs are forced to equal the decoder output via
        # equality constraints inside add_decoder_and_tie. We do NOT add the pixel
        # box here - the decoder's output is already restricted to the manifold of
        # images it was trained on (training images have pixel values roughly in
        # [0,1] post-preprocessing), so constraining further risks infeasibility
        # due to reconstruction error. We add a loose [0,1] safety bound only.
        latent_vars = add_decoder_and_tie(net, input_vars, decoder_layers, latent_dim)
        set_latent_box(net, latent_vars, z_min, z_max)
        for i in range(NUM_PIXELS):
            net.setLowerBound(int(input_vars[i]), 0.0)
            net.setUpperBound(int(input_vars[i]), 1.0)

    apply_output_constraint(net, property_kind)

    options = Marabou.createOptions(verbosity=0, timeoutInSeconds=600)
    t0 = time.time()
    status, vals, _ = net.solve(verbose=False, options=options)
    elapsed = time.time() - t0
    status = status.strip().lower()

    result = {
        "status": status,
        "time_seconds": round(elapsed, 4),
    }

    if status == "sat":
        result["counterexample"] = {
            "CTE": round(float(vals[192]), 6),
            "HE":  round(float(vals[193]), 6),
        }
        # If decoder-composed, also record the latent that produced the witness
        if decoder_layers is not None:
            result["counterexample"]["latent"] = [
                round(float(vals[int(v)]), 6) for v in latent_vars
            ]
        pixel_values = np.array([vals[int(input_vars[i])] for i in range(NUM_PIXELS)])
        result["counterexample"]["input_mean"] = round(float(pixel_values.mean()), 6)
        result["counterexample"]["input_min"] = round(float(pixel_values.min()), 6)
        result["counterexample"]["input_max"] = round(float(pixel_values.max()), 6)

    return result


def latent_box_of_subset(encoder_latents, mask):
    sub = encoder_latents[mask]
    return sub.min(axis=0), sub.max(axis=0)


def main():
    print("Loading data...")
    images, labels = load_data()
    cte_labels = labels[:, 0]

    print("Loading trained autoencoder...")
    decoder_layers, all_latents, latent_min, latent_max = load_decoder()
    latent_dim = all_latents.shape[1]
    print(f"  Latent dim: {latent_dim}")
    print(f"  Latent min (all training): {latent_min.tolist()}")
    print(f"  Latent max (all training): {latent_max.tolist()}")

    # Subsets
    centered_mask = np.abs(cte_labels) < 1.0
    left_mask = cte_labels > 2.0
    right_mask = cte_labels < -2.0
    offctr_mask = np.abs(cte_labels) > 1.0

    centered_latent_min, centered_latent_max = latent_box_of_subset(all_latents, centered_mask)
    left_latent_min, left_latent_max = latent_box_of_subset(all_latents, left_mask)
    right_latent_min, right_latent_max = latent_box_of_subset(all_latents, right_mask)
    offctr_latent_min, offctr_latent_max = latent_box_of_subset(all_latents, offctr_mask)

    centered_pmins, centered_pmaxs = compute_box(images[centered_mask])
    left_pmins, left_pmaxs = compute_box(images[left_mask])
    right_pmins, right_pmaxs = compute_box(images[right_mask])
    offctr_pmins, offctr_pmaxs = compute_box(images[offctr_mask])

    queries = [
        # (name, subset_pmins/pmaxs, subset_latent_min/max, kind)
        ("P1a_CTE_upper", (centered_pmins, centered_pmaxs), (centered_latent_min, centered_latent_max), "P1a"),
        ("P1b_CTE_lower", (centered_pmins, centered_pmaxs), (centered_latent_min, centered_latent_max), "P1b"),
        ("P4_C_left",    (left_pmins, left_pmaxs),         (left_latent_min, left_latent_max),         "P4_C_left"),
        ("P4_C_right",   (right_pmins, right_pmaxs),       (right_latent_min, right_latent_max),       "P4_C_right"),
        ("P5_deadzone",  (offctr_pmins, offctr_pmaxs),     (offctr_latent_min, offctr_latent_max),     "P5"),
    ]

    all_results = {
        "autoencoder_info": {
            "latent_dim": int(latent_dim),
            "latent_min_all": latent_min.tolist(),
            "latent_max_all": latent_max.tolist(),
        },
        "queries": {},
    }

    for name, (pmins, pmaxs), (zmin_sub, zmax_sub), kind in queries:
        print(f"\n=== {name} ===")
        all_results["queries"][name] = {}

        # (a) Unconstrained pixel box
        print("  [unconstrained pixel box]", end=" ", flush=True)
        r = solve_unconstrained(kind, pmins, pmaxs)
        print(f"{r['status']} ({r['time_seconds']:.2f}s)")
        all_results["queries"][name]["unconstrained"] = r

        # (b) + mean=0.5
        print("  [pixel box + mean=0.5]   ", end=" ", flush=True)
        r = solve_unconstrained(kind, pmins, pmaxs, mean_constraint=True)
        print(f"{r['status']} ({r['time_seconds']:.2f}s)")
        all_results["queries"][name]["mean_only"] = r

        # (c) Decoder over full-dataset latent range
        print("  [decoder, full-data latent box]   ", end=" ", flush=True)
        r = solve_unconstrained(kind, pmins, pmaxs, decoder_layers=decoder_layers,
                                latent_dim=latent_dim, z_min=latent_min, z_max=latent_max)
        print(f"{r['status']} ({r['time_seconds']:.2f}s)")
        all_results["queries"][name]["ae_full_range"] = r

        # (d) Decoder over subset-specific latent range
        print("  [decoder, subset latent box]     ", end=" ", flush=True)
        r = solve_unconstrained(kind, pmins, pmaxs, decoder_layers=decoder_layers,
                                latent_dim=latent_dim, z_min=zmin_sub, z_max=zmax_sub)
        print(f"{r['status']} ({r['time_seconds']:.2f}s)")
        all_results["queries"][name]["ae_subset_range"] = r

    print(f"\nSaving to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary table
    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    print(f"{'Query':<18} {'Pixel box':<14} {'+ mean':<14} {'+ AE (full)':<14} {'+ AE (subset)':<14}")
    print("-" * 96)
    for name in all_results["queries"]:
        r = all_results["queries"][name]
        print(f"{name:<18} "
              f"{r['unconstrained']['status']:<14} "
              f"{r['mean_only']['status']:<14} "
              f"{r['ae_full_range']['status']:<14} "
              f"{r['ae_subset_range']['status']:<14}")


if __name__ == "__main__":
    main()
