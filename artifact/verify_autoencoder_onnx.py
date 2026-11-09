"""
Plausibility-constrained verification via composed ONNX network.

Loads composed_decoder_tinytaxi.onnx (a single feedforward ReLU network
z -> decoder -> TinyTaxiNet -> (CTE, HE)) into Marabou and runs queries
over the latent box defined by training data encodings.

For comparison, also runs the original per-pixel box queries without the
plausibility constraint.
"""
import json
import time
import numpy as np
import h5py
from maraboupy import Marabou

ONNX_PATH = "composed_decoder_tinytaxi.onnx"
NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
AE_WEIGHTS = "autoencoder_weights.npz"
OUTPUT_JSON = "verification_autoencoder_onnx.json"

NUM_PIXELS = 128


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].reshape(-1, NUM_PIXELS)
        labels = f["X_train"][:]
    return images, labels


def compute_box(images):
    flat = images.reshape(images.shape[0], -1)
    return flat.min(axis=0), flat.max(axis=0)


# -------- Pixel-box baseline queries (for comparison) --------

def solve_pixel_box(kind, pmins, pmaxs, mean_constraint=False):
    net = Marabou.read_nnet(NNET_PATH)
    input_vars = net.inputVars[0].flatten()
    for i in range(NUM_PIXELS):
        net.setLowerBound(int(input_vars[i]), float(pmins[i]))
        net.setUpperBound(int(input_vars[i]), float(pmaxs[i]))
    if mean_constraint:
        net.addEquality([int(v) for v in input_vars], [1.0] * NUM_PIXELS, 64.0)
    apply_postcondition_pixel(net, kind)
    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=600)
    t0 = time.time()
    status, vals, _ = net.solve(verbose=False, options=opts)
    elapsed = time.time() - t0
    status = status.strip().lower()
    out = {"status": status, "time_seconds": round(elapsed, 4)}
    if status == "sat":
        out["counterexample"] = {
            "CTE": round(float(vals[192]), 6),
            "HE":  round(float(vals[193]), 6),
            "input_mean": round(float(np.mean([vals[int(input_vars[i])] for i in range(NUM_PIXELS)])), 6),
        }
    return out


def apply_postcondition_pixel(net, kind):
    """Postcondition negation on TinyTaxiNet alone (outputs 192, 193)."""
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


# -------- Autoencoder-composed queries --------

def solve_ae_composed(kind, z_min, z_max):
    net = Marabou.read_onnx(ONNX_PATH)
    # Input is an (1, 8) tensor in ONNX; inputVars[0] is shape (1, 8)
    input_vars = np.array(net.inputVars[0]).flatten()   # 8 latent vars
    output_vars = np.array(net.outputVars[0]).flatten() # 2: [CTE_var, HE_var]
    cte_var = int(output_vars[0])
    he_var = int(output_vars[1])

    for i in range(len(input_vars)):
        net.setLowerBound(int(input_vars[i]), float(z_min[i]))
        net.setUpperBound(int(input_vars[i]), float(z_max[i]))

    # Postcondition on composed-network outputs
    if kind == "P1a":
        net.setLowerBound(cte_var, 10.0)
    elif kind == "P1b":
        net.setUpperBound(cte_var, -10.0)
    elif kind == "P4_C_left":
        net.setUpperBound(cte_var, 0.0)
    elif kind == "P4_C_right":
        net.setLowerBound(cte_var, 0.0)
    elif kind == "P5":
        net.setLowerBound(cte_var, -0.01); net.setUpperBound(cte_var, 0.01)
        net.setLowerBound(he_var, -0.01); net.setUpperBound(he_var, 0.01)

    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=600)
    t0 = time.time()
    status, vals, _ = net.solve(verbose=False, options=opts)
    elapsed = time.time() - t0
    status = status.strip().lower()
    out = {"status": status, "time_seconds": round(elapsed, 4)}
    if status == "sat":
        out["counterexample"] = {
            "CTE": round(float(vals[cte_var]), 6),
            "HE":  round(float(vals[he_var]), 6),
            "latent": [round(float(vals[int(v)]), 6) for v in input_vars],
        }
    return out


def main():
    print("Loading data and autoencoder info...")
    images, labels = load_data()
    cte_labels = labels[:, 0]

    ae = np.load(AE_WEIGHTS)
    all_latents = ae['latents']
    latent_full_min = ae['latent_min']
    latent_full_max = ae['latent_max']
    print(f"  Full latent box (training data): min={latent_full_min.tolist()}")
    print(f"                                    max={latent_full_max.tolist()}")

    # Subset masks
    centered_mask = np.abs(cte_labels) < 1.0
    left_mask = cte_labels > 2.0
    right_mask = cte_labels < -2.0
    offctr_mask = np.abs(cte_labels) > 1.0

    # Per-subset latent boxes
    centered_z = all_latents[centered_mask]
    left_z = all_latents[left_mask]
    right_z = all_latents[right_mask]
    offctr_z = all_latents[offctr_mask]

    # Per-subset pixel boxes
    centered_pmins, centered_pmaxs = compute_box(images[centered_mask])
    left_pmins, left_pmaxs = compute_box(images[left_mask])
    right_pmins, right_pmaxs = compute_box(images[right_mask])
    offctr_pmins, offctr_pmaxs = compute_box(images[offctr_mask])

    queries = [
        ("P1a_CTE_upper", "P1a", centered_pmins, centered_pmaxs,
         centered_z.min(axis=0), centered_z.max(axis=0)),
        ("P1b_CTE_lower", "P1b", centered_pmins, centered_pmaxs,
         centered_z.min(axis=0), centered_z.max(axis=0)),
        ("P4_C_left", "P4_C_left", left_pmins, left_pmaxs,
         left_z.min(axis=0), left_z.max(axis=0)),
        ("P4_C_right", "P4_C_right", right_pmins, right_pmaxs,
         right_z.min(axis=0), right_z.max(axis=0)),
        ("P5_deadzone", "P5", offctr_pmins, offctr_pmaxs,
         offctr_z.min(axis=0), offctr_z.max(axis=0)),
    ]

    results = {}
    for name, kind, pmins, pmaxs, zmin_sub, zmax_sub in queries:
        print(f"\n=== {name} ===")
        results[name] = {}

        print("  [pixel box]            ", end=" ", flush=True)
        r = solve_pixel_box(kind, pmins, pmaxs)
        print(f"{r['status']} ({r['time_seconds']:.2f}s)")
        results[name]["pixel_box"] = r

        print("  [pixel box + mean=0.5] ", end=" ", flush=True)
        r = solve_pixel_box(kind, pmins, pmaxs, mean_constraint=True)
        print(f"{r['status']} ({r['time_seconds']:.2f}s)")
        results[name]["mean_only"] = r

        print("  [AE latent, full data] ", end=" ", flush=True)
        r = solve_ae_composed(kind, latent_full_min, latent_full_max)
        print(f"{r['status']} ({r['time_seconds']:.2f}s)")
        if r["status"] == "sat":
            print(f"    latent witness: {r['counterexample']['latent']}")
        results[name]["ae_full"] = r

        print("  [AE latent, subset]    ", end=" ", flush=True)
        r = solve_ae_composed(kind, zmin_sub, zmax_sub)
        print(f"{r['status']} ({r['time_seconds']:.2f}s)")
        if r["status"] == "sat":
            print(f"    latent witness: {r['counterexample']['latent']}")
        results[name]["ae_subset"] = r

    print(f"\nSaving to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump({
            "latent_full_min": latent_full_min.tolist(),
            "latent_full_max": latent_full_max.tolist(),
            "queries": results,
        }, f, indent=2)

    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    print(f"{'Query':<20} {'Pixel box':<12} {'+ mean':<12} {'AE full':<12} {'AE subset':<12}")
    print("-" * 96)
    for name in results:
        r = results[name]
        print(f"{name:<20} "
              f"{r['pixel_box']['status']:<12} "
              f"{r['mean_only']['status']:<12} "
              f"{r['ae_full']['status']:<12} "
              f"{r['ae_subset']['status']:<12}")


if __name__ == "__main__":
    main()
