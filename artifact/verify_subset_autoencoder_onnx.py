"""
Plausibility-constrained verification using subset-specific autoencoders.

Mapping:
  P1a, P1b   -> composed_centered.onnx    (AE trained only on centered images)
  P4 C_left  -> composed_left.onnx        (AE trained only on left-offset images)
  P4 C_right -> composed_right.onnx       (AE trained only on right-offset images)
  P5 deadzone-> composed_off_center.onnx  (AE trained only on off-center images)

This eliminates cross-pollination: each composed network's decoder can only
produce images from that specific regime, so counterexamples cannot be blends
of centered and off-center images.
"""
import json
import time
import numpy as np
import h5py
from maraboupy import Marabou

NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "verification_subset_autoencoder_onnx.json"
NUM_PIXELS = 128


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].reshape(-1, NUM_PIXELS)
        labels = f["X_train"][:]
    return images, labels


def compute_box(images):
    flat = images.reshape(images.shape[0], -1)
    return flat.min(axis=0), flat.max(axis=0)


def solve_pixel_box(kind, pmins, pmaxs, mean_constraint=False):
    net = Marabou.read_nnet(NNET_PATH)
    input_vars = net.inputVars[0].flatten()
    for i in range(NUM_PIXELS):
        net.setLowerBound(int(input_vars[i]), float(pmins[i]))
        net.setUpperBound(int(input_vars[i]), float(pmaxs[i]))
    if mean_constraint:
        net.addEquality([int(v) for v in input_vars], [1.0] * NUM_PIXELS, 64.0)
    _apply_postcondition(net, kind, 192, 193)
    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=600)
    t0 = time.time()
    status, vals, _ = net.solve(verbose=False, options=opts)
    elapsed = time.time() - t0
    status = status.strip().lower()
    out = {"status": status, "time_seconds": round(elapsed, 4)}
    if status == "sat":
        pix = [vals[int(input_vars[i])] for i in range(NUM_PIXELS)]
        out["counterexample"] = {
            "CTE": round(float(vals[192]), 6),
            "HE":  round(float(vals[193]), 6),
            "input_mean": round(float(np.mean(pix)), 6),
        }
    return out


def solve_ae_composed(onnx_path, kind, z_min, z_max):
    net = Marabou.read_onnx(onnx_path)
    input_vars = np.array(net.inputVars[0]).flatten()
    output_vars = np.array(net.outputVars[0]).flatten()
    cte_var = int(output_vars[0])
    he_var = int(output_vars[1])

    for i in range(len(input_vars)):
        net.setLowerBound(int(input_vars[i]), float(z_min[i]))
        net.setUpperBound(int(input_vars[i]), float(z_max[i]))

    _apply_postcondition(net, kind, cte_var, he_var)

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


def _apply_postcondition(net, kind, cte_var, he_var):
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


def main():
    print("Loading data...")
    images, labels = load_data()
    cte = labels[:, 0]

    # Subset AEs: load latent bounding boxes
    ae_info = {}
    for subset in ["centered", "left", "right", "off_center"]:
        data = np.load(f"autoencoder_{subset}_weights.npz")
        ae_info[subset] = {
            "latent_min": data["latent_min"],
            "latent_max": data["latent_max"],
        }

    centered_pmins, centered_pmaxs = compute_box(images[np.abs(cte) < 1.0])
    left_pmins, left_pmaxs = compute_box(images[cte > 2.0])
    right_pmins, right_pmaxs = compute_box(images[cte < -2.0])
    offctr_pmins, offctr_pmaxs = compute_box(images[np.abs(cte) > 1.0])

    queries = [
        # (name, kind, pixel box for baseline, onnx path, subset key for latent box)
        ("P1a_CTE_upper", "P1a", centered_pmins, centered_pmaxs, "composed_centered.onnx", "centered"),
        ("P1b_CTE_lower", "P1b", centered_pmins, centered_pmaxs, "composed_centered.onnx", "centered"),
        ("P4_C_left",     "P4_C_left",  left_pmins,  left_pmaxs,  "composed_left.onnx",     "left"),
        ("P4_C_right",    "P4_C_right", right_pmins, right_pmaxs, "composed_right.onnx",    "right"),
        ("P5_deadzone",   "P5",         offctr_pmins, offctr_pmaxs, "composed_off_center.onnx", "off_center"),
    ]

    results = {}
    for name, kind, pmins, pmaxs, onnx, subset_key in queries:
        print(f"\n=== {name} (AE subset: {subset_key}) ===")
        results[name] = {"onnx": onnx, "subset": subset_key}

        print("  [pixel box]            ", end=" ", flush=True)
        r = solve_pixel_box(kind, pmins, pmaxs)
        print(f"{r['status']} ({r['time_seconds']:.2f}s)")
        results[name]["pixel_box"] = r

        print("  [pixel box + mean=0.5] ", end=" ", flush=True)
        r = solve_pixel_box(kind, pmins, pmaxs, mean_constraint=True)
        print(f"{r['status']} ({r['time_seconds']:.2f}s)")
        results[name]["mean_only"] = r

        zmin = ae_info[subset_key]["latent_min"]
        zmax = ae_info[subset_key]["latent_max"]
        print("  [AE subset-specific]   ", end=" ", flush=True)
        r = solve_ae_composed(onnx, kind, zmin, zmax)
        print(f"{r['status']} ({r['time_seconds']:.2f}s)")
        if r["status"] == "sat":
            cx = r["counterexample"]
            print(f"    CTE={cx['CTE']:.3f}, HE={cx['HE']:.3f}")
            print(f"    latent: {cx['latent']}")
        results[name]["ae_subset"] = r

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Query':<20} {'Pixel box':<14} {'+ mean':<14} {'AE (subset-specific)':<22}")
    print("-" * 80)
    for name in results:
        r = results[name]
        if r["ae_subset"]["status"] == "sat":
            cx = r["ae_subset"]["counterexample"]
            ae_detail = f"sat CTE={cx['CTE']:.2f}"
        else:
            ae_detail = r["ae_subset"]["status"]
        print(f"{name:<20} {r['pixel_box']['status']:<14} {r['mean_only']['status']:<14} {ae_detail:<22}")


if __name__ == "__main__":
    main()
