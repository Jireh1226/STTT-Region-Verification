"""
Plausibility-constrained verification with mean(x)=0.5 enforced on decoded pixels.

Uses composed ONNX networks that expose pixel_sum as a third output, allowing
us to add sum(pixels) = 64 as a hard constraint in the Marabou query.

This addresses the critique that our previous AE-composed queries didn't enforce
the known preprocessing invariant. Now Marabou must find counterexamples where
the decoded image satisfies BOTH:
  - z in [z_min, z_max] (latent bounding box)
  - sum(decoded_pixels) = 64  (mean = 0.5 preprocessing invariant)
"""
import json
import time
import numpy as np
import h5py
from maraboupy import Marabou

DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "verification_ae_with_mean_constraint.json"
NUM_PIXELS = 128

# Mapping property -> AE subset
AE_MAPPING = {
    "P1a": "centered",
    "P1b": "centered",
    "P4_C_left": "left",
    "P4_C_right": "right",
    "P5": "off_center",
}


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].reshape(-1, NUM_PIXELS)
        labels = f["X_train"][:]
    return images, labels


def solve_ae_with_mean(onnx_path, kind, z_min, z_max, enforce_mean=True):
    net = Marabou.read_onnx(onnx_path)
    input_vars = np.array(net.inputVars[0]).flatten()
    output_vars = np.array(net.outputVars[0]).flatten()
    # Output layout: [CTE, HE, pixel_sum]
    cte_var = int(output_vars[0])
    he_var = int(output_vars[1])
    sum_var = int(output_vars[2])

    # Latent bounds
    for i in range(len(input_vars)):
        net.setLowerBound(int(input_vars[i]), float(z_min[i]))
        net.setUpperBound(int(input_vars[i]), float(z_max[i]))

    # Mean-of-pixels constraint: sum = 64  ⇔  sum ∈ [64, 64]
    if enforce_mean:
        net.setLowerBound(sum_var, 64.0)
        net.setUpperBound(sum_var, 64.0)

    # Postcondition negation on outputs
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
            "pixel_sum": round(float(vals[sum_var]), 6),
            "latent": [round(float(vals[int(v)]), 6) for v in input_vars],
        }
    return out


def main():
    print("Loading data and subset AE info...")
    images, labels = load_data()
    cte = labels[:, 0]

    subset_data = {}
    for subset in ["centered", "left", "right", "off_center"]:
        data = np.load(f"autoencoder_{subset}_weights.npz")
        subset_data[subset] = {
            "latent_min": data["latent_min"],
            "latent_max": data["latent_max"],
        }

    queries = [
        ("P1a_CTE_upper", "P1a"),
        ("P1b_CTE_lower", "P1b"),
        ("P4_C_left",     "P4_C_left"),
        ("P4_C_right",    "P4_C_right"),
        ("P5_deadzone",   "P5"),
    ]

    results = {}
    for name, kind in queries:
        subset = AE_MAPPING[kind]
        onnx_path = f"composed_{subset}_with_sum.onnx"
        z_min = subset_data[subset]["latent_min"]
        z_max = subset_data[subset]["latent_max"]

        print(f"\n=== {name} (AE: {subset}) ===")
        results[name] = {"ae_subset": subset}

        # (a) AE subset, no mean constraint (reproduce prior result)
        print("  [AE subset, no mean]       ", end=" ", flush=True)
        r = solve_ae_with_mean(onnx_path, kind, z_min, z_max, enforce_mean=False)
        msg = r["status"]
        if r["status"] == "sat":
            cx = r["counterexample"]
            msg += f"  CTE={cx['CTE']:+.3f}, pixel_sum={cx['pixel_sum']:.3f}"
        print(f"{msg} ({r['time_seconds']:.2f}s)")
        results[name]["ae_no_mean"] = r

        # (b) AE subset + mean=0.5 on decoded pixels
        print("  [AE subset + mean=0.5]     ", end=" ", flush=True)
        r = solve_ae_with_mean(onnx_path, kind, z_min, z_max, enforce_mean=True)
        msg = r["status"]
        if r["status"] == "sat":
            cx = r["counterexample"]
            msg += f"  CTE={cx['CTE']:+.3f}, pixel_sum={cx['pixel_sum']:.3f}"
        print(f"{msg} ({r['time_seconds']:.2f}s)")
        results[name]["ae_with_mean"] = r

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUTPUT_JSON}")

    print("\n" + "=" * 80)
    print("SUMMARY: Effect of adding mean(x)=0.5 to AE-composed queries")
    print("=" * 80)
    print(f"{'Query':<18} {'AE alone':<24} {'AE + mean(x)=0.5':<24}")
    print("-" * 80)
    for name in results:
        r = results[name]
        def fmt(mode):
            q = r[mode]
            if q["status"] == "sat":
                cx = q["counterexample"]
                return f"sat CTE={cx['CTE']:+.2f}"
            return q["status"]
        print(f"{name:<18} {fmt('ae_no_mean'):<24} {fmt('ae_with_mean'):<24}")


if __name__ == "__main__":
    main()
