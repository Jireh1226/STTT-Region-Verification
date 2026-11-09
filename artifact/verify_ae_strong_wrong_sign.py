"""
Probe how strong the P4 wrong-sign failures are under mean-constrained
AE verification. Sweeps the CTE threshold to find the strongest witness:
  - CTE >= 0    (any sign error)
  - CTE >= 0.5  (must be at least 0.5m wrong-sign)
  - CTE >= 1.0  (at least 1m wrong-sign)
  - CTE >= 2.0  (at least 2m wrong-sign)
  - CTE >= 3.0  (matches previous CTE=+3.25 finding?)
"""
import json
import time
import numpy as np
from maraboupy import Marabou

NUM_PIXELS = 128


def solve(onnx_path, threshold, direction, z_min, z_max, enforce_mean=True):
    """
    direction: 'upper' -> find witness with CTE >= threshold
               'lower' -> find witness with CTE <= -threshold
    """
    net = Marabou.read_onnx(onnx_path)
    input_vars = np.array(net.inputVars[0]).flatten()
    output_vars = np.array(net.outputVars[0]).flatten()
    cte_var = int(output_vars[0])
    he_var = int(output_vars[1])
    sum_var = int(output_vars[2])

    for i in range(len(input_vars)):
        net.setLowerBound(int(input_vars[i]), float(z_min[i]))
        net.setUpperBound(int(input_vars[i]), float(z_max[i]))

    if enforce_mean:
        net.setLowerBound(sum_var, 64.0)
        net.setUpperBound(sum_var, 64.0)

    if direction == 'upper':
        net.setLowerBound(cte_var, float(threshold))
    else:
        net.setUpperBound(cte_var, float(-threshold))

    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=300)
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
    subset_data = {}
    for subset in ["left", "right", "off_center"]:
        data = np.load(f"autoencoder_{subset}_weights.npz")
        subset_data[subset] = {
            "latent_min": data["latent_min"],
            "latent_max": data["latent_max"],
        }

    thresholds = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    results = {"with_mean_0.5": {}, "without_mean": {}}

    for enforce_mean in [True, False]:
        mode_key = "with_mean_0.5" if enforce_mean else "without_mean"
        print(f"\n{'='*70}")
        print(f"Sweep: {'mean=0.5 enforced' if enforce_mean else 'no mean constraint'}")
        print('='*70)
        for prop, subset, direction in [
            ("P4_C_right (CTE > 0 req)", "right", "upper"),
            ("P4_C_left  (CTE < 0 req)", "left", "lower"),
        ]:
            print(f"\n  {prop}")
            results[mode_key][prop] = {}
            onnx = f"composed_{subset}_with_sum.onnx"
            z_min = subset_data[subset]["latent_min"]
            z_max = subset_data[subset]["latent_max"]
            for thr in thresholds:
                r = solve(onnx, thr, direction, z_min, z_max, enforce_mean=enforce_mean)
                if r["status"] == "sat":
                    cx = r["counterexample"]
                    cte_display = cx["CTE"]
                    violation = f"CTE={cte_display:+.3f}"
                else:
                    violation = r["status"]
                sign = "≥" if direction == "upper" else "≤"
                print(f"    |CTE| {sign} {thr:.1f}:   {r['status']:<5}  {violation}  ({r['time_seconds']:.1f}s)")
                results[mode_key][prop][f"threshold_{thr}"] = r

    with open("verification_strong_wrong_sign_sweep.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
