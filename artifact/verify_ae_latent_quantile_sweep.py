"""
Latent-box quantile sweep on AE-composed verification.

The current paper baseline (verify_ae_with_mean_constraint.py) defines the AE
verification region as the per-dimension axis-aligned bounding box of all
training-set latents, i.e. [min_i, max_i] per latent dimension i, then composes
the subset-specific decoder with TinyTaxiNet and enforces sum(decoded_pixels)=64
(mean=0.5 preprocessing invariant).

The support audit (autoencoder_support_audit.json) found that AE SAT witnesses
tend to sit on the latent-box boundary (several dims pinned to an edge). This
script rebuilds the same AE-composed queries with progressively tighter
quantile-based latent boxes:

    * min/max         (current paper baseline; reproduced as sanity check)
    * 0.5% / 99.5%
    * 1%   / 99%
    * 5%   / 95%

For each (quantile setting, property) we record SAT/UNSAT verdict, CTE/HE,
the 8-D latent witness, which latent dimensions sit within 5% of the bounding
box edge, and solver time. Results are saved to
verification_ae_latent_quantile_sweep.json. The closing summary highlights
whether each property's verdict changed across quantile settings.

This script does NOT modify any other file; it only depends on the existing
composed_<subset>_with_sum.onnx networks and the autoencoder_<subset>_weights.npz
archives.
"""
import json
import os
import subprocess
import time
from datetime import datetime, timezone

import numpy as np
from maraboupy import Marabou


# --- Configuration ---------------------------------------------------------

OUTPUT_JSON = "verification_ae_latent_quantile_sweep.json"
TIMEOUT_S = 300  # match existing AE verification scripts
EDGE_FRACTION = 0.05  # "within 5% of box edge"

# Property -> AE subset mapping (matches verify_ae_with_mean_constraint.py)
AE_MAPPING = {
    "P1a": "centered",
    "P1b": "centered",
    "P4_C_left": "left",
    "P4_C_right": "right",
    "P5": "off_center",
}

QUERIES = [
    ("P1a_CTE_upper", "P1a"),
    ("P1b_CTE_lower", "P1b"),
    ("P4_C_left",     "P4_C_left"),
    ("P4_C_right",    "P4_C_right"),
    ("P5_deadzone",   "P5"),
]

# (label, low_q, high_q) where None means use raw min/max
QUANTILE_SETTINGS = [
    ("min_max",       None,   None),
    ("q0p5_q99p5",    0.005,  0.995),
    ("q1_q99",        0.01,   0.99),
    ("q5_q95",        0.05,   0.95),
]


# --- Helpers ---------------------------------------------------------------

def get_latent_bounds(latents, low_q, high_q):
    """Per-dimension latent bounds. low_q=high_q=None reproduces raw min/max."""
    if low_q is None or high_q is None:
        return latents.min(axis=0), latents.max(axis=0)
    return (np.quantile(latents, low_q, axis=0),
            np.quantile(latents, high_q, axis=0))


def edge_proximity(latent_witness, z_min, z_max, frac=EDGE_FRACTION):
    """For each latent dim, normalized distance to nearest edge of [z_min,z_max].
    Returns list of (dim_index, distance_to_nearest_edge_fraction_of_width).
    Also returns list of dims within `frac` of an edge."""
    z_min = np.asarray(z_min, dtype=np.float64)
    z_max = np.asarray(z_max, dtype=np.float64)
    z = np.asarray(latent_witness, dtype=np.float64)
    width = z_max - z_min
    width_safe = np.where(width > 0, width, 1.0)
    dist_lo = (z - z_min) / width_safe
    dist_hi = (z_max - z) / width_safe
    nearest = np.minimum(dist_lo, dist_hi)
    within = [int(i) for i, d in enumerate(nearest) if d <= frac]
    return {
        "normalized_distance_to_nearest_edge": [round(float(x), 6) for x in nearest],
        "dims_within_5pct_of_edge": within,
        "n_dims_within_5pct_of_edge": len(within),
    }


def solve_one(onnx_path, kind, z_min, z_max, timeout_s=TIMEOUT_S):
    """Run a single AE-composed verification query and return a result dict.
    Mirrors verify_ae_with_mean_constraint.solve_ae_with_mean with enforce_mean=True.
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

    # mean(x)=0.5 -> sum=64
    net.setLowerBound(sum_var, 64.0)
    net.setUpperBound(sum_var, 64.0)

    # Postcondition negation on outputs (must match verify_ae_with_mean_constraint.py)
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
    else:
        raise ValueError(f"Unknown property kind: {kind}")

    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=timeout_s)
    t0 = time.time()
    try:
        status, vals, _ = net.solve(verbose=False, options=opts)
    except Exception as e:
        elapsed = time.time() - t0
        return {"status": "error", "error_message": str(e),
                "time_seconds": round(elapsed, 4)}
    elapsed = time.time() - t0
    status = status.strip().lower()

    out = {"status": status, "time_seconds": round(elapsed, 4)}
    if status == "sat":
        latent_w = [float(vals[int(v)]) for v in input_vars]
        out["counterexample"] = {
            "CTE":        round(float(vals[cte_var]), 6),
            "HE":         round(float(vals[he_var]), 6),
            "pixel_sum":  round(float(vals[sum_var]), 6),
            "latent":     [round(x, 6) for x in latent_w],
        }
        out["edge_proximity"] = edge_proximity(latent_w, z_min, z_max)
    return out


def git_commit_short():
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


# --- Baseline reproduction guard ------------------------------------------

# Verdicts from verification_ae_with_mean_constraint.json (ae_with_mean) — the
# current paper baseline. The min/max quantile setting MUST reproduce these or
# we stop before continuing.
BASELINE_AE_WITH_MEAN = {
    "P1a_CTE_upper": "unsat",
    "P1b_CTE_lower": "unsat",
    "P4_C_left":     "sat",
    "P4_C_right":    "sat",
    "P5_deadzone":   "sat",
}


# --- Main ------------------------------------------------------------------

def main():
    print("Loading subset AE latents...", flush=True)
    subset_data = {}
    for subset in ["centered", "left", "right", "off_center"]:
        d = np.load(f"autoencoder_{subset}_weights.npz")
        latents = d["latents"]
        subset_data[subset] = {
            "latents": latents,
            "raw_min": latents.min(axis=0),
            "raw_max": latents.max(axis=0),
            "n_train": int(latents.shape[0]),
        }
        print(f"  {subset}: n={latents.shape[0]}, latent_dim={latents.shape[1]}",
              flush=True)

    results = {
        "provenance": {
            "script": "verify_ae_latent_quantile_sweep.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit_short(),
            "timeout_seconds": TIMEOUT_S,
            "edge_proximity_fraction": EDGE_FRACTION,
            "mean_constraint_enforced": True,
            "quantile_settings": [
                {"label": lbl, "low_q": lo, "high_q": hi}
                for (lbl, lo, hi) in QUANTILE_SETTINGS
            ],
            "ae_mapping": AE_MAPPING,
        },
        "subset_info": {
            s: {
                "n_train": subset_data[s]["n_train"],
                "raw_latent_min": [round(float(x), 6) for x in subset_data[s]["raw_min"]],
                "raw_latent_max": [round(float(x), 6) for x in subset_data[s]["raw_max"]],
            }
            for s in subset_data
        },
        "quantile_sweep": {},
    }

    # ---- Step 1: run min/max baseline first and validate ------------------
    print("\n" + "=" * 80, flush=True)
    print("STEP 1: Reproducing min/max baseline (paper's current AE+mean verdicts)",
          flush=True)
    print("=" * 80, flush=True)

    baseline_label = QUANTILE_SETTINGS[0][0]
    baseline_results = {}

    for name, kind in QUERIES:
        subset = AE_MAPPING[kind]
        latents = subset_data[subset]["latents"]
        z_min, z_max = get_latent_bounds(latents, None, None)
        onnx_path = f"composed_{subset}_with_sum.onnx"
        print(f"  {name} (AE: {subset})  ", end="", flush=True)
        r = solve_one(onnx_path, kind, z_min, z_max)
        r["latent_bounds"] = {
            "z_min": [round(float(x), 6) for x in z_min],
            "z_max": [round(float(x), 6) for x in z_max],
        }
        r["ae_subset"] = subset
        msg = r["status"]
        if r["status"] == "sat":
            msg += f"  CTE={r['counterexample']['CTE']:+.3f}"
        print(f"-> {msg} ({r['time_seconds']:.2f}s)", flush=True)
        baseline_results[name] = r

    results["quantile_sweep"][baseline_label] = baseline_results

    # Compare against expected baseline verdicts
    mismatches = []
    for name, expected in BASELINE_AE_WITH_MEAN.items():
        got = baseline_results[name]["status"]
        if got != expected:
            mismatches.append((name, expected, got))

    if mismatches:
        print("\n" + "!" * 80, flush=True)
        print("BASELINE REPRODUCTION FAILED — stopping before remaining quantiles.",
              flush=True)
        for name, exp, got in mismatches:
            print(f"  {name}: expected {exp}, got {got}", flush=True)
        print("!" * 80, flush=True)
        results["baseline_reproduction"] = {
            "ok": False,
            "mismatches": [{"name": n, "expected": e, "got": g}
                           for (n, e, g) in mismatches],
        }
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Partial results saved to {OUTPUT_JSON}", flush=True)
        return

    results["baseline_reproduction"] = {"ok": True}
    print("\nBaseline reproduction OK — proceeding with tighter quantiles.\n",
          flush=True)

    # ---- Step 2: run the three tighter quantile settings ------------------
    for q_label, low_q, high_q in QUANTILE_SETTINGS[1:]:
        print("=" * 80, flush=True)
        print(f"Quantile setting: {q_label}  (low={low_q}, high={high_q})",
              flush=True)
        print("=" * 80, flush=True)
        per_query = {}
        for name, kind in QUERIES:
            subset = AE_MAPPING[kind]
            latents = subset_data[subset]["latents"]
            z_min, z_max = get_latent_bounds(latents, low_q, high_q)
            onnx_path = f"composed_{subset}_with_sum.onnx"
            print(f"  {name} (AE: {subset})  ", end="", flush=True)
            r = solve_one(onnx_path, kind, z_min, z_max)
            r["latent_bounds"] = {
                "z_min": [round(float(x), 6) for x in z_min],
                "z_max": [round(float(x), 6) for x in z_max],
            }
            r["ae_subset"] = subset
            msg = r["status"]
            if r["status"] == "sat":
                msg += f"  CTE={r['counterexample']['CTE']:+.3f}"
            print(f"-> {msg} ({r['time_seconds']:.2f}s)", flush=True)
            per_query[name] = r
        results["quantile_sweep"][q_label] = per_query

        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=2)

    # ---- Step 3: summary --------------------------------------------------
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUTPUT_JSON}", flush=True)

    print("\n" + "=" * 96, flush=True)
    print("SUMMARY: Verdict (and CTE) per quantile setting", flush=True)
    print("=" * 96, flush=True)
    header = f"{'Query':<18}"
    for lbl, _, _ in QUANTILE_SETTINGS:
        header += f" | {lbl:<16}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    def cell(r):
        st = r["status"]
        if st == "sat":
            return f"sat CTE={r['counterexample']['CTE']:+.2f}"
        if st == "timeout":
            return f"timeout {r['time_seconds']:.0f}s"
        return st

    for name, _ in QUERIES:
        row = f"{name:<18}"
        for lbl, _, _ in QUANTILE_SETTINGS:
            r = results["quantile_sweep"][lbl][name]
            row += f" | {cell(r):<16}"
        print(row, flush=True)

    print("\nHeadline (verdict changes across quantile width):", flush=True)
    for name, _ in QUERIES:
        verdicts = [results["quantile_sweep"][lbl][name]["status"]
                    for lbl, _, _ in QUANTILE_SETTINGS]
        unique = set(verdicts)
        if len(unique) == 1:
            print(f"  {name}: stable ({verdicts[0]}) across all four widths",
                  flush=True)
        else:
            transitions = " -> ".join(verdicts)
            print(f"  {name}: VERDICT CHANGED  {transitions}", flush=True)


if __name__ == "__main__":
    main()
