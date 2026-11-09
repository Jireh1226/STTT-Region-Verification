"""
AE-composed verification with additional math (TV / photometric) constraints
on the DECODED pixels.

Tests the methodology question: do AE-region SAT witnesses also satisfy math
constraints, or are they non-physical along at least one math axis?

The composed ONNX networks (composed_<subset>_with_sum.onnx) produce
[CTE, HE, pixel_sum] as their declared outputs. Marabou's ONNX reader supports
an `outputNames` parameter that exposes additional intermediate tensors as
outputs. We use this to expose the decoded-pixel tensor

    StatefulPartitionedCall/composed_with_sum/pixels/BiasAdd:0   (shape [1,128])

as a second output, which gives us direct Marabou variable handles for each of
the 128 decoded pixels. We can then add TV / photometric constraints on those
variables exactly like the pixel-space scripts do
(verify_p1_p4_p5_with_tv_bound.py, verify_p1_p4_p5_with_photometric_basis.py).

For each (property, setting) we record status, time, and (if SAT) CTE, HE, the
128 witness pixels, and post-hoc TV / photometric residual on the witness.

Usage:
    ./prophecy_env/bin/python verify_ae_with_math.py
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import subprocess
import time

import numpy as np
from maraboupy import Marabou


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
OUTPUT_JSON = "verification_ae_plus_math.json"

NUM_PIXELS = 128
ROWS, COLS = 8, 16
MEAN_TARGET = 0.5
SUM_TARGET = MEAN_TARGET * NUM_PIXELS  # 64.0

TIMEOUT_S = 300
P1_CTE_BOUND = 10.0
P5_DEADZONE = 0.01

# Tensor name (in every composed_<subset>_with_sum.onnx) that produces the 128
# decoded pixels right after the autoencoder.
DECODED_PIXELS_TENSOR = (
    "StatefulPartitionedCall/composed_with_sum/pixels/BiasAdd:0"
)
COMBINED_OUT_TENSOR = "combined_out"

AE_MAPPING = {
    "P1a":      "centered",
    "P1b":      "centered",
    "P4_left":  "left",
    "P4_right": "right",
    "P5":       "off_center",
}

# Math settings: (label, kind, value)
#   tv_12  -> TV(x) <= 12.0   (centered p99 is 12.18, so 12.0 is just below)
#   tv_10  -> TV(x) <= 10.0   (below the centered median 9.37 is too tight;
#                              10.0 is between p50 and p90)
#   photo_0p20 -> ||r||_inf <= 0.20  (just below centered residual median 0.208)
MATH_SETTINGS = [
    ("tv_12",      "tv",    12.0),
    ("tv_10",      "tv",    10.0),
    ("photo_0p20", "photo", 0.20),
]


# -----------------------------------------------------------------------------
# Math helpers (pixel space)
# -----------------------------------------------------------------------------
def neighbor_pairs() -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(ROWS):
        for j in range(COLS - 1):
            pairs.append((i * COLS + j, i * COLS + j + 1))
    for i in range(ROWS - 1):
        for j in range(COLS):
            pairs.append((i * COLS + j, (i + 1) * COLS + j))
    return pairs


NEIGHBOR_PAIRS = neighbor_pairs()
assert len(NEIGHBOR_PAIRS) == ROWS * (COLS - 1) + (ROWS - 1) * COLS == 232


def build_photometric_basis() -> tuple[np.ndarray, np.ndarray]:
    b1 = np.ones(NUM_PIXELS, dtype=np.float64) / math.sqrt(NUM_PIXELS)
    grad = np.zeros((ROWS, COLS), dtype=np.float64)
    for i in range(ROWS):
        for j in range(COLS):
            grad[i, j] = j - 7.5
    g = grad.flatten()
    b2 = g / float(np.linalg.norm(g))
    return b1, b2


B1, B2 = build_photometric_basis()


def tv_of_image(flat128: np.ndarray) -> float:
    img = np.asarray(flat128, dtype=np.float64).reshape(ROWS, COLS)
    return float(np.abs(img[:, :-1] - img[:, 1:]).sum()
                 + np.abs(img[:-1, :] - img[1:, :]).sum())


def photometric_residual_linf(flat128: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(flat128, dtype=np.float64).reshape(-1)
    alpha = float(np.dot(x, B1))
    beta = float(np.dot(x, B2))
    r = x - alpha * B1 - beta * B2
    return float(np.abs(r).max()), alpha, beta


# -----------------------------------------------------------------------------
# Marabou query construction
# -----------------------------------------------------------------------------
def load_subset_latent_box(subset: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(f"autoencoder_{subset}_weights.npz")
    return data["latent_min"], data["latent_max"]


def add_tv_bound(net, pixel_vars: list[int], tau: float) -> dict:
    """For each neighbor pair (a, b) introduce d >= 0 with d >= |a - b|, then
    sum(d) <= tau. Returns book-keeping counts."""
    d_vars = []
    for (a_idx, b_idx) in NEIGHBOR_PAIRS:
        a = int(pixel_vars[a_idx])
        b = int(pixel_vars[b_idx])
        d = net.getNewVariable()
        net.setLowerBound(d, 0.0)
        # d >= a - b  <=>  a - b - d <= 0
        net.addInequality([a, b, d], [1.0, -1.0, -1.0], 0.0)
        # d >= b - a  <=>  b - a - d <= 0
        net.addInequality([b, a, d], [1.0, -1.0, -1.0], 0.0)
        d_vars.append(d)
    # sum(d) <= tau
    net.addInequality(d_vars, [1.0] * len(d_vars), float(tau))
    return {
        "tv_aux_vars": len(d_vars),
        "tv_inequalities": 2 * len(d_vars) + 1,
        "tau": tau,
    }


def add_photometric_bound(net, pixel_vars: list[int], epsilon: float) -> dict:
    """For each pixel i introduce r_i = x_i - alpha*b1[i] - beta*b2[i] with
    -eps <= r_i <= +eps. alpha, beta are free."""
    BIG = 1e6
    alpha = net.getNewVariable()
    beta = net.getNewVariable()
    net.setLowerBound(alpha, -BIG); net.setUpperBound(alpha, BIG)
    net.setLowerBound(beta, -BIG);  net.setUpperBound(beta, BIG)
    for i in range(NUM_PIXELS):
        x_i = int(pixel_vars[i])
        r_i = net.getNewVariable()
        net.setLowerBound(r_i, -float(epsilon))
        net.setUpperBound(r_i,  float(epsilon))
        # x_i - alpha*b1[i] - beta*b2[i] - r_i = 0
        net.addEquality(
            [x_i, int(alpha), int(beta), r_i],
            [1.0, -float(B1[i]), -float(B2[i]), -1.0],
            0.0,
        )
    return {
        "photo_aux_vars": 2 + NUM_PIXELS,
        "photo_equalities": NUM_PIXELS,
        "epsilon": epsilon,
    }


def build_query(subset: str, math_setting: tuple | None):
    """Construct a fresh Marabou query on the composed ONNX network with:
      - latent box from the subset AE
      - mean(pixels) = 0.5 (via the auxiliary pixel_sum output)
      - optional TV or photometric constraint on the decoded pixels
    Returns (net, input_vars, ttn_out_vars [CTE,HE,pixel_sum], pixel_vars,
             encoding_info)."""
    onnx_path = f"composed_{subset}_with_sum.onnx"
    net = Marabou.read_onnx(
        onnx_path,
        outputNames=[COMBINED_OUT_TENSOR, DECODED_PIXELS_TENSOR],
    )
    input_vars = [int(v) for v in np.array(net.inputVars[0]).flatten()]
    ttn_out = np.array(net.outputVars[0]).flatten()
    cte_var, he_var, sum_var = (int(ttn_out[0]), int(ttn_out[1]),
                                int(ttn_out[2]))
    pixel_vars = [int(v) for v in np.array(net.outputVars[1]).flatten()]
    assert len(pixel_vars) == NUM_PIXELS

    # Latent box
    z_min, z_max = load_subset_latent_box(subset)
    for i, var in enumerate(input_vars):
        net.setLowerBound(var, float(z_min[i]))
        net.setUpperBound(var, float(z_max[i]))

    # Mean-of-pixels constraint via the existing pixel_sum auxiliary output
    net.setLowerBound(sum_var, SUM_TARGET)
    net.setUpperBound(sum_var, SUM_TARGET)

    encoding_info = {}
    if math_setting is not None:
        kind, value = math_setting
        if kind == "tv":
            encoding_info = add_tv_bound(net, pixel_vars, value)
        elif kind == "photo":
            encoding_info = add_photometric_bound(net, pixel_vars, value)
        else:
            raise ValueError(f"Unknown math kind: {kind}")
    encoding_info["kind"] = math_setting[0] if math_setting else "none"

    return net, input_vars, (cte_var, he_var, sum_var), pixel_vars, encoding_info


def apply_postcondition(net, ttn_outs: tuple[int, int, int], kind: str) -> None:
    cte_var, he_var, _ = ttn_outs
    if kind == "P1a":
        net.setLowerBound(cte_var, P1_CTE_BOUND)
    elif kind == "P1b":
        net.setUpperBound(cte_var, -P1_CTE_BOUND)
    elif kind == "P4_left":
        net.setUpperBound(cte_var, 0.0)
    elif kind == "P4_right":
        net.setLowerBound(cte_var, 0.0)
    elif kind == "P5":
        net.setLowerBound(cte_var, -P5_DEADZONE)
        net.setUpperBound(cte_var,  P5_DEADZONE)
        net.setLowerBound(he_var,  -P5_DEADZONE)
        net.setUpperBound(he_var,   P5_DEADZONE)
    else:
        raise ValueError(f"Unknown property kind: {kind}")


# -----------------------------------------------------------------------------
# Solve and record
# -----------------------------------------------------------------------------
def record_witness(vals, input_vars, ttn_outs, pixel_vars) -> dict:
    cte_var, he_var, sum_var = ttn_outs
    pixels = np.array([float(vals[v]) for v in pixel_vars], dtype=np.float64)
    tv = tv_of_image(pixels)
    res_inf, alpha, beta = photometric_residual_linf(pixels)
    return {
        "CTE": round(float(vals[cte_var]), 6),
        "HE":  round(float(vals[he_var]),  6),
        "decoded_pixel_sum_var": round(float(vals[sum_var]), 6),
        "decoded_pixel_sum_check": round(float(pixels.sum()), 6),
        "decoded_pixel_mean": round(float(pixels.mean()), 6),
        "decoded_pixel_min": round(float(pixels.min()), 6),
        "decoded_pixel_max": round(float(pixels.max()), 6),
        "decoded_TV": round(tv, 6),
        "decoded_photo_residual_linf": round(res_inf, 6),
        "decoded_photo_alpha": round(alpha, 6),
        "decoded_photo_beta": round(beta, 6),
        "latent": [round(float(vals[v]), 6) for v in input_vars],
        "decoded_pixels": [round(float(p), 6) for p in pixels.tolist()],
    }


def solve_one(prop: str, math_setting: tuple | None) -> dict:
    subset = AE_MAPPING[prop]
    net, input_vars, ttn_outs, pixel_vars, encoding_info = build_query(
        subset, math_setting
    )
    apply_postcondition(net, ttn_outs, prop)

    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    try:
        status_raw, vals, _ = net.solve(verbose=False, options=opts)
    except Exception as e:
        return {
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "time_seconds": round(time.time() - t0, 4),
            "ae_subset": subset,
            "encoding": encoding_info,
        }
    elapsed = time.time() - t0
    status = (status_raw or "").strip().lower()
    if status == "":
        status = "timeout"

    out = {
        "status": status,
        "time_seconds": round(elapsed, 4),
        "ae_subset": subset,
        "encoding": encoding_info,
    }
    if status == "sat":
        out.update(record_witness(vals, input_vars, ttn_outs, pixel_vars))
    return out


# -----------------------------------------------------------------------------
# Provenance
# -----------------------------------------------------------------------------
def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# -----------------------------------------------------------------------------
# Main sweep
# -----------------------------------------------------------------------------
PROPERTIES = ["P1a", "P1b", "P4_left", "P4_right", "P5"]


def run_block(label: str, math_setting: tuple | None) -> dict:
    print(f"\n=== block: {label} ===", flush=True)
    block_results: dict = {}
    for prop in PROPERTIES:
        print(f"  {prop:<10} ", end="", flush=True)
        r = solve_one(prop, math_setting)
        msg = r["status"].upper()
        if r["status"] == "sat":
            msg += (
                f"  CTE={r['CTE']:+.3f}  HE={r['HE']:+.3f}  "
                f"mean={r['decoded_pixel_mean']:.3f}  "
                f"TV={r['decoded_TV']:.3f}  "
                f"photoRes={r['decoded_photo_residual_linf']:.3f}"
            )
        print(f"{msg:<70}  t={r['time_seconds']:.2f}s", flush=True)
        block_results[prop] = r
    return block_results


def main() -> None:
    print("AE + math composed verification", flush=True)
    print(f"  timeout per query: {TIMEOUT_S}s", flush=True)
    print(f"  properties: {PROPERTIES}", flush=True)
    print(f"  AE mapping: {AE_MAPPING}", flush=True)

    results: dict = {}
    # Baseline: AE region + mean=0.5 only (reproduces ae_with_mean from
    # verification_ae_with_mean_constraint.json under the modified ONNX read).
    results["ae_baseline"] = run_block("ae_baseline (no math)", math_setting=None)

    for label, kind, value in MATH_SETTINGS:
        block_label = f"ae_plus_{label}"
        results[block_label] = run_block(block_label, math_setting=(kind, value))

    output = {
        "provenance": {
            "script": "verify_ae_with_math.py",
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "git_commit": git_commit(),
            "timeout_seconds_per_query": TIMEOUT_S,
            "rows": ROWS,
            "cols": COLS,
            "num_pixels": NUM_PIXELS,
            "mean_target": MEAN_TARGET,
            "sum_target": SUM_TARGET,
            "p1_cte_bound": P1_CTE_BOUND,
            "p5_deadzone": P5_DEADZONE,
            "ae_mapping": AE_MAPPING,
            "ae_subsets_used": sorted(set(AE_MAPPING.values())),
            "onnx_decoded_pixels_tensor": DECODED_PIXELS_TENSOR,
            "onnx_combined_out_tensor": COMBINED_OUT_TENSOR,
            "math_settings": [
                {"label": lab, "kind": k, "value": v}
                for (lab, k, v) in MATH_SETTINGS
            ],
            "tv_pairs": len(NEIGHBOR_PAIRS),
        },
        "implementation_note": (
            "We expose the autoencoder's decoded-pixel tensor "
            f"'{DECODED_PIXELS_TENSOR}' as an extra Marabou output by passing "
            f"outputNames=[{COMBINED_OUT_TENSOR!r}, {DECODED_PIXELS_TENSOR!r}] "
            "to Marabou.read_onnx. This gives 128 Marabou variables (one per "
            "decoded pixel) on which TV and photometric constraints are "
            "imposed exactly as in the pixel-space scripts "
            "(verify_p1_p4_p5_with_tv_bound.py and "
            "verify_p1_p4_p5_with_photometric_basis.py). The TV encoding adds "
            "one slack variable per neighbour pair (232 aux vars + 465 linear "
            "inequalities) and the photometric encoding adds alpha, beta and "
            "128 residual variables with 128 linear equalities and 256 box "
            "bounds. mean(x)=0.5 is enforced via the existing pixel_sum "
            "auxiliary output (sum = 64). No script regeneration of the ONNX "
            "files was required."
        ),
        "results": results,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}", flush=True)

    # Headline summary
    print("\n" + "=" * 88, flush=True)
    print("SUMMARY: AE + math constraints (rows = properties, cols = settings)",
          flush=True)
    print("=" * 88, flush=True)
    headers = ["property", "ae_baseline"] + [f"ae+{lab}"
                                              for (lab, _, _) in MATH_SETTINGS]
    print("  " + "  ".join(f"{h:<16}" for h in headers), flush=True)
    print("  " + "-" * (18 * len(headers)), flush=True)
    for prop in PROPERTIES:
        row = [prop]
        for block in ["ae_baseline"] + [f"ae_plus_{lab}"
                                         for (lab, _, _) in MATH_SETTINGS]:
            r = results[block][prop]
            if r["status"] == "sat":
                cell = f"sat({r['CTE']:+.2f})"
            else:
                cell = r["status"]
            row.append(cell)
        print("  " + "  ".join(f"{c:<16}" for c in row), flush=True)


if __name__ == "__main__":
    main()
