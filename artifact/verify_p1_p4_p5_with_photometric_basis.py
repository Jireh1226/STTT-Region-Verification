"""
Photometric-basis-bounded verification for P1/P4/P5 on TinyTaxiNet.

Hypothesis being tested
-----------------------
Real runway images approximately decompose along two physically motivated
lighting axes plus a small residual:

    x ~ alpha * b1 + beta * b2 + r,        ||r||_inf small,

where
    b1[i,j] = 1 / sqrt(128)                                  (uniform brightness)
    b2[i,j] = (j - 7.5) / sqrt(sum_{i,j} (j - 7.5)**2)       (horizontal gradient,
                                                              mean-subtracted)

b1 and b2 are orthonormal by construction (b2 has zero column-mean, hence is
orthogonal to b1, and both are unit-norm). For an input x, the (oblique-but-
here-orthogonal) projection scalars are
    alpha = <x, b1>,  beta = <x, b2>,
and the residual is
    r = x - alpha * b1 - beta * b2.

We add the photometric region constraint ||r||_inf <= epsilon to the standard
per-pixel-box + mean(x)=0.5 region. Compared to TV (local L1 gradient) and
Fourier (high-frequency mass), this constraint forces the witness to live on
a 2-D affine subspace (any brightness, any horizontal gradient) within an
L_inf ball of radius epsilon. It is the most aggressive of the three
"physical" regularisers.

Marabou encoding (per query):
  - 2 new variables: alpha, beta (free in sign, no bounds).
  - 128 equalities of the form r_i = x_i - alpha * b1[i] - beta * b2[i]
    i.e.  x_i - alpha*b1[i] - beta*b2[i] - r_i = 0,
  - 256 box bounds: -epsilon <= r_i <= +epsilon.

At epsilon = inf no constraint is added (baseline reproduction).

Per-query timeout: 300 s. Writes JSON to verification_photometric_bound.json.

Usage:
  ./prophecy_env/bin/python verify_p1_p4_p5_with_photometric_basis.py
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import subprocess
import time

import h5py
import numpy as np
from maraboupy import Marabou


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "verification_photometric_bound.json"

NUM_PIXELS = 128
ROWS, COLS = 8, 16
MEAN_TARGET = 0.5
SUM_TARGET = MEAN_TARGET * NUM_PIXELS  # 64.0

CTE_OUT_VAR = 192
HE_OUT_VAR = 193

P1_CTE_BOUND = 10.0
P5_DEADZONE = 0.01

TIMEOUT_S = 300


# -----------------------------------------------------------------------------
# Photometric basis (two orthonormal vectors in 128-D pixel space)
# -----------------------------------------------------------------------------
def _build_basis() -> tuple[np.ndarray, np.ndarray]:
    """Build (b1, b2): brightness and horizontal-gradient orthonormal vectors."""
    b1 = np.ones(NUM_PIXELS, dtype=np.float64) / math.sqrt(NUM_PIXELS)

    grad = np.zeros((ROWS, COLS), dtype=np.float64)
    for i in range(ROWS):
        for j in range(COLS):
            grad[i, j] = (j - 7.5)
    g = grad.flatten()
    g_norm = float(np.linalg.norm(g))
    b2 = g / g_norm
    return b1, b2


B1, B2 = _build_basis()


def _basis_info() -> dict:
    dot = float(np.dot(B1, B2))
    return {
        "b1_description": "uniform brightness basis: b1[k] = 1 / sqrt(128)",
        "b2_description": (
            "horizontal-position gradient basis: b2[i,j] proportional to "
            "(j - 7.5), normalised to unit L2 norm; mean-subtracted by "
            "construction so orthogonal to b1"
        ),
        "b1_norm": round(float(np.linalg.norm(B1)), 12),
        "b2_norm": round(float(np.linalg.norm(B2)), 12),
        "b1_dot_b2": dot,
        "b1_dot_b2_abs": abs(dot),
        "orthonormal_check": (
            "abs(b1.b2) < 1e-12 and ||b1||=||b2||=1 (to floating-point)"
        ),
        "rows": ROWS,
        "cols": COLS,
    }


# -----------------------------------------------------------------------------
# Data + photometric calibration
# -----------------------------------------------------------------------------
def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].astype(np.float64)
        labels = f["X_train"][:].astype(np.float64)
    return images, labels


def compute_box(images: np.ndarray):
    flat = images.reshape(images.shape[0], -1)
    return flat.min(axis=0), flat.max(axis=0)


def photometric_residual(flat128: np.ndarray) -> np.ndarray:
    x = np.asarray(flat128, dtype=np.float64).reshape(-1)
    alpha = float(np.dot(x, B1))
    beta = float(np.dot(x, B2))
    return x - alpha * B1 - beta * B2


def residual_linf(flat128: np.ndarray) -> float:
    return float(np.abs(photometric_residual(flat128)).max())


def alpha_beta(flat128: np.ndarray) -> tuple[float, float]:
    x = np.asarray(flat128, dtype=np.float64).reshape(-1)
    return float(np.dot(x, B1)), float(np.dot(x, B2))


def residual_quantiles(images_flat: np.ndarray) -> dict:
    rs = np.array([residual_linf(x) for x in images_flat], dtype=np.float64)
    return {
        "n": int(len(rs)),
        "min": round(float(rs.min()), 6),
        "p01": round(float(np.percentile(rs, 1)), 6),
        "p25": round(float(np.percentile(rs, 25)), 6),
        "p50": round(float(np.percentile(rs, 50)), 6),
        "p75": round(float(np.percentile(rs, 75)), 6),
        "p90": round(float(np.percentile(rs, 90)), 6),
        "p99": round(float(np.percentile(rs, 99)), 6),
        "max": round(float(rs.max()), 6),
        "mean": round(float(rs.mean()), 6),
        "std": round(float(rs.std()), 6),
    }


# -----------------------------------------------------------------------------
# Photometric encoding into a Marabou query
# -----------------------------------------------------------------------------
def add_photometric_bound(net, input_vars: list[int], epsilon: float) -> None:
    """For each pixel i introduce r_i = x_i - alpha*b1[i] - beta*b2[i] with
    -epsilon <= r_i <= +epsilon. alpha and beta are free Marabou variables."""
    if not math.isfinite(epsilon):
        return  # epsilon = inf -> no constraint added

    alpha = net.getNewVariable()
    beta = net.getNewVariable()
    # alpha and beta are unbounded in sign; give them very wide box bounds so
    # Marabou treats them as ordinary continuous variables rather than as
    # uninitialised.
    BIG = 1e6
    net.setLowerBound(alpha, -BIG)
    net.setUpperBound(alpha, BIG)
    net.setLowerBound(beta, -BIG)
    net.setUpperBound(beta, BIG)

    for i in range(NUM_PIXELS):
        x_i = int(input_vars[i])
        r_i = net.getNewVariable()
        net.setLowerBound(r_i, -float(epsilon))
        net.setUpperBound(r_i, float(epsilon))
        # x_i - alpha*b1[i] - beta*b2[i] - r_i = 0
        net.addEquality(
            [x_i, int(alpha), int(beta), r_i],
            [1.0, -float(B1[i]), -float(B2[i]), -1.0],
            0.0,
        )


# -----------------------------------------------------------------------------
# Query builder + runner
# -----------------------------------------------------------------------------
def build_query(pixel_mins: np.ndarray, pixel_maxs: np.ndarray, epsilon: float):
    """Build a fresh Marabou network with per-pixel box + mean=0.5 +
    photometric residual L_inf bound. Returns (net, input_vars)."""
    net = Marabou.read_nnet(NNET_PATH)
    input_vars = [int(v) for v in net.inputVars[0].flatten()]
    assert len(input_vars) == NUM_PIXELS
    for i in range(NUM_PIXELS):
        net.setLowerBound(input_vars[i], float(pixel_mins[i]))
        net.setUpperBound(input_vars[i], float(pixel_maxs[i]))
    net.addEquality(input_vars, [1.0] * NUM_PIXELS, SUM_TARGET)
    add_photometric_bound(net, input_vars, epsilon)
    return net, input_vars


def _record_sat_witness(vals, input_vars: list[int]) -> dict:
    cx = np.array([vals[v] for v in input_vars], dtype=np.float64)
    a, b = alpha_beta(cx)
    return {
        "CTE": round(float(vals[CTE_OUT_VAR]), 6),
        "HE": round(float(vals[HE_OUT_VAR]), 6),
        "witness_mean": round(float(cx.mean()), 6),
        "witness_residual_linf": round(residual_linf(cx), 6),
        "witness_alpha": round(a, 6),
        "witness_beta": round(b, 6),
        "witness": [round(float(v), 6) for v in cx.tolist()],
    }


def run_single_output_bound(name: str,
                            pixel_mins: np.ndarray,
                            pixel_maxs: np.ndarray,
                            epsilon: float,
                            out_var: int,
                            direction: str,
                            bound: float) -> dict:
    """For P1a/P1b/P4_left/P4_right: a single output bound constraint."""
    net, input_vars = build_query(pixel_mins, pixel_maxs, epsilon)
    if direction == ">=":
        net.setLowerBound(out_var, float(bound))
    else:
        net.setUpperBound(out_var, float(bound))
    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    try:
        status, vals, _ = net.solve(verbose=False, options=opts)
    except Exception as e:  # pragma: no cover -- defensive
        return {
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "time_seconds": round(time.time() - t0, 4),
        }
    elapsed = time.time() - t0
    status = (status or "").strip().lower()
    if status == "":
        status = "timeout"
    out: dict = {
        "status": status,
        "time_seconds": round(elapsed, 4),
    }
    if status == "sat":
        out.update(_record_sat_witness(vals, input_vars))
    return out


def run_p5(pixel_mins: np.ndarray,
           pixel_maxs: np.ndarray,
           epsilon: float,
           deadzone: float = P5_DEADZONE) -> dict:
    """|CTE| <= deadzone AND |HE| <= deadzone on off-center box."""
    net, input_vars = build_query(pixel_mins, pixel_maxs, epsilon)
    net.setLowerBound(CTE_OUT_VAR, -deadzone)
    net.setUpperBound(CTE_OUT_VAR, deadzone)
    net.setLowerBound(HE_OUT_VAR, -deadzone)
    net.setUpperBound(HE_OUT_VAR, deadzone)
    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    try:
        status, vals, _ = net.solve(verbose=False, options=opts)
    except Exception as e:  # pragma: no cover
        return {
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "time_seconds": round(time.time() - t0, 4),
        }
    elapsed = time.time() - t0
    status = (status or "").strip().lower()
    if status == "":
        status = "timeout"
    out: dict = {
        "status": status,
        "time_seconds": round(elapsed, 4),
        "deadzone": deadzone,
    }
    if status == "sat":
        out.update(_record_sat_witness(vals, input_vars))
    return out


# -----------------------------------------------------------------------------
# Main sweep
# -----------------------------------------------------------------------------
def eps_key(label: str) -> str:
    return f"eps_{label}"


def run_eps(pixel_boxes: dict, epsilon: float) -> dict:
    results: dict = {}
    pmin, pmax = pixel_boxes["P1"]
    results["P1a"] = run_single_output_bound(
        "P1a", pmin, pmax, epsilon, CTE_OUT_VAR, ">=", P1_CTE_BOUND
    )
    results["P1b"] = run_single_output_bound(
        "P1b", pmin, pmax, epsilon, CTE_OUT_VAR, "<=", -P1_CTE_BOUND
    )
    pmin, pmax = pixel_boxes["P4_left"]
    results["P4_left"] = run_single_output_bound(
        "P4_left", pmin, pmax, epsilon, CTE_OUT_VAR, "<=", 0.0
    )
    pmin, pmax = pixel_boxes["P4_right"]
    results["P4_right"] = run_single_output_bound(
        "P4_right", pmin, pmax, epsilon, CTE_OUT_VAR, ">=", 0.0
    )
    pmin, pmax = pixel_boxes["P5"]
    results["P5"] = run_p5(pmin, pmax, epsilon, deadzone=P5_DEADZONE)
    return results


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _pick_epsilons(centered_q: dict) -> list[tuple[str, float]]:
    """Choose the epsilon sweep:
      eps_inf    = math.inf      (baseline reproduction)
      eps_loose  = 0.5           (very loose: any pixel can drift up to 0.5)
      eps_99p    = p99 of centered residual L_inf
      eps_max    = max of centered residual L_inf
      eps_tight  = halfway between p25 (a value < p50) and p01 of centered
                   (i.e. below median, to push solver into the photometric ball).
    """
    p99 = float(centered_q["p99"])
    pmax = float(centered_q["max"])
    p25 = float(centered_q["p25"])
    p01 = float(centered_q["p01"])
    tight = round(0.5 * (p01 + p25), 6)
    return [
        ("inf", math.inf),
        ("loose", 0.5),
        ("99p", p99),
        ("max", pmax),
        ("tight", tight),
    ]


def main() -> None:
    print("Loading data...", flush=True)
    images, labels = load_data()
    cte = labels[:, 0]
    flat_images = images.reshape(images.shape[0], -1)

    centered = flat_images[np.abs(cte) < 1.0]
    left = flat_images[cte > 2.0]
    right = flat_images[cte < -2.0]
    off_center = flat_images[np.abs(cte) > 1.0]
    print(
        f"  N(total)={len(flat_images)}  centered={len(centered)}  "
        f"left={len(left)}  right={len(right)}  off_center={len(off_center)}",
        flush=True,
    )

    pixel_boxes = {
        "P1": compute_box(centered),
        "P4_left": compute_box(left),
        "P4_right": compute_box(right),
        "P5": compute_box(off_center),
    }

    print("Computing photometric residual L_inf quantiles...", flush=True)
    centered_q = residual_quantiles(centered)
    full_q = residual_quantiles(flat_images)
    photometric_calibration = {
        "centered_residual_linf_quantiles": centered_q,
        "full_dataset_residual_linf_quantiles": full_q,
    }
    print(
        f"  centered residual L_inf [min={centered_q['min']}, "
        f"p50={centered_q['p50']}, p99={centered_q['p99']}, "
        f"max={centered_q['max']}]",
        flush=True,
    )
    print(
        f"  full     residual L_inf [min={full_q['min']}, "
        f"p50={full_q['p50']}, p99={full_q['p99']}, "
        f"max={full_q['max']}]",
        flush=True,
    )

    eps_schedule = _pick_epsilons(centered_q)
    print("Epsilon sweep:", flush=True)
    for label, eps in eps_schedule:
        eps_str = "inf" if not math.isfinite(eps) else f"{eps:.6f}"
        print(f"  eps_{label} = {eps_str}", flush=True)

    results_by_eps: dict = {}
    eps_values_log: dict = {}
    for label, epsilon in eps_schedule:
        key = eps_key(label)
        eps_values_log[key] = (
            "inf" if not math.isfinite(epsilon) else float(epsilon)
        )
        print(
            f"\n=== Running sweep at {key} (epsilon="
            f"{'inf' if not math.isfinite(epsilon) else epsilon}) ===",
            flush=True,
        )
        t0 = time.time()
        results_by_eps[key] = run_eps(pixel_boxes, epsilon)
        for prop_name, r in results_by_eps[key].items():
            extra = ""
            if r["status"] == "sat":
                extra = (
                    f"  CTE={r['CTE']:+.4f}  HE={r['HE']:+.4f}  "
                    f"mean={r['witness_mean']:.4f}  "
                    f"resL_inf={r['witness_residual_linf']:.4f}  "
                    f"alpha={r['witness_alpha']:+.4f}  "
                    f"beta={r['witness_beta']:+.4f}"
                )
            print(
                f"  {prop_name:<8} {r['status'].upper():<7} "
                f"t={r['time_seconds']:>7.2f}s{extra}",
                flush=True,
            )
        print(f"  epsilon wall: {time.time()-t0:.2f}s", flush=True)

    output = {
        "provenance": {
            "script": "verify_p1_p4_p5_with_photometric_basis.py",
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "git_commit": _git_commit(),
            "nnet": NNET_PATH,
            "data": DATA_PATH,
            "timeout_seconds_per_query": TIMEOUT_S,
            "mean_target": MEAN_TARGET,
            "sum_target": SUM_TARGET,
            "p1_cte_bound": P1_CTE_BOUND,
            "p5_deadzone": P5_DEADZONE,
            "epsilon_sweep_labels": [lab for lab, _ in eps_schedule],
            "epsilon_sweep_values": eps_values_log,
            "photometric_encoding_aux_vars_per_query": 2 + NUM_PIXELS,
            "photometric_encoding_equalities_per_query": NUM_PIXELS,
            "rows": ROWS,
            "cols": COLS,
        },
        "basis_info": _basis_info(),
        "photometric_calibration": photometric_calibration,
        "results": results_by_eps,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
