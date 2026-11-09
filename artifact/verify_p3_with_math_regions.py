"""
P3 robustness verification with math-region constraints (TV / Fourier-HF / Photometric)
intersected with the per-image L_inf perturbation ball.

Regime
------
Unlike the P1/P4/P5 math-region scripts that use a per-subset min/max pixel box,
this script tests robustness around 20 anchor (centered) images at
epsilon = 0.05 (per-pixel L_inf), intersected with mean(x) = 0.5 (sum = 64).
For each anchor and each math-region setting we run all four P3 sub-queries
(CTE >= 10, CTE <= -10, HE >= 90, HE <= -90); an image is verified iff
all four are UNSAT.

Math regions (one at a time, in addition to the L_inf box + mean constraint):
  1. TV bound:                 TV(x) <= tau_TV,  tau_TV in {12, 13, 15}
  2. Fourier HF L1 bound:      HF-L1(x) <= tau_F, tau_F in {7.0, 7.92, 8.16}
  3. Photometric residual:     ||x - alpha*b1 - beta*b2||_inf <= eps_P,
                                eps_P in {0.18, 0.25, 0.26}

Encodings are copy-paste of the P1/P4/P5 scripts:
  TV          : verify_p1_p4_p5_with_tv_bound.add_tv_bound
  Fourier     : verify_p1_p4_p5_with_fourier_bound.add_fourier_bound
  Photometric : verify_p1_p4_p5_with_photometric_basis.add_photometric_bound

Sample selection is identical to verify_p3_mean_constrained.py
(np.linspace over the centered-CTE indices).

Per-query timeout: 300 s.  Output: verification_math_regions_p3_eps0p05.json.

Usage:
  ./prophecy_env/bin/python verify_p3_with_math_regions.py
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
OUTPUT_JSON = "verification_math_regions_p3_eps0p05.json"

NUM_PIXELS = 128
ROWS, COLS = 8, 16
MEAN_TARGET = 0.5
SUM_TARGET = MEAN_TARGET * NUM_PIXELS  # 64.0
PIXEL_MIN = 0.0
PIXEL_MAX = 1.0

CTE_OUT_VAR = 192
HE_OUT_VAR = 193

P3_CTE_BOUND = 10.0
P3_HE_BOUND = 90.0
P3_CENTERED_THRESHOLD = 1.0
EPS_PERTURB = 0.05
NUM_IMAGES = 20
TIMEOUT_S = 300


# -----------------------------------------------------------------------------
# TV encoding (copied from verify_p1_p4_p5_with_tv_bound.py)
# -----------------------------------------------------------------------------
def _neighbor_pairs() -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(ROWS):
        for j in range(COLS - 1):
            pairs.append((i * COLS + j, i * COLS + j + 1))
    for i in range(ROWS - 1):
        for j in range(COLS):
            pairs.append((i * COLS + j, (i + 1) * COLS + j))
    return pairs


_NEIGHBOR_PAIRS = _neighbor_pairs()
assert len(_NEIGHBOR_PAIRS) == ROWS * (COLS - 1) + (ROWS - 1) * COLS == 232


def tv_of_image(flat128: np.ndarray) -> float:
    img = np.asarray(flat128, dtype=np.float64).reshape(ROWS, COLS)
    horiz = np.abs(img[:, :-1] - img[:, 1:]).sum()
    vert = np.abs(img[:-1, :] - img[1:, :]).sum()
    return float(horiz + vert)


def add_tv_bound(net, input_vars: list[int], tau: float) -> None:
    if not math.isfinite(tau):
        return
    d_vars: list[int] = []
    for (a_idx, b_idx) in _NEIGHBOR_PAIRS:
        a = int(input_vars[a_idx])
        b = int(input_vars[b_idx])
        d = net.getNewVariable()
        net.setLowerBound(d, 0.0)
        net.addInequality([a, b, d], [1.0, -1.0, -1.0], 0.0)
        net.addInequality([b, a, d], [1.0, -1.0, -1.0], 0.0)
        d_vars.append(d)
    net.addInequality(d_vars, [1.0] * len(d_vars), float(tau))


# -----------------------------------------------------------------------------
# Fourier high-frequency L1 encoding (copied from verify_p1_p4_p5_with_fourier_bound.py)
# -----------------------------------------------------------------------------
_DFT_NORM = 1.0 / math.sqrt(NUM_PIXELS)


def _high_freq_indices() -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for k in range(ROWS):
        for l in range(COLS):
            if k >= ROWS // 2 or l >= COLS // 2:
                out.append((k, l))
    return out


_HIGH_FREQ = _high_freq_indices()
assert len(_HIGH_FREQ) == NUM_PIXELS - (ROWS // 2) * (COLS // 2) == 96


def _dft_coeff_vectors(k: int, l: int) -> tuple[np.ndarray, np.ndarray]:
    re = np.empty(NUM_PIXELS, dtype=np.float64)
    im = np.empty(NUM_PIXELS, dtype=np.float64)
    for i in range(ROWS):
        for j in range(COLS):
            phase = 2.0 * math.pi * (k * i / ROWS + l * j / COLS)
            p = i * COLS + j
            re[p] = _DFT_NORM * math.cos(phase)
            im[p] = -_DFT_NORM * math.sin(phase)
    return re, im


_HIGH_FREQ_COEFFS: list[tuple[tuple[int, int], np.ndarray, np.ndarray]] = []
for (_k, _l) in _HIGH_FREQ:
    _re, _im = _dft_coeff_vectors(_k, _l)
    _HIGH_FREQ_COEFFS.append(((_k, _l), _re, _im))


def fourier_hf_l1(flat128: np.ndarray) -> float:
    x = np.asarray(flat128, dtype=np.float64).reshape(-1)
    total = 0.0
    for (_kl, re_vec, im_vec) in _HIGH_FREQ_COEFFS:
        cr = float(re_vec @ x)
        ci = float(im_vec @ x)
        total += abs(cr) + abs(ci)
    return total


def add_fourier_bound(net, input_vars: list[int], tau: float) -> None:
    if not math.isfinite(tau):
        return
    d_vars: list[int] = []
    for ((k, l), re_vec, im_vec) in _HIGH_FREQ_COEFFS:
        c_re = net.getNewVariable()
        net.setLowerBound(c_re, -1e6)
        net.setUpperBound(c_re, 1e6)
        eq_vars = list(input_vars) + [c_re]
        eq_coefs = list(re_vec.tolist()) + [-1.0]
        net.addEquality(eq_vars, eq_coefs, 0.0)

        c_im = net.getNewVariable()
        net.setLowerBound(c_im, -1e6)
        net.setUpperBound(c_im, 1e6)
        eq_vars = list(input_vars) + [c_im]
        eq_coefs = list(im_vec.tolist()) + [-1.0]
        net.addEquality(eq_vars, eq_coefs, 0.0)

        d_re = net.getNewVariable()
        net.setLowerBound(d_re, 0.0)
        net.addInequality([c_re, d_re], [1.0, -1.0], 0.0)
        net.addInequality([c_re, d_re], [-1.0, -1.0], 0.0)
        d_vars.append(d_re)

        d_im = net.getNewVariable()
        net.setLowerBound(d_im, 0.0)
        net.addInequality([c_im, d_im], [1.0, -1.0], 0.0)
        net.addInequality([c_im, d_im], [-1.0, -1.0], 0.0)
        d_vars.append(d_im)

    net.addInequality(d_vars, [1.0] * len(d_vars), float(tau))


# -----------------------------------------------------------------------------
# Photometric basis encoding (copied from verify_p1_p4_p5_with_photometric_basis.py)
# -----------------------------------------------------------------------------
def _build_basis() -> tuple[np.ndarray, np.ndarray]:
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


def add_photometric_bound(net, input_vars: list[int], epsilon: float) -> None:
    if not math.isfinite(epsilon):
        return
    alpha = net.getNewVariable()
    beta = net.getNewVariable()
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
        net.addEquality(
            [x_i, int(alpha), int(beta), r_i],
            [1.0, -float(B1[i]), -float(B2[i]), -1.0],
            0.0,
        )


# -----------------------------------------------------------------------------
# Data loading and sample selection (identical to verify_p3_mean_constrained.py)
# -----------------------------------------------------------------------------
def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].astype(np.float64)
        labels = f["X_train"][:].astype(np.float64)
    return images, labels


def select_centered_sample(images, labels) -> np.ndarray:
    cte = labels[:, 0]
    centered_idx = np.where(np.abs(cte) < P3_CENTERED_THRESHOLD)[0]
    sample = centered_idx[np.linspace(0, len(centered_idx) - 1, NUM_IMAGES, dtype=int)]
    return sample


def feasibility_check(x0: np.ndarray, epsilon: float) -> tuple[float, float]:
    lo = np.maximum(PIXEL_MIN, x0 - epsilon)
    hi = np.minimum(PIXEL_MAX, x0 + epsilon)
    return float(lo.sum()), float(hi.sum())


# -----------------------------------------------------------------------------
# Query builder + runner
# -----------------------------------------------------------------------------
def build_query(x0: np.ndarray, epsilon: float, region_kind: str, region_param: float):
    """Per-image L_inf ball + mean(x)=0.5 + math-region constraint."""
    net = Marabou.read_nnet(NNET_PATH)
    input_vars = [int(v) for v in net.inputVars[0].flatten()]
    assert len(input_vars) == NUM_PIXELS
    for i in range(NUM_PIXELS):
        lo = max(PIXEL_MIN, float(x0[i] - epsilon))
        hi = min(PIXEL_MAX, float(x0[i] + epsilon))
        net.setLowerBound(input_vars[i], lo)
        net.setUpperBound(input_vars[i], hi)
    net.addEquality(input_vars, [1.0] * NUM_PIXELS, SUM_TARGET)

    if region_kind == "tv":
        add_tv_bound(net, input_vars, region_param)
    elif region_kind == "fourier":
        add_fourier_bound(net, input_vars, region_param)
    elif region_kind == "photometric":
        add_photometric_bound(net, input_vars, region_param)
    else:
        raise ValueError(f"Unknown region_kind: {region_kind}")
    return net, input_vars


def _record_sat_witness(vals, input_vars: list[int], x0: np.ndarray) -> dict:
    cx = np.array([vals[v] for v in input_vars], dtype=np.float64)
    a, b = alpha_beta(cx)
    return {
        "CTE": round(float(vals[CTE_OUT_VAR]), 6),
        "HE": round(float(vals[HE_OUT_VAR]), 6),
        "witness_mean": round(float(cx.mean()), 6),
        "witness_TV": round(tv_of_image(cx), 6),
        "witness_fourier_hf_l1": round(fourier_hf_l1(cx), 6),
        "witness_residual_linf": round(residual_linf(cx), 6),
        "witness_alpha": round(a, 6),
        "witness_beta": round(b, 6),
        "linf_from_x0": round(float(np.max(np.abs(cx - x0))), 6),
    }


def solve_one(x0: np.ndarray,
              epsilon: float,
              region_kind: str,
              region_param: float,
              out_var: int,
              direction: str,
              bound: float) -> dict:
    net, input_vars = build_query(x0, epsilon, region_kind, region_param)
    if direction == ">=":
        net.setLowerBound(out_var, float(bound))
    else:
        net.setUpperBound(out_var, float(bound))
    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT_S)
    t0 = time.time()
    try:
        status, vals, _ = net.solve(verbose=False, options=opts)
    except Exception as e:
        return {
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "time_seconds": round(time.time() - t0, 4),
        }
    elapsed = time.time() - t0
    status = (status or "").strip().lower()
    if status == "":
        status = "timeout"
    out: dict = {"status": status, "time_seconds": round(elapsed, 4)}
    if status == "sat":
        out.update(_record_sat_witness(vals, input_vars, x0))
    return out


# -----------------------------------------------------------------------------
# Main sweep
# -----------------------------------------------------------------------------
SUB_QUERIES = [
    ("a", CTE_OUT_VAR, ">=",  P3_CTE_BOUND),
    ("b", CTE_OUT_VAR, "<=", -P3_CTE_BOUND),
    ("c", HE_OUT_VAR,  ">=",  P3_HE_BOUND),
    ("d", HE_OUT_VAR,  "<=", -P3_HE_BOUND),
]


REGION_SETTINGS: list[tuple[str, str, float]] = [
    # key,                kind,           param
    ("tv_tau_12",         "tv",          12.0),
    ("tv_tau_13",         "tv",          13.0),
    ("tv_tau_15",         "tv",          15.0),
    ("fourier_tau_7p0",   "fourier",      7.00),
    ("fourier_tau_7p92",  "fourier",      7.92),
    ("fourier_tau_8p16",  "fourier",      8.16),
    ("photometric_eps_0p18", "photometric", 0.18),
    ("photometric_eps_0p25", "photometric", 0.25),
    ("photometric_eps_0p26", "photometric", 0.26),
]


def run_region(images: np.ndarray,
               sample_idx: np.ndarray,
               region_key: str,
               region_kind: str,
               region_param: float) -> dict:
    """Run all 20 images x 4 sub-queries at one math-region setting."""
    print(f"\n=== {region_key} ({region_kind}={region_param}) ===", flush=True)
    per_image: dict = {}
    n_verified = 0
    n_violated = 0
    n_infeasible = 0

    t_region = time.time()
    for k, idx in enumerate(sample_idx):
        x0 = images[int(idx)].flatten().astype(np.float64)
        x0_mean = float(x0.mean())
        lo_sum, hi_sum = feasibility_check(x0, EPS_PERTURB)
        region_feasible = (lo_sum <= SUM_TARGET <= hi_sum)

        entry: dict = {
            "dataset_index": int(idx),
            "x0_mean": round(x0_mean, 6),
            "x0_TV": round(tv_of_image(x0), 6),
            "x0_fourier_hf_l1": round(fourier_hf_l1(x0), 6),
            "x0_residual_linf": round(residual_linf(x0), 6),
            "ball_sum_range": [round(lo_sum, 4), round(hi_sum, 4)],
            "region_feasible": region_feasible,
        }

        if not region_feasible:
            n_infeasible += 1
            entry["sub_queries"] = "skipped (mean+L_inf region empty)"
            entry["verified"] = None
            per_image[f"image_{k+1}"] = entry
            print(
                f"  img{k+1:>2} idx={idx} mean={x0_mean:.4f}  region empty "
                f"(sum range [{lo_sum:.2f}, {hi_sum:.2f}] excludes 64.0)",
                flush=True,
            )
            continue

        sub_results: dict = {}
        all_unsat = True
        for tag, var, direc, b in SUB_QUERIES:
            r = solve_one(x0, EPS_PERTURB, region_kind, region_param,
                          var, direc, b)
            sub_results[tag] = r
            if r["status"] != "unsat":
                all_unsat = False

        entry["sub_queries"] = sub_results
        entry["verified"] = bool(all_unsat)
        per_image[f"image_{k+1}"] = entry

        if all_unsat:
            n_verified += 1
            tag = "VERIFIED"
        else:
            n_violated += 1
            tag = "VIOLATED"
        statuses = " ".join(sub_results[t]["status"][:3] for t, *_ in SUB_QUERIES)
        print(
            f"  img{k+1:>2} idx={idx} mean={x0_mean:.4f}  {tag:<8}  [{statuses}]",
            flush=True,
        )

    summary = {
        "region_kind": region_kind,
        "region_param": region_param,
        "n_verified": n_verified,
        "n_violated": n_violated,
        "n_infeasible": n_infeasible,
        "n_total": NUM_IMAGES,
        "wall_seconds": round(time.time() - t_region, 4),
        "per_image": per_image,
    }
    print(
        f"  => {n_verified}/{NUM_IMAGES} verified, "
        f"{n_violated}/{NUM_IMAGES} violated, "
        f"{n_infeasible}/{NUM_IMAGES} infeasible  "
        f"(wall {summary['wall_seconds']:.1f}s)",
        flush=True,
    )
    return summary


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    print("Loading data...", flush=True)
    images, labels = load_data()
    sample_idx = select_centered_sample(images, labels)
    print(
        f"Selected {len(sample_idx)} centered images, epsilon = {EPS_PERTURB}, "
        f"mean(x) = 0.5 (sum = {SUM_TARGET})",
        flush=True,
    )

    results_by_region: dict = {}
    t0_all = time.time()
    for region_key, region_kind, region_param in REGION_SETTINGS:
        results_by_region[region_key] = run_region(
            images, sample_idx, region_key, region_kind, region_param
        )

    total_wall = time.time() - t0_all

    output = {
        "provenance": {
            "script": "verify_p3_with_math_regions.py",
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "git_commit": _git_commit(),
            "nnet": NNET_PATH,
            "data": DATA_PATH,
            "output_json": OUTPUT_JSON,
            "timeout_seconds_per_query": TIMEOUT_S,
            "mean_target": MEAN_TARGET,
            "sum_target": SUM_TARGET,
            "epsilon_perturbation": EPS_PERTURB,
            "num_images": NUM_IMAGES,
            "p3_cte_bound": P3_CTE_BOUND,
            "p3_he_bound": P3_HE_BOUND,
            "p3_centered_threshold": P3_CENTERED_THRESHOLD,
            "rows": ROWS,
            "cols": COLS,
            "sample_indices": [int(i) for i in sample_idx.tolist()],
            "region_settings": [
                {"key": k, "kind": kind, "param": p}
                for (k, kind, p) in REGION_SETTINGS
            ],
            "total_wall_seconds": round(total_wall, 2),
        },
        "epsilon_perturbation": EPS_PERTURB,
        "baseline_result": {
            "n_verified": 4,
            "n_total": NUM_IMAGES,
            "source": "verification_p3_mean_constrained.json at epsilon=0.05",
        },
        "results_by_region": results_by_region,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}  (total wall {total_wall:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
