"""
Combined math/physics-bounded verification for P1/P4/P5 on TinyTaxiNet.

Hypothesis being tested
-----------------------
The three single-constraint experiments (TV bound, Fourier high-frequency
bound, photometric basis residual) each retained P1/P4/P5 SAT verdicts at
tightness below typical real-image structure (modulo P4-left strong
wrong-sign which collapsed under each). Here we impose ALL THREE
constraints simultaneously: a SAT witness in this region must be

  (a) spatially smooth   (TV(x) <= tau_TV)
  (b) low-frequency-dominated (sum |Re|+|Im| of high-freq DFT <= tau_F)
  (c) photometrically decomposable
       (||x - alpha*b1 - beta*b2||_inf <= eps_P)

If P1a/P1b/P5 SAT verdicts survive even this combined constraint, that
is strong methodology evidence the failure mode is physically reachable:
the SAT verdict can be witnessed by an input simultaneously satisfying
three independent physics-inspired regularity conditions.

Two sweeps:
  Loose: tau_TV=15 (full-dataset max), tau_F=8.16 (centered max),
         eps_P=0.26 (centered max)
  Tight: tau_TV=12 (just below centered p99), tau_F=7.0 (just below
         centered p99), eps_P=0.20 (below centered median 0.208)

Encoding sizes per query (on top of the 128 input vars + 32 ReLUs):
  TV         : 232 aux vars (slacks),  ~465 inequalities, 1 sum-bound
  Fourier HF : 96 coeffs * (c_re,c_im,d_re,d_im) = 384 aux vars,
               192 equalities, 4*96+1=385 inequalities
  Photometric: 2 (alpha,beta) + 128 (r_i) = 130 aux vars, 128 equalities
  Total      : ~746 auxiliary variables on top of the input/ReLU vars.

Per-query timeout 300 s.  Output JSON: verification_math_combined.json.

Usage:
  ./prophecy_env/bin/python verify_p1_p4_p5_combined_math.py
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
OUTPUT_JSON = "verification_math_combined.json"

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
# TV encoding helpers (copied from verify_p1_p4_p5_with_tv_bound.py)
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
assert len(_NEIGHBOR_PAIRS) == 232


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
# Fourier high-freq encoding helpers (copied from fourier script)
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
    BIG = 1e6
    for ((k, l), re_vec, im_vec) in _HIGH_FREQ_COEFFS:
        c_re = net.getNewVariable()
        net.setLowerBound(c_re, -BIG)
        net.setUpperBound(c_re, BIG)
        eq_vars = list(input_vars) + [c_re]
        eq_coefs = list(re_vec.tolist()) + [-1.0]
        net.addEquality(eq_vars, eq_coefs, 0.0)

        c_im = net.getNewVariable()
        net.setLowerBound(c_im, -BIG)
        net.setUpperBound(c_im, BIG)
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
# Photometric basis encoding helpers (copied from photometric script)
# -----------------------------------------------------------------------------
def _build_basis() -> tuple[np.ndarray, np.ndarray]:
    b1 = np.ones(NUM_PIXELS, dtype=np.float64) / math.sqrt(NUM_PIXELS)
    grad = np.zeros((ROWS, COLS), dtype=np.float64)
    for i in range(ROWS):
        for j in range(COLS):
            grad[i, j] = (j - 7.5)
    g = grad.flatten()
    b2 = g / float(np.linalg.norm(g))
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
# Data loading
# -----------------------------------------------------------------------------
def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].astype(np.float64)
        labels = f["X_train"][:].astype(np.float64)
    return images, labels


def compute_box(images: np.ndarray):
    flat = images.reshape(images.shape[0], -1)
    return flat.min(axis=0), flat.max(axis=0)


# -----------------------------------------------------------------------------
# Combined query builder + runner
# -----------------------------------------------------------------------------
def build_query(pixel_mins: np.ndarray,
                pixel_maxs: np.ndarray,
                tau_TV: float,
                tau_F: float,
                eps_P: float):
    net = Marabou.read_nnet(NNET_PATH)
    input_vars = [int(v) for v in net.inputVars[0].flatten()]
    assert len(input_vars) == NUM_PIXELS
    for i in range(NUM_PIXELS):
        net.setLowerBound(input_vars[i], float(pixel_mins[i]))
        net.setUpperBound(input_vars[i], float(pixel_maxs[i]))
    # mean(x) = 0.5
    net.addEquality(input_vars, [1.0] * NUM_PIXELS, SUM_TARGET)
    # All three math/physics constraints
    add_tv_bound(net, input_vars, tau_TV)
    add_fourier_bound(net, input_vars, tau_F)
    add_photometric_bound(net, input_vars, eps_P)
    return net, input_vars


def _record_sat_witness(vals, input_vars: list[int]) -> dict:
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
        "witness": [round(float(v), 6) for v in cx.tolist()],
    }


def run_single_output_bound(name: str,
                            pixel_mins: np.ndarray,
                            pixel_maxs: np.ndarray,
                            tau_TV: float,
                            tau_F: float,
                            eps_P: float,
                            out_var: int,
                            direction: str,
                            bound: float) -> dict:
    net, input_vars = build_query(pixel_mins, pixel_maxs, tau_TV, tau_F, eps_P)
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
        out.update(_record_sat_witness(vals, input_vars))
    return out


def run_p5(pixel_mins: np.ndarray,
           pixel_maxs: np.ndarray,
           tau_TV: float,
           tau_F: float,
           eps_P: float,
           deadzone: float = P5_DEADZONE) -> dict:
    net, input_vars = build_query(pixel_mins, pixel_maxs, tau_TV, tau_F, eps_P)
    net.setLowerBound(CTE_OUT_VAR, -deadzone)
    net.setUpperBound(CTE_OUT_VAR, deadzone)
    net.setLowerBound(HE_OUT_VAR, -deadzone)
    net.setUpperBound(HE_OUT_VAR, deadzone)
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
    out: dict = {
        "status": status,
        "time_seconds": round(elapsed, 4),
        "deadzone": deadzone,
    }
    if status == "sat":
        out.update(_record_sat_witness(vals, input_vars))
    return out


def run_setting(pixel_boxes: dict,
                tau_TV: float,
                tau_F: float,
                eps_P: float) -> dict:
    results: dict = {}
    pmin, pmax = pixel_boxes["P1"]
    results["P1a"] = run_single_output_bound(
        "P1a", pmin, pmax, tau_TV, tau_F, eps_P,
        CTE_OUT_VAR, ">=", P1_CTE_BOUND,
    )
    results["P1b"] = run_single_output_bound(
        "P1b", pmin, pmax, tau_TV, tau_F, eps_P,
        CTE_OUT_VAR, "<=", -P1_CTE_BOUND,
    )
    pmin, pmax = pixel_boxes["P4_left"]
    results["P4_left"] = run_single_output_bound(
        "P4_left", pmin, pmax, tau_TV, tau_F, eps_P,
        CTE_OUT_VAR, "<=", 0.0,
    )
    pmin, pmax = pixel_boxes["P4_right"]
    results["P4_right"] = run_single_output_bound(
        "P4_right", pmin, pmax, tau_TV, tau_F, eps_P,
        CTE_OUT_VAR, ">=", 0.0,
    )
    pmin, pmax = pixel_boxes["P5"]
    results["P5"] = run_p5(
        pmin, pmax, tau_TV, tau_F, eps_P, deadzone=P5_DEADZONE
    )
    return results


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
SETTINGS: dict = {
    "loose": {"tau_TV": 15.0, "tau_F": 8.16, "eps_P": 0.26},
    "tight": {"tau_TV": 12.0, "tau_F": 7.0,  "eps_P": 0.20},
}


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

    results_by_setting: dict = {}
    for setting_name, params in SETTINGS.items():
        tau_TV = float(params["tau_TV"])
        tau_F = float(params["tau_F"])
        eps_P = float(params["eps_P"])
        print(
            f"\n=== Running combined-math sweep [{setting_name}] "
            f"tau_TV={tau_TV} tau_F={tau_F} eps_P={eps_P} ===",
            flush=True,
        )
        t0 = time.time()
        results_by_setting[setting_name] = run_setting(
            pixel_boxes, tau_TV, tau_F, eps_P
        )
        for prop_name, r in results_by_setting[setting_name].items():
            extra = ""
            if r["status"] == "sat":
                extra = (
                    f"  CTE={r['CTE']:+.4f}  HE={r['HE']:+.4f}  "
                    f"mean={r['witness_mean']:.4f}  "
                    f"TV={r['witness_TV']:.3f}  "
                    f"HFL1={r['witness_fourier_hf_l1']:.3f}  "
                    f"resLinf={r['witness_residual_linf']:.4f}"
                )
            print(
                f"  {prop_name:<8} {r['status'].upper():<7} "
                f"t={r['time_seconds']:>7.2f}s{extra}",
                flush=True,
            )
        print(f"  {setting_name} wall: {time.time()-t0:.2f}s", flush=True)

    output = {
        "provenance": {
            "script": "verify_p1_p4_p5_combined_math.py",
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
            "p1_cte_bound": P1_CTE_BOUND,
            "p5_deadzone": P5_DEADZONE,
            "rows": ROWS,
            "cols": COLS,
            "encoding": {
                "tv_pairs": len(_NEIGHBOR_PAIRS),
                "tv_aux_vars_per_query": len(_NEIGHBOR_PAIRS),
                "fourier_high_freq_coeffs": len(_HIGH_FREQ),
                "fourier_aux_vars_per_query": 4 * len(_HIGH_FREQ),
                "fourier_equalities_per_query": 2 * len(_HIGH_FREQ),
                "photometric_aux_vars_per_query": 2 + NUM_PIXELS,
                "photometric_equalities_per_query": NUM_PIXELS,
                "combined_aux_vars_per_query": (
                    len(_NEIGHBOR_PAIRS)
                    + 4 * len(_HIGH_FREQ)
                    + 2 + NUM_PIXELS
                ),
            },
        },
        "settings": SETTINGS,
        "results": results_by_setting,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
