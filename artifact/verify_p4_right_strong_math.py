"""
Experiment 1: P4-right strong wrong-sign sweep under math regions.

We test whether the right-subset (CTE < -2m) per-pixel box admits a
strong-magnitude wrong-sign witness (CTE >= +3) under each of the three
math-encoded regions (TV, Fourier high-freq L1, photometric residual),
at loose and tight tightness levels.

Tightness levels (loose / tight):
  TV:           tau_TV in {15, 12}
  Fourier:      tau_F  in {8.16, 7.0}
  Photometric:  eps_P  in {0.26, 0.20}

Postcondition: CTE >= +3 (strong wrong-sign on right subset)
  -- SAT means the math region admits a witness x in the right-subset
     pixel box with CTE >= +3m, an analogue to the P4-left strong (-3 to
     -4.38m) witnesses observed under AE+mean.

Baseline (no math region) and the existing math-region experiments show
P4-right SAT at boundary CTE=0 only. Here we ask: can the solver find
CTE >= +3 inside the same math regions, or does the strong-magnitude
witness collapse (mirror of P4-left)?

Per-query timeout: 300 s. Output: verification_p4_right_strong_math.json.

Usage:
  ./prophecy_env/bin/python verify_p4_right_strong_math.py
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
OUTPUT_JSON = "verification_p4_right_strong_math.json"

NUM_PIXELS = 128
ROWS, COLS = 8, 16
MEAN_TARGET = 0.5
SUM_TARGET = MEAN_TARGET * NUM_PIXELS  # 64.0

CTE_OUT_VAR = 192
HE_OUT_VAR = 193

# Strong wrong-sign postcondition: CTE >= STRONG_CTE
STRONG_CTE = 3.0

TIMEOUT_S = 300

# Tightness schedule per the spec
SETTINGS = {
    "tv":          {"loose": 15.0, "tight": 12.0},
    "fourier":     {"loose": 8.16, "tight": 7.0},
    "photometric": {"loose": 0.26, "tight": 0.20},
}


# -----------------------------------------------------------------------------
# Data
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
# TV encoding
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
# Fourier high-freq encoding
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
# Photometric basis encoding
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
# Generic query build + solve
# -----------------------------------------------------------------------------
def build_query(pixel_mins: np.ndarray,
                pixel_maxs: np.ndarray,
                region: str,
                tightness: float):
    """region in {tv, fourier, photometric}, tightness is its scalar param."""
    net = Marabou.read_nnet(NNET_PATH)
    input_vars = [int(v) for v in net.inputVars[0].flatten()]
    assert len(input_vars) == NUM_PIXELS
    for i in range(NUM_PIXELS):
        net.setLowerBound(input_vars[i], float(pixel_mins[i]))
        net.setUpperBound(input_vars[i], float(pixel_maxs[i]))
    net.addEquality(input_vars, [1.0] * NUM_PIXELS, SUM_TARGET)
    if region == "tv":
        add_tv_bound(net, input_vars, tightness)
    elif region == "fourier":
        add_fourier_bound(net, input_vars, tightness)
    elif region == "photometric":
        add_photometric_bound(net, input_vars, tightness)
    else:
        raise ValueError(f"unknown region {region}")
    return net, input_vars


def _record_witness(vals, input_vars: list[int]) -> dict:
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


def run_p4_right_strong(pixel_mins: np.ndarray,
                        pixel_maxs: np.ndarray,
                        region: str,
                        tightness: float) -> dict:
    """P4-right strong wrong-sign: CTE >= +STRONG_CTE on right per-pixel box."""
    net, input_vars = build_query(pixel_mins, pixel_maxs, region, tightness)
    net.setLowerBound(CTE_OUT_VAR, STRONG_CTE)
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
        "postcondition": f"CTE >= {STRONG_CTE}",
        "region": region,
        "tightness": tightness,
    }
    if status == "sat":
        out.update(_record_witness(vals, input_vars))
    return out


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
    cte = labels[:, 0]
    flat_images = images.reshape(images.shape[0], -1)
    right = flat_images[cte < -2.0]
    print(f"  right-subset size = {len(right)}", flush=True)
    pmin, pmax = compute_box(right)

    results: dict = {}
    for region, sched in SETTINGS.items():
        for level in ("loose", "tight"):
            key = f"{region}_{level}"
            tightness = sched[level]
            print(
                f"\n=== {key}: region={region} tightness={tightness} ===",
                flush=True,
            )
            t0 = time.time()
            r = run_p4_right_strong(pmin, pmax, region, tightness)
            extra = ""
            if r["status"] == "sat":
                extra = (
                    f"  CTE={r['CTE']:+.4f}  HE={r['HE']:+.4f}  "
                    f"mean={r['witness_mean']:.4f}  TV={r['witness_TV']:.4f}  "
                    f"HF_L1={r['witness_fourier_hf_l1']:.4f}  "
                    f"resLinf={r['witness_residual_linf']:.4f}"
                )
            print(
                f"  P4_right_strong {r['status'].upper():<7} "
                f"t={r['time_seconds']:>7.2f}s{extra}",
                flush=True,
            )
            results[key] = {"P4_right_strong": r}
            print(f"  wall: {time.time()-t0:.2f}s", flush=True)

    output = {
        "provenance": {
            "script": "verify_p4_right_strong_math.py",
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "git_commit": _git_commit(),
            "nnet": NNET_PATH,
            "data": DATA_PATH,
            "timeout_seconds_per_query": TIMEOUT_S,
            "mean_target": MEAN_TARGET,
            "sum_target": SUM_TARGET,
            "strong_cte_threshold": STRONG_CTE,
            "subset": "right (CTE < -2 m)",
            "rows": ROWS,
            "cols": COLS,
            "tv_pairs": len(_NEIGHBOR_PAIRS),
            "fourier_high_freq_coeffs": len(_HIGH_FREQ),
            "photometric_basis_dim": 2,
        },
        "settings": SETTINGS,
        "results": results,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
