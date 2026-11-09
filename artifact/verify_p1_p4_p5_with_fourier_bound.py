"""
Fourier-high-frequency-bounded verification for P1/P4/P5 on TinyTaxiNet.

Hypothesis being tested
-----------------------
Real runway images have power spectra dominated by low spatial frequencies
(roughly 1/f^2). Random pixel-box corners that Marabou might pick as SAT
witnesses have flat or noise-like spectra with substantial high-frequency
content. Bounding the L1-norm of the high-frequency Fourier coefficients
(real and imaginary parts) forces solver witnesses to live in the
low-frequency-dominated regime that real images inhabit.

This experiment is complementary to the TV (total variation) experiment:
TV penalises local non-smoothness, while a Fourier bound penalises any
non-local high-frequency structure. The two constraints are roughly
orthogonal -- one can have low TV with high-frequency content (e.g. a
smooth low-amplitude ripple) or low high-frequency Fourier content with
high TV (e.g. a single sharp step). Whichever property flips reveals
which mathematical structure the SAT witness depended on.

Math
----
For an 8x16 image x[i,j] the 2-D discrete Fourier transform is
    C[k,l] = (1/sqrt(8*16)) sum_{i,j} x[i,j] * exp(-2*pi*i_unit*(k*i/8 + l*j/16)).
The real and imaginary parts are linear in x[i,j]:
    Re C[k,l] = (1/sqrt(128)) sum_{i,j} x[i,j] *  cos(2*pi*(k*i/8 + l*j/16))
    Im C[k,l] = (1/sqrt(128)) sum_{i,j} x[i,j] * -sin(2*pi*(k*i/8 + l*j/16))

High-frequency set:  H = {(k,l) : k >= 4 OR l >= 8}.
8 has Nyquist at 4 and 16 has Nyquist at 8; standard DFT indexing puts
the >Nyquist half in the upper indices, so H captures the high spatial
frequencies. |H| = 128 - 4*8 = 96 coefficients.

Constraint:  sum_{(k,l) in H} (|Re C[k,l]| + |Im C[k,l]|) <= tau.

Encoded with the standard L1 slack trick (Marabou has no abs):
  per coefficient k,l in H:
    c_re_kl = sum_pix coeffs_re[k,l,i,j] * x[i,j]                (equality)
    c_im_kl = sum_pix coeffs_im[k,l,i,j] * x[i,j]                (equality)
    d_re_kl >= 0,  d_re_kl >= c_re_kl,  d_re_kl >= -c_re_kl     (=> |c|)
    d_im_kl >= 0,  d_im_kl >= c_im_kl,  d_im_kl >= -c_im_kl
  one global sum:  sum_(k,l) in H of (d_re_kl + d_im_kl) <= tau

|H| = 96 high-freq coefficients
  ->  96*2 = 192 equality-defined aux vars (c_re, c_im)
  ->  96*2 = 192 slack vars (d_re, d_im)
  ->  96*4 + 1 = 385 inequalities (4 per coeff for slack, +1 sum bound)
  ->  192 equalities

Properties (under per-pixel-box + mean(x) = 0.5 + Fourier(x) <= tau):
  P1a:      CTE >= 10   on centered per-pixel box
  P1b:      CTE <= -10  on centered per-pixel box
  P4_left:  CTE <= 0    on left per-pixel box (wrong-sign query)
  P4_right: CTE >= 0    on right per-pixel box (wrong-sign query)
  P5:       |CTE|<=0.01 AND |HE|<=0.01 on off-center per-pixel box

Per-query timeout: 300 s. Writes JSON to verification_fourier_bound.json.

Usage:
  ./prophecy_env/bin/python verify_p1_p4_p5_with_fourier_bound.py
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
OUTPUT_JSON = "verification_fourier_bound.json"

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
# DFT coefficient matrices (precomputed once)
# -----------------------------------------------------------------------------
_DFT_NORM = 1.0 / math.sqrt(NUM_PIXELS)


def _high_freq_indices() -> list[tuple[int, int]]:
    """High-frequency set H = {(k,l) : k >= 4 OR l >= 8} for an 8x16 grid."""
    out: list[tuple[int, int]] = []
    for k in range(ROWS):
        for l in range(COLS):
            if k >= ROWS // 2 or l >= COLS // 2:
                out.append((k, l))
    return out


_HIGH_FREQ = _high_freq_indices()
# 128 total coefficients minus 4*8 = 32 low-freq -> 96 high-freq
assert len(_HIGH_FREQ) == NUM_PIXELS - (ROWS // 2) * (COLS // 2) == 96


def _dft_coeff_vectors(k: int, l: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (coeffs_re, coeffs_im) of length NUM_PIXELS such that
       Re C[k,l] = sum_p coeffs_re[p] * x[p]  (with p = i*COLS + j)
       Im C[k,l] = sum_p coeffs_im[p] * x[p]."""
    re = np.empty(NUM_PIXELS, dtype=np.float64)
    im = np.empty(NUM_PIXELS, dtype=np.float64)
    for i in range(ROWS):
        for j in range(COLS):
            phase = 2.0 * math.pi * (k * i / ROWS + l * j / COLS)
            p = i * COLS + j
            re[p] = _DFT_NORM * math.cos(phase)
            im[p] = -_DFT_NORM * math.sin(phase)
    return re, im


# Precompute all coefficient vectors for H
_HIGH_FREQ_COEFFS: list[tuple[tuple[int, int], np.ndarray, np.ndarray]] = []
for (_k, _l) in _HIGH_FREQ:
    _re, _im = _dft_coeff_vectors(_k, _l)
    _HIGH_FREQ_COEFFS.append(((_k, _l), _re, _im))


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
# Fourier high-freq L1 calibration (post-hoc on real images and witnesses)
# -----------------------------------------------------------------------------
def fourier_hf_l1(flat128: np.ndarray) -> float:
    """sum over (k,l) in H of (|Re C[k,l]| + |Im C[k,l]|), computed from the
    same DFT convention used in the Marabou encoding. Linear in pixels, so
    matches the linear encoding exactly."""
    x = np.asarray(flat128, dtype=np.float64).reshape(-1)
    total = 0.0
    for (_kl, re_vec, im_vec) in _HIGH_FREQ_COEFFS:
        cr = float(re_vec @ x)
        ci = float(im_vec @ x)
        total += abs(cr) + abs(ci)
    return total


def fourier_quantiles(images_flat: np.ndarray) -> dict:
    qs = np.array([fourier_hf_l1(x) for x in images_flat], dtype=np.float64)
    return {
        "n": int(len(qs)),
        "min": round(float(qs.min()), 6),
        "p01": round(float(np.percentile(qs, 1)), 6),
        "p50": round(float(np.percentile(qs, 50)), 6),
        "p90": round(float(np.percentile(qs, 90)), 6),
        "p99": round(float(np.percentile(qs, 99)), 6),
        "max": round(float(qs.max()), 6),
        "mean": round(float(qs.mean()), 6),
        "std": round(float(qs.std()), 6),
    }


# -----------------------------------------------------------------------------
# Fourier high-frequency L1 encoding into a Marabou query
# -----------------------------------------------------------------------------
def add_fourier_bound(net, input_vars: list[int], tau: float) -> None:
    """For each high-frequency coefficient (k,l) introduce
         c_re, c_im  (defined by linear equalities in x)
         d_re, d_im >= 0  with  d_re >= |c_re|,  d_im >= |c_im|
       and one sum bound  sum(d_re + d_im) <= tau."""
    if not math.isfinite(tau):
        return  # tau = inf -> no constraint added

    d_vars: list[int] = []
    for ((k, l), re_vec, im_vec) in _HIGH_FREQ_COEFFS:
        # c_re_kl = sum_p re_vec[p] * x[p]
        c_re = net.getNewVariable()
        # Marabou variables are unbounded by default; explicitly mark these
        # auxiliaries as free so the solver knows they are unconstrained
        # except by the equality / slack inequalities we add below.
        # (Some Marabou versions assume new variables are bounded; setting
        # very loose bounds is safe.)
        net.setLowerBound(c_re, -1e6)
        net.setUpperBound(c_re, 1e6)
        # sum re_vec[p]*x[p] - c_re = 0
        eq_vars = list(input_vars) + [c_re]
        eq_coefs = list(re_vec.tolist()) + [-1.0]
        net.addEquality(eq_vars, eq_coefs, 0.0)

        c_im = net.getNewVariable()
        net.setLowerBound(c_im, -1e6)
        net.setUpperBound(c_im, 1e6)
        eq_vars = list(input_vars) + [c_im]
        eq_coefs = list(im_vec.tolist()) + [-1.0]
        net.addEquality(eq_vars, eq_coefs, 0.0)

        # Slack: d_re >= |c_re|
        d_re = net.getNewVariable()
        net.setLowerBound(d_re, 0.0)
        # d_re >= c_re  <=>  c_re - d_re <= 0
        net.addInequality([c_re, d_re], [1.0, -1.0], 0.0)
        # d_re >= -c_re  <=>  -c_re - d_re <= 0
        net.addInequality([c_re, d_re], [-1.0, -1.0], 0.0)
        d_vars.append(d_re)

        # Slack: d_im >= |c_im|
        d_im = net.getNewVariable()
        net.setLowerBound(d_im, 0.0)
        net.addInequality([c_im, d_im], [1.0, -1.0], 0.0)
        net.addInequality([c_im, d_im], [-1.0, -1.0], 0.0)
        d_vars.append(d_im)

    # sum(d_re + d_im) <= tau  (one inequality across all high-freq slacks)
    net.addInequality(d_vars, [1.0] * len(d_vars), float(tau))


# -----------------------------------------------------------------------------
# Query builder + runner
# -----------------------------------------------------------------------------
def build_query(pixel_mins: np.ndarray, pixel_maxs: np.ndarray, tau: float):
    net = Marabou.read_nnet(NNET_PATH)
    input_vars = [int(v) for v in net.inputVars[0].flatten()]
    assert len(input_vars) == NUM_PIXELS
    for i in range(NUM_PIXELS):
        net.setLowerBound(input_vars[i], float(pixel_mins[i]))
        net.setUpperBound(input_vars[i], float(pixel_maxs[i]))
    # mean(x) = 0.5
    net.addEquality(input_vars, [1.0] * NUM_PIXELS, SUM_TARGET)
    add_fourier_bound(net, input_vars, tau)
    return net, input_vars


def _record_sat_witness(vals, input_vars: list[int]) -> dict:
    cx = np.array([vals[v] for v in input_vars], dtype=np.float64)
    return {
        "CTE": round(float(vals[CTE_OUT_VAR]), 6),
        "HE": round(float(vals[HE_OUT_VAR]), 6),
        "witness_mean": round(float(cx.mean()), 6),
        "witness_fourier_hf_l1": round(fourier_hf_l1(cx), 6),
        "witness": [round(float(v), 6) for v in cx.tolist()],
    }


def run_single_output_bound(name: str,
                            pixel_mins: np.ndarray,
                            pixel_maxs: np.ndarray,
                            tau: float,
                            out_var: int,
                            direction: str,
                            bound: float) -> dict:
    net, input_vars = build_query(pixel_mins, pixel_maxs, tau)
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
           tau: float,
           deadzone: float = P5_DEADZONE) -> dict:
    net, input_vars = build_query(pixel_mins, pixel_maxs, tau)
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


# -----------------------------------------------------------------------------
# Main sweep
# -----------------------------------------------------------------------------
def tau_key(tau: float, label: str) -> str:
    return f"tau_{label}"


def run_tau(pixel_boxes: dict, tau: float) -> dict:
    results: dict = {}
    pmin, pmax = pixel_boxes["P1"]
    results["P1a"] = run_single_output_bound(
        "P1a", pmin, pmax, tau, CTE_OUT_VAR, ">=", P1_CTE_BOUND
    )
    results["P1b"] = run_single_output_bound(
        "P1b", pmin, pmax, tau, CTE_OUT_VAR, "<=", -P1_CTE_BOUND
    )
    pmin, pmax = pixel_boxes["P4_left"]
    results["P4_left"] = run_single_output_bound(
        "P4_left", pmin, pmax, tau, CTE_OUT_VAR, "<=", 0.0
    )
    pmin, pmax = pixel_boxes["P4_right"]
    results["P4_right"] = run_single_output_bound(
        "P4_right", pmin, pmax, tau, CTE_OUT_VAR, ">=", 0.0
    )
    pmin, pmax = pixel_boxes["P5"]
    results["P5"] = run_p5(pmin, pmax, tau, deadzone=P5_DEADZONE)
    return results


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

    print("Computing Fourier-HF-L1 calibration quantiles...", flush=True)
    centered_q = fourier_quantiles(centered)
    full_q = fourier_quantiles(flat_images)
    fourier_calibration = {
        "centered_quantiles": centered_q,
        "full_dataset_quantiles": full_q,
        "n_high_freq_coefficients": len(_HIGH_FREQ),
        "high_freq_set_definition": "{(k,l) : k >= 4 OR l >= 8}",
        "dft_normalization": "orthonormal: 1/sqrt(N) = 1/sqrt(128)",
    }
    print(
        f"  centered HF-L1 [min={centered_q['min']}, p99={centered_q['p99']}, "
        f"max={centered_q['max']}]",
        flush=True,
    )
    print(
        f"  full HF-L1     [min={full_q['min']}, p99={full_q['p99']}, "
        f"max={full_q['max']}]",
        flush=True,
    )

    # Pick the tau sweep per the spec:
    #   inf      -> no constraint (baseline reproduction)
    #   p99      -> 99th percentile of centered
    #   max      -> max of centered
    #   tight    -> one tighter value below the median (we use p50/2)
    tau_p99 = float(centered_q["p99"])
    tau_max = float(centered_q["max"])
    tau_tight = float(centered_q["p50"]) / 2.0

    tau_sweep = [
        ("inf", math.inf),
        ("99p", tau_p99),
        ("max", tau_max),
        ("tight", tau_tight),
    ]

    results_by_tau: dict = {}
    for label, tau in tau_sweep:
        key = f"tau_{label}"
        print(
            f"\n=== Running sweep at {key} (tau={tau}) ===",
            flush=True,
        )
        t0 = time.time()
        results_by_tau[key] = {
            "tau_value": ("inf" if not math.isfinite(tau) else round(tau, 6)),
            "queries": run_tau(pixel_boxes, tau),
        }
        for prop_name, r in results_by_tau[key]["queries"].items():
            extra = ""
            if r["status"] == "sat":
                extra = (
                    f"  CTE={r['CTE']:+.4f}  HE={r['HE']:+.4f}  "
                    f"mean={r['witness_mean']:.4f}  "
                    f"HF_L1={r['witness_fourier_hf_l1']:.4f}"
                )
            print(
                f"  {prop_name:<8} {r['status'].upper():<7} "
                f"t={r['time_seconds']:>7.2f}s{extra}",
                flush=True,
            )
        print(f"  tau wall: {time.time()-t0:.2f}s", flush=True)

    output = {
        "provenance": {
            "script": "verify_p1_p4_p5_with_fourier_bound.py",
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
            "tau_sweep": [
                ("inf" if not math.isfinite(t) else round(t, 6))
                for (_lab, t) in tau_sweep
            ],
            "tau_sweep_labels": [lab for (lab, _t) in tau_sweep],
            "fourier_encoding_high_freq_coeffs": len(_HIGH_FREQ),
            "fourier_encoding_aux_vars_per_query": 4 * len(_HIGH_FREQ),
            "fourier_encoding_equalities_per_query": 2 * len(_HIGH_FREQ),
            "fourier_encoding_inequalities_per_query": 4 * len(_HIGH_FREQ) + 1,
            "rows": ROWS,
            "cols": COLS,
        },
        "fourier_calibration": fourier_calibration,
        "results": results_by_tau,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
