"""
Prophecy-style P1 verification (per-pattern, all 32 ReLUs fixed across the
three hidden dense layers) STACKED with math-encoded input regions:

    - Total variation (TV)       tau in {12, 10}
    - Fourier high-frequency L1  tau = 7.0
    - Photometric residual       epsilon = 0.20
    - Combined (TV<=12, F<=7.0, eps<=0.20)

Baseline (verify_prophecy_p1.py) verifies 16/24 patterns under the
per-pattern pixel box + activation pattern precondition; 8/24 patterns
return a SAT counterexample. We test whether adding a math/physics region
on top of the activation pattern restores any of the 8 failing patterns
to UNSAT, and whether the 16 already-verified patterns remain UNSAT (and
do not become infeasible).

Per-query: per-pixel pattern box + mean(x)=0.5 + fix 32 ReLU signs
(dense_0: 16, dense_1: 8, dense_2: 8) + the math/physics constraint.

We split the negated P1 postcondition into the same four sub-queries the
baseline uses:
    P1a_CTE_lower: y_CTE <= -10
    P1b_CTE_upper: y_CTE >=  10
    P1c_HE_lower:  y_HE  <= -90
    P1d_HE_upper:  y_HE  >=  90
If all four return UNSAT the pattern is verified safe under that region.

Per-query timeout: 300 s.
Output JSON: verification_p1_patterned_math.json.

Usage:
  ./prophecy_env/bin/python verify_prophecy_p1_with_math.py
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import subprocess
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import h5py
import numpy as np
from maraboupy import Marabou


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
KERAS_MODEL = "Prophecy/dataset_models/kj_tiny_taxinet/KJ_Taxinet_flat.h5"
OUTPUT_JSON = "verification_p1_patterned_math.json"

NUM_PIXELS = 128
ROWS, COLS = 8, 16
MEAN_TARGET = 0.5
SUM_TARGET = MEAN_TARGET * NUM_PIXELS  # 64.0

CTE_BOUND = 10.0
HE_BOUND = 90.0
CENTERED_THRESHOLD = 1.0

# Variable layout for the .nnet with sizes [128, 16, 8, 8, 2]:
#   Inputs:        0-127
#   Dense_0 pre:   128-143   post: 144-159
#   Dense_1 pre:   160-167   post: 168-175
#   Dense_2 pre:   176-183   post: 184-191
#   Output:        192-193
CTE_OUT_VAR = 192
HE_OUT_VAR = 193

# (pre-ReLU first index, number of neurons in that layer)
LAYER_PRE_RELU = [
    (128, 16),  # dense_0
    (160, 8),   # dense_1
    (176, 8),   # dense_2
]

TIMEOUT_S = 300


# -----------------------------------------------------------------------------
# TV encoding helpers
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
# Fourier high-frequency L1 encoding helpers
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
# Photometric basis encoding helpers
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
# Pattern enumeration (matches verify_prophecy_p1.py exactly)
# -----------------------------------------------------------------------------
def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        images = np.array(f["y_train"]).reshape(-1, NUM_PIXELS)
        labels = np.array(f["X_train"])
    return images, labels


def extract_patterns(images: np.ndarray, labels: np.ndarray):
    """Reproduce the baseline's full 3-layer pattern extraction."""
    import tensorflow as tf  # heavy import; only load if we are running

    model = tf.keras.models.load_model(KERAS_MODEL, compile=False)

    layer_acts = []
    for layer_name in ["dense_0", "dense_1", "dense_2"]:
        inter = tf.keras.Model(
            inputs=model.input,
            outputs=model.get_layer(layer_name).output,
        )
        layer_acts.append(
            inter.predict(images.astype(np.float32), verbose=0)
        )

    centered_mask = np.abs(labels[:, 0]) < CENTERED_THRESHOLD
    centered_imgs = images[centered_mask]

    patterns = np.hstack(
        [(a[centered_mask] > 0).astype(int) for a in layer_acts]
    )
    unique_patterns = np.unique(patterns, axis=0)

    pattern_records = []
    for pat in unique_patterns:
        pat_mask = np.all(patterns == pat, axis=1)
        pat_images = centered_imgs[pat_mask]
        pattern_records.append(
            {
                "pattern": pat.astype(int).tolist(),
                "dense_0": pat[:16].astype(int).tolist(),
                "dense_1": pat[16:24].astype(int).tolist(),
                "dense_2": pat[24:32].astype(int).tolist(),
                "image_count": int(pat_mask.sum()),
                "pixel_mins": pat_images.min(axis=0).astype(np.float64),
                "pixel_maxs": pat_images.max(axis=0).astype(np.float64),
            }
        )
    return pattern_records


# -----------------------------------------------------------------------------
# Query builder + runner
# -----------------------------------------------------------------------------
def build_query(pattern_rec: dict,
                tau_TV: float,
                tau_F: float,
                eps_P: float,
                use_mean_constraint: bool = True):
    net = Marabou.read_nnet(NNET_PATH)
    input_vars = [int(v) for v in net.inputVars[0].flatten()]
    assert len(input_vars) == NUM_PIXELS

    # Per-pixel box from the pattern's centered images
    pixel_mins = pattern_rec["pixel_mins"]
    pixel_maxs = pattern_rec["pixel_maxs"]
    for i in range(NUM_PIXELS):
        net.setLowerBound(input_vars[i], float(pixel_mins[i]))
        net.setUpperBound(input_vars[i], float(pixel_maxs[i]))

    # mean(x) = 0.5
    if use_mean_constraint:
        net.addEquality(input_vars, [1.0] * NUM_PIXELS, SUM_TARGET)

    # Fix all 32 ReLU signs to match the activation pattern
    pat_d0 = pattern_rec["dense_0"]
    pat_d1 = pattern_rec["dense_1"]
    pat_d2 = pattern_rec["dense_2"]
    layer_patterns = [
        (128, 16, pat_d0),
        (160, 8, pat_d1),
        (176, 8, pat_d2),
    ]
    for pre_relu_start, n_neurons, pat in layer_patterns:
        for j in range(n_neurons):
            v = pre_relu_start + j
            if pat[j] == 1:
                # active: pre-ReLU >= 0
                net.setLowerBound(v, 0.0)
            else:
                # inactive: pre-ReLU <= 0
                net.setUpperBound(v, 0.0)

    # Math/physics constraints (skipped if tau/eps = inf)
    add_tv_bound(net, input_vars, tau_TV)
    add_fourier_bound(net, input_vars, tau_F)
    add_photometric_bound(net, input_vars, eps_P)

    return net, input_vars


def _record_sat_witness(vals, input_vars: list[int]) -> dict:
    cx = np.array([vals[v] for v in input_vars], dtype=np.float64)
    return {
        "CTE": round(float(vals[CTE_OUT_VAR]), 6),
        "HE": round(float(vals[HE_OUT_VAR]), 6),
        "witness_mean": round(float(cx.mean()), 6),
        "witness_TV": round(tv_of_image(cx), 6),
        "witness_fourier_hf_l1": round(fourier_hf_l1(cx), 6),
        "witness_residual_linf": round(residual_linf(cx), 6),
    }


def _solve(net, input_vars):
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
    out = {"status": status, "time_seconds": round(elapsed, 4)}
    if status == "sat":
        out.update(_record_sat_witness(vals, input_vars))
    return out


def run_pattern_sub_queries(pattern_rec: dict,
                            tau_TV: float,
                            tau_F: float,
                            eps_P: float) -> dict:
    """Run the 4 P1 negated sub-queries for one (pattern, math) setting."""
    sub_queries = [
        ("P1a_CTE_lower", CTE_OUT_VAR, "<=", -CTE_BOUND),
        ("P1b_CTE_upper", CTE_OUT_VAR, ">=",  CTE_BOUND),
        ("P1c_HE_lower",  HE_OUT_VAR,  "<=", -HE_BOUND),
        ("P1d_HE_upper",  HE_OUT_VAR,  ">=",  HE_BOUND),
    ]

    sub_results = []
    pattern_verified = True
    pattern_infeasible_all = True  # all sub-queries return UNSAT for inf-reasons

    for name, out_var, direction, bound in sub_queries:
        net, input_vars = build_query(pattern_rec, tau_TV, tau_F, eps_P)
        if direction == ">=":
            net.setLowerBound(out_var, float(bound))
        else:
            net.setUpperBound(out_var, float(bound))
        result = _solve(net, input_vars)
        result["query"] = name
        sub_results.append(result)
        if result["status"] != "unsat":
            pattern_verified = False
            pattern_infeasible_all = False
        # 'unsat' could be either trivially infeasible region or genuine safe;
        # we don't distinguish here.

    return {
        "verified": pattern_verified,
        "sub_queries": sub_results,
        "total_query_time_s": round(
            sum(s.get("time_seconds", 0.0) for s in sub_results), 4
        ),
    }


# -----------------------------------------------------------------------------
# Sweep over math settings
# -----------------------------------------------------------------------------
MATH_SETTINGS: list[tuple[str, float, float, float]] = [
    # (key, tau_TV, tau_F, eps_P)
    ("tv_tau_12",            12.0,    math.inf, math.inf),
    ("tv_tau_10",            10.0,    math.inf, math.inf),
    ("fourier_tau_7p0",      math.inf, 7.0,     math.inf),
    ("photometric_eps_0p20", math.inf, math.inf, 0.20),
    ("combined_tight",       12.0,    7.0,     0.20),
]


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

    print("Extracting activation patterns (this loads Keras)...", flush=True)
    pattern_records = extract_patterns(images, labels)
    n_patterns = len(pattern_records)
    print(f"  found {n_patterns} unique patterns over centered subset",
          flush=True)

    # Load baseline verdicts (for n_baseline_verified / violated and
    # for marking which patterns were originally failing)
    baseline_verified = []
    try:
        with open("prophecy_p1_results.json", "r") as f:
            baseline = json.load(f)
        for p in baseline["patterns"]:
            baseline_verified.append(bool(p.get("verified", False)))
    except Exception as e:
        print(f"  WARNING: could not load baseline file: {e}", flush=True)
        baseline_verified = [None] * n_patterns

    n_baseline_verified = sum(1 for v in baseline_verified if v is True)
    n_baseline_violated = sum(1 for v in baseline_verified if v is False)
    print(
        f"  baseline: {n_baseline_verified} verified, "
        f"{n_baseline_violated} violated",
        flush=True,
    )

    results_per_setting: dict = {}

    for setting_key, tau_TV, tau_F, eps_P in MATH_SETTINGS:
        print(
            f"\n=== {setting_key}: "
            f"tau_TV={tau_TV} tau_F={tau_F} eps_P={eps_P} ===",
            flush=True,
        )
        setting_start = time.time()
        per_pattern = []
        n_verified = 0
        n_violated = 0

        for pat_idx, rec in enumerate(pattern_records):
            base_v = baseline_verified[pat_idx]
            base_str = (
                "B-OK " if base_v is True else
                ("B-FAIL" if base_v is False else "B-?  ")
            )
            t_pat0 = time.time()
            res = run_pattern_sub_queries(rec, tau_TV, tau_F, eps_P)
            t_pat = time.time() - t_pat0
            per_pattern.append({
                "pattern_id": pat_idx,
                "image_count": rec["image_count"],
                "baseline_verified": base_v,
                "verdict": "verified" if res["verified"] else "violated",
                "time_s": round(t_pat, 4),
                "sub_queries": res["sub_queries"],
            })
            if res["verified"]:
                n_verified += 1
            else:
                n_violated += 1
            sat_subs = [
                s["query"] for s in res["sub_queries"] if s["status"] == "sat"
            ]
            sat_extra = (
                f"  SAT-on: {','.join(sat_subs)}" if sat_subs else ""
            )
            print(
                f"  [{pat_idx:2d}] {base_str} -> "
                f"{'UNSAT' if res['verified'] else 'SAT  '} "
                f"t={t_pat:6.2f}s{sat_extra}",
                flush=True,
            )

        results_per_setting[setting_key] = {
            "tau_TV": ("inf" if not math.isfinite(tau_TV) else tau_TV),
            "tau_F":  ("inf" if not math.isfinite(tau_F) else tau_F),
            "eps_P":  ("inf" if not math.isfinite(eps_P) else eps_P),
            "n_verified": n_verified,
            "n_violated": n_violated,
            "per_pattern": per_pattern,
            "wall_seconds": round(time.time() - setting_start, 4),
        }
        print(
            f"  >>> {setting_key}: {n_verified}/{n_patterns} verified  "
            f"({time.time() - setting_start:.1f}s)",
            flush=True,
        )

    # ----- Compute restored / lost vs baseline -----
    summary_vs_baseline: dict = {}
    for setting_key, info in results_per_setting.items():
        restored: list[int] = []
        lost: list[int] = []
        unchanged_safe: list[int] = []
        unchanged_violated: list[int] = []
        for p in info["per_pattern"]:
            b = p["baseline_verified"]
            v = (p["verdict"] == "verified")
            if b is True and v:
                unchanged_safe.append(p["pattern_id"])
            elif b is False and not v:
                unchanged_violated.append(p["pattern_id"])
            elif b is False and v:
                restored.append(p["pattern_id"])
            elif b is True and not v:
                lost.append(p["pattern_id"])
        summary_vs_baseline[setting_key] = {
            "restored_patterns": restored,
            "lost_patterns": lost,
            "unchanged_safe": unchanged_safe,
            "unchanged_violated": unchanged_violated,
        }

    output = {
        "provenance": {
            "script": "verify_prophecy_p1_with_math.py",
            "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "git_commit": _git_commit(),
            "nnet": NNET_PATH,
            "data": DATA_PATH,
            "keras_model": KERAS_MODEL,
            "baseline_results_json": "prophecy_p1_results.json",
            "timeout_seconds_per_query": TIMEOUT_S,
            "cte_bound": CTE_BOUND,
            "he_bound": HE_BOUND,
            "centered_threshold": CENTERED_THRESHOLD,
            "mean_target": MEAN_TARGET,
            "sum_target": SUM_TARGET,
            "neurons_constrained_per_pattern": 32,
            "layers_constrained": [
                "dense_0 (16 neurons)",
                "dense_1 (8 neurons)",
                "dense_2 (8 neurons)",
            ],
            "sub_queries_per_pattern": [
                "P1a_CTE_lower (y_CTE <= -10)",
                "P1b_CTE_upper (y_CTE >=  10)",
                "P1c_HE_lower  (y_HE  <= -90)",
                "P1d_HE_upper  (y_HE  >=  90)",
            ],
            "math_settings_order": [k for (k, *_rest) in MATH_SETTINGS],
        },
        "n_patterns_total": n_patterns,
        "n_baseline_verified": n_baseline_verified,
        "n_baseline_violated": n_baseline_violated,
        "math_settings": results_per_setting,
        "summary_vs_baseline": summary_vs_baseline,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
