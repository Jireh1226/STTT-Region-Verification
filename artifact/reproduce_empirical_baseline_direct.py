#!/usr/bin/env python3
"""Direct TinyTaxiNet empirical baseline reproduction.

This reproduces the key empirical-baseline counts without using maraboupy.
It parses the released `.nnet` weights and runs the same feed-forward ReLU MLP
on `SK_DownsampledGANFocusAreaData.h5`.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
NNET = ROOT / "VerifyGAN/models/TinyTaxiNet.nnet"
OUT = ROOT / "reproduced_empirical_baseline_direct.json"


def load_nnet(path: Path):
    lines = path.read_text().splitlines()
    i = 0
    while lines[i].startswith("//"):
        i += 1
    header = [int(x) for x in lines[i].split(",") if x.strip()]
    num_layers = header[0]
    i += 1
    sizes = [int(x) for x in lines[i].split(",") if x.strip()]
    i += 1
    i += 5
    weights, biases = [], []
    for layer in range(num_layers):
        rows, cols = sizes[layer + 1], sizes[layer]
        weight = np.zeros((rows, cols), dtype=np.float64)
        for r in range(rows):
            vals = [float(x) for x in lines[i].strip().rstrip(",").split(",")]
            weight[r] = vals[:cols]
            i += 1
        bias = np.zeros(rows, dtype=np.float64)
        for r in range(rows):
            bias[r] = float(lines[i].strip().rstrip(","))
            i += 1
        weights.append(weight)
        biases.append(bias)
    return weights, biases


def forward(x: np.ndarray, weights, biases) -> np.ndarray:
    h = x
    for idx, (weight, bias) in enumerate(zip(weights, biases)):
        h = h @ weight.T + bias
        if idx < len(weights) - 1:
            h = np.maximum(h, 0.0)
    return h


def wilson(k: int, n: int, z: float = 1.959963984540054):
    phat = k / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n) / denom
    return center - margin, center + margin


def wrong_sign(true: np.ndarray, pred: np.ndarray, dead_eps: float):
    mask = np.abs(true) > dead_eps
    wrong = np.sign(true[mask]) != np.sign(pred[mask])
    return int(wrong.sum()), int(mask.sum()), float(wrong.mean())


def main() -> None:
    with h5py.File(DATA, "r") as f:
        images = f["y_train"][:].reshape(-1, 128).astype(np.float64)
        labels = f["X_train"][:].astype(np.float64)
    weights, biases = load_nnet(NNET)
    pred = forward(images, weights, biases)
    true_cte, true_he = labels[:, 0], labels[:, 1]
    pred_cte, pred_he = pred[:, 0], pred[:, 1]
    cte_err = pred_cte - true_cte
    he_err = pred_he - true_he

    centered = np.abs(true_cte) < 1.0
    near = (np.abs(true_cte) > 0.5) & (np.abs(true_cte) < 1.0)
    offset_2 = np.abs(true_cte) > 2.0
    far_5 = np.abs(true_cte) > 5.0
    left = true_cte > 2.0
    right = true_cte < -2.0

    near_wrong, near_n, near_rate = wrong_sign(true_cte[near], pred_cte[near], 0.0)
    off_wrong, off_n, off_rate = wrong_sign(true_cte[offset_2], pred_cte[offset_2], 0.0)
    far_wrong, far_n, far_rate = wrong_sign(true_cte[far_5], pred_cte[far_5], 0.0)
    left_wrong, left_n, left_rate = wrong_sign(true_cte[left], pred_cte[left], 0.0)
    right_wrong, right_n, right_rate = wrong_sign(true_cte[right], pred_cte[right], 0.0)

    result = {
        "provenance": {
            "script": "reproduce_empirical_baseline_direct.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "method": "direct .nnet parser and vectorized ReLU forward pass",
        },
        "counts": {
            "n_images": int(len(images)),
            "centered_abs_cte_lt_1": int(centered.sum()),
            "near_centerline_0p5_lt_abs_cte_lt_1": near_n,
            "offset_abs_cte_gt_2": off_n,
            "far_abs_cte_gt_5": far_n,
            "left_cte_gt_2": left_n,
            "right_cte_lt_minus_2": right_n,
        },
        "wrong_sign": {
            "near_centerline": {
                "wrong": near_wrong,
                "n": near_n,
                "rate": near_rate,
                "wilson95": wilson(near_wrong, near_n),
            },
            "offset_abs_cte_gt_2": {
                "wrong": off_wrong,
                "n": off_n,
                "rate": off_rate,
                "wilson95": wilson(off_wrong, off_n),
            },
            "far_abs_cte_gt_5": {
                "wrong": far_wrong,
                "n": far_n,
                "rate": far_rate,
                "wilson95": wilson(far_wrong, far_n),
            },
            "left_cte_gt_2": {"wrong": left_wrong, "n": left_n, "rate": left_rate},
            "right_cte_lt_minus_2": {"wrong": right_wrong, "n": right_n, "rate": right_rate},
        },
        "error_extrema": {
            "centered_max_abs_cte_error": float(np.abs(cte_err[centered]).max()),
            "full_max_abs_cte_error": float(np.abs(cte_err).max()),
            "full_max_abs_he_error": float(np.abs(he_err).max()),
            "centered_prediction_cte_range": [
                float(pred_cte[centered].min()),
                float(pred_cte[centered].max()),
            ],
            "centered_prediction_he_range": [
                float(pred_he[centered].min()),
                float(pred_he[centered].max()),
            ],
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

