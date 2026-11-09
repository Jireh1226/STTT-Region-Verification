#!/usr/bin/env python3
"""Fifth-fake-family classifier stress test.

The TinyTaxiNet plausibility classifier was trained on four negative families:
box corners, cross-image blends, uniform noise, and high-frequency patterns.
This script scores a fifth, unseen fake family: smooth low-frequency
mean-normalized images that are not runway images. The point is not to create
a physically realistic alternative; it is to test whether the classifier's
"real" score generalizes beyond the exact fake mechanisms it saw.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "classifier_fifth_fake_family_audit.json"


def load_classifier(name: str):
    """Load a ReLU plausibility classifier from .npz.

    Both released classifiers (original and disjoint-split holdout) store the
    same six weight arrays. The disjoint-split file additionally stores
    holdout_idx/train_idx; we read only the weights. (The earlier version of
    this script guarded the load with a 5 s SIGALRM, but the first np.load in
    a fresh process pays a one-time cold-cache cost of ~7-8 s on this machine,
    which tripped that alarm and silently skipped the disjoint-split
    classifier. The load itself is sub-millisecond once warm.)"""
    w = np.load(ROOT / name)
    return [
        (w["c1_W"].astype(np.float64), w["c1_b"].astype(np.float64)),
        (w["c2_W"].astype(np.float64), w["c2_b"].astype(np.float64)),
        (w["score_W"].astype(np.float64), w["score_b"].astype(np.float64)),
    ]


def classify(x: np.ndarray, layers) -> np.ndarray:
    h = x.astype(np.float64)
    for idx, (weight, bias) in enumerate(layers):
        h = h @ weight + bias
        if idx < len(layers) - 1:
            h = np.maximum(h, 0.0)
    logits = np.clip(h.reshape(-1), -30, 30)
    return 1.0 / (1.0 + np.exp(-logits))


def normalize_mean_half(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True) + 0.5
    return np.clip(x, 0.0, 1.0)


def make_smooth_fakes(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    rows = np.linspace(-1.0, 1.0, 8)
    cols = np.linspace(-1.0, 1.0, 16)
    yy, xx = np.meshgrid(rows, cols, indexing="ij")

    families: dict[str, list[np.ndarray]] = {
        "smooth_sinusoids": [],
        "smooth_gaussian_blobs": [],
        "row_band_gradients": [],
    }
    for _ in range(n):
        phase_x, phase_y = rng.uniform(0, 2 * np.pi, 2)
        freq_x = rng.uniform(0.5, 2.0)
        freq_y = rng.uniform(0.5, 2.0)
        img = (
            0.5
            + 0.22 * np.sin(freq_x * np.pi * xx + phase_x)
            + 0.18 * np.cos(freq_y * np.pi * yy + phase_y)
            + 0.08 * xx
        )
        families["smooth_sinusoids"].append(img)

        cx1, cy1 = rng.uniform(-0.8, 0.8, 2)
        cx2, cy2 = rng.uniform(-0.8, 0.8, 2)
        s1, s2 = rng.uniform(0.25, 0.7, 2)
        blob1 = np.exp(-((xx - cx1) ** 2 + (yy - cy1) ** 2) / (2 * s1 * s1))
        blob2 = np.exp(-((xx - cx2) ** 2 + (yy - cy2) ** 2) / (2 * s2 * s2))
        img = 0.35 + 0.45 * blob1 - 0.25 * blob2 + 0.08 * yy
        families["smooth_gaussian_blobs"].append(img)

        row_profile = np.interp(np.arange(8), [0, 2, 5, 7], rng.uniform(0.2, 0.8, 4))
        col_profile = np.linspace(rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), 16)
        img = row_profile[:, None] + col_profile[None, :]
        families["row_band_gradients"].append(img)

    out = {}
    for name, imgs in families.items():
        arr = np.stack([img.reshape(-1) for img in imgs]).astype(np.float64)
        out[name] = normalize_mean_half(arr)
    out["combined"] = np.concatenate(list(out.values()), axis=0)
    return out


def stats(probs: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(probs)),
        "mean": float(np.mean(probs)),
        "median": float(np.median(probs)),
        "p05": float(np.percentile(probs, 5)),
        "p95": float(np.percentile(probs, 95)),
        "frac_below_0.1": float(np.mean(probs < 0.1)),
        "frac_above_0.5": float(np.mean(probs > 0.5)),
        "frac_above_0.9": float(np.mean(probs > 0.9)),
    }


def main() -> None:
    rng = np.random.default_rng(20260623)
    fakes = make_smooth_fakes(1000, rng)
    classifier_files = {
        "original": "plausibility_classifier.npz",
        "disjoint_split": "plausibility_classifier_holdout.npz",
    }

    results = {
        "provenance": {
            "script": "audit_classifier_fifth_fake_family.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "fake_family": (
                "smooth low-frequency mean-normalized images: sinusoids, "
                "Gaussian blobs, and row-band gradients; not used in classifier training"
            ),
            "n_per_family": 1000,
        },
        "by_classifier": {},
        "load_failures": {},
        "pixel_stats": {},
    }

    for family, x in fakes.items():
        results["pixel_stats"][family] = {
            "mean_pixel_mean": float(np.mean(x.mean(axis=1))),
            "min_pixel": float(np.min(x)),
            "max_pixel": float(np.max(x)),
            "mean_linf_from_0p5": float(np.mean(np.max(np.abs(x - 0.5), axis=1))),
        }

    for clf_name, filename in classifier_files.items():
        try:
            layers = load_classifier(filename)
        except Exception as exc:  # record and keep the experiment moving
            results["load_failures"][clf_name] = {
                "file": filename,
                "error": repr(exc),
            }
            continue
        block = {}
        for family, x in fakes.items():
            block[family] = stats(classify(x, layers))
        results["by_classifier"][clf_name] = block

    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
