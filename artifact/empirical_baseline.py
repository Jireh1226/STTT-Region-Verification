"""
Empirical baseline for TinyTaxiNet ground-truth audit.

Loads the released TinyTaxiNet network and SK_DownsampledGANFocusAreaData.h5
dataset, runs a forward pass on every image, and reports the empirical CTE
and HE error distributions. The point is to give the verification report an
end-to-end performance reference so verification rates can be compared against
nominal model accuracy on the same data the verification queries use.

Outputs:
  - empirical_baseline.json  (raw numeric summary)
  - stdout                   (human-readable summary table)

This script does NOT call Marabou's solver; it only uses Marabou.read_nnet
to evaluate the network, so it is fast (~5 s for 10000 images).
"""
import json
import subprocess
import time
from datetime import datetime, timezone

import h5py
import numpy as np
from maraboupy import Marabou

NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
OUTPUT_JSON = "empirical_baseline.json"
NUM_PIXELS = 128


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        imgs = f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float64)
        labels = f["X_train"][:].astype(np.float64)
    return imgs, labels


def forward_all(images):
    net = Marabou.read_nnet(NNET_PATH)
    n = len(images)
    preds = np.zeros((n, 2), dtype=np.float64)
    for i in range(n):
        out = np.array(net.evaluate([images[i]])).flatten()
        preds[i, 0] = out[0]
        preds[i, 1] = out[1]
    return preds


def distribution_summary(errs, n_bins=10):
    arr = np.asarray(errs, dtype=np.float64)
    abs_arr = np.abs(arr)
    hist_counts, hist_edges = np.histogram(abs_arr, bins=n_bins,
                                           range=(0.0, float(abs_arr.max())))
    return {
        "count": int(arr.size),
        "mean_abs_error": float(abs_arr.mean()),
        "median_abs_error": float(np.median(abs_arr)),
        "std_abs_error": float(abs_arr.std()),
        "max_abs_error": float(abs_arr.max()),
        "min_abs_error": float(abs_arr.min()),
        "rmse": float(np.sqrt((arr ** 2).mean())),
        "quantiles_abs": {
            "p50": float(np.quantile(abs_arr, 0.50)),
            "p75": float(np.quantile(abs_arr, 0.75)),
            "p90": float(np.quantile(abs_arr, 0.90)),
            "p95": float(np.quantile(abs_arr, 0.95)),
            "p99": float(np.quantile(abs_arr, 0.99)),
        },
        "histogram": {
            "bin_edges": [float(x) for x in hist_edges],
            "counts": [int(x) for x in hist_counts],
        },
    }


def tail_rates(abs_errs, thresholds):
    n = len(abs_errs)
    out = {}
    for t in thresholds:
        cnt = int(np.sum(abs_errs > t))
        out[str(t)] = {"count": cnt, "rate": cnt / n}
    return out


def wrong_sign_rate(true_vals, pred_vals, dead_eps=0.0):
    """Wrong-sign = sign(true) != sign(pred), only counted where |true| > dead_eps
    so we don't penalize tiny CTE near zero whose 'true' sign is ambiguous."""
    mask = np.abs(true_vals) > dead_eps
    n = int(mask.sum())
    if n == 0:
        return {"count_eligible": 0, "wrong": 0, "rate": 0.0}
    t = true_vals[mask]
    p = pred_vals[mask]
    wrong = int(np.sum(np.sign(t) != np.sign(p)))
    return {"count_eligible": n, "wrong": wrong, "rate": wrong / n,
            "dead_eps": dead_eps}


def main():
    print("Loading data ...", flush=True)
    images, labels = load_data()
    cte_true = labels[:, 0]
    he_true = labels[:, 1]
    n = len(images)

    centered_mask = np.abs(cte_true) < 1.0
    offset_mask = ~centered_mask
    left_mask = cte_true > 2.0
    right_mask = cte_true < -2.0
    off_center_mask = np.abs(cte_true) > 5.0
    print(f"Loaded {n} images.  centered={centered_mask.sum()}, "
          f"offset={offset_mask.sum()}, left={left_mask.sum()}, "
          f"right={right_mask.sum()}, off_center={off_center_mask.sum()}",
          flush=True)

    print("Running forward passes via Marabou.read_nnet ...", flush=True)
    t0 = time.time()
    preds = forward_all(images)
    print(f"  done in {time.time() - t0:.1f}s", flush=True)

    cte_pred = preds[:, 0]
    he_pred = preds[:, 1]
    cte_err = cte_pred - cte_true
    he_err = he_pred - he_true

    cte_summary_all = distribution_summary(cte_err)
    he_summary_all = distribution_summary(he_err)

    by_subset = {}
    for name, mask in [("all", np.ones(n, dtype=bool)),
                       ("centered_|cte|<1", centered_mask),
                       ("offset_|cte|>=1", offset_mask),
                       ("left_cte>2", left_mask),
                       ("right_cte<-2", right_mask),
                       ("off_center_|cte|>5", off_center_mask)]:
        if mask.sum() == 0:
            continue
        sub_cte = cte_err[mask]
        sub_he = he_err[mask]
        sub_cte_true = cte_true[mask]
        sub_cte_pred = cte_pred[mask]
        sub_he_true = he_true[mask]
        sub_he_pred = he_pred[mask]
        by_subset[name] = {
            "n": int(mask.sum()),
            "cte_error": distribution_summary(sub_cte),
            "he_error": distribution_summary(sub_he),
            "cte_tail_rates": tail_rates(np.abs(sub_cte),
                                         [0.5, 1.0, 2.0, 5.0, 10.0]),
            "he_tail_rates": tail_rates(np.abs(sub_he),
                                        [1.0, 5.0, 10.0, 30.0, 90.0]),
            "cte_wrong_sign": wrong_sign_rate(sub_cte_true, sub_cte_pred,
                                              dead_eps=0.5),
            "he_wrong_sign": wrong_sign_rate(sub_he_true, sub_he_pred,
                                             dead_eps=1.0),
            "cte_normalized_rmse": float(np.sqrt((sub_cte ** 2).mean())
                                         / 11.0),  # divide by half-range
            "he_normalized_rmse": float(np.sqrt((sub_he ** 2).mean())
                                        / 30.0),
        }

    # CTE-vs-HE fragility comparison (normalized by their full label range)
    cte_range = 22.0  # ~ [-11, +11]
    he_range = 60.0   # ~ [-30, +30]
    cte_norm_err = np.abs(cte_err) / (cte_range / 2)
    he_norm_err = np.abs(he_err) / (he_range / 2)

    fragility = {
        "cte_label_half_range_m": cte_range / 2,
        "he_label_half_range_deg": he_range / 2,
        "cte_mean_normalized_err": float(cte_norm_err.mean()),
        "he_mean_normalized_err": float(he_norm_err.mean()),
        "cte_median_normalized_err": float(np.median(cte_norm_err)),
        "he_median_normalized_err": float(np.median(he_norm_err)),
        "cte_more_fragile_than_he": bool(cte_norm_err.mean()
                                         > he_norm_err.mean()),
        "ratio_cte_to_he_mean": float(cte_norm_err.mean()
                                      / he_norm_err.mean()),
    }

    out = {
        "provenance": {
            "script": "empirical_baseline.py",
            "git_commit": git_commit(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "nnet_path": NNET_PATH,
            "data_path": DATA_PATH,
            "n_images": int(n),
        },
        "global": {
            "cte_error": cte_summary_all,
            "he_error": he_summary_all,
            "cte_tail_rates": tail_rates(np.abs(cte_err),
                                         [0.5, 1.0, 2.0, 5.0, 10.0]),
            "he_tail_rates": tail_rates(np.abs(he_err),
                                        [1.0, 5.0, 10.0, 30.0, 90.0]),
            "cte_wrong_sign_dead_eps_0.5m":
                wrong_sign_rate(cte_true, cte_pred, dead_eps=0.5),
            "cte_wrong_sign_dead_eps_2.0m":
                wrong_sign_rate(cte_true, cte_pred, dead_eps=2.0),
            "he_wrong_sign_dead_eps_1.0deg":
                wrong_sign_rate(he_true, he_pred, dead_eps=1.0),
            "he_wrong_sign_dead_eps_5.0deg":
                wrong_sign_rate(he_true, he_pred, dead_eps=5.0),
        },
        "by_subset": by_subset,
        "cte_vs_he_fragility": fragility,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {OUTPUT_JSON}", flush=True)

    # Concise summary print
    print("\n=== Empirical baseline summary ===")
    print(f"N total = {n}")
    print(f"CTE: mean|err|={cte_summary_all['mean_abs_error']:.3f} m, "
          f"p95={cte_summary_all['quantiles_abs']['p95']:.3f} m, "
          f"max={cte_summary_all['max_abs_error']:.3f} m")
    print(f"HE:  mean|err|={he_summary_all['mean_abs_error']:.3f} deg, "
          f"p95={he_summary_all['quantiles_abs']['p95']:.3f} deg, "
          f"max={he_summary_all['max_abs_error']:.3f} deg")
    print(f"CTE wrong-sign rate (|true|>0.5m): "
          f"{out['global']['cte_wrong_sign_dead_eps_0.5m']['rate']*100:.2f}%")
    print(f"CTE wrong-sign rate (|true|>2.0m): "
          f"{out['global']['cte_wrong_sign_dead_eps_2.0m']['rate']*100:.2f}%")
    print(f"HE wrong-sign rate (|true|>1deg): "
          f"{out['global']['he_wrong_sign_dead_eps_1.0deg']['rate']*100:.2f}%")
    print(f"CTE more fragile than HE (normalized): "
          f"{fragility['cte_more_fragile_than_he']} "
          f"(ratio {fragility['ratio_cte_to_he_mean']:.2f}x)")


if __name__ == "__main__":
    main()
