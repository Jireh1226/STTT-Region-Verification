#!/usr/bin/env python3
"""Technical characterization of the TinyTaxiNet P4 wrong-sign CTE failure mode,
beyond the headline count. Pure numpy (released .nnet + released dataset); no
solver, no TF.

Wrong-sign (P4) = sign(true_CTE) != sign(pred_CTE) among images with |true_CTE|>2 m.

Reports, for the wrong-sign set:
  - joint distribution in (true_CTE, true_HE) and predicted (pred_CTE) space,
  - split by side (left true_CTE>2 vs right true_CTE<-2) and |true_CTE| band,
  - decision-boundary profile: |pred_CTE| (marginal vs confident wrong sign),
  - image-support / nearest-neighbour profile: L2 to nearest real image and the
    true side of that neighbour; k-NN opposite-side fraction; intra-set clustering
    vs a matched random baseline,
to answer: localized dataset-supported mode, or scattered simulator noise?

Output: p4_wrong_sign_characterization.json (+ printed summary).
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
NNET = ROOT / "VerifyGAN/models/TinyTaxiNet.nnet"
OUT = ROOT / "p4_wrong_sign_characterization.json"


def load_nnet(path):
    lines = path.read_text().splitlines()
    i = 0
    while lines[i].startswith("//"):
        i += 1
    header = [int(x) for x in lines[i].split(",") if x.strip()]
    num_layers = header[0]; i += 1
    sizes = [int(x) for x in lines[i].split(",") if x.strip()]; i += 1
    i += 5
    Ws, bs = [], []
    for layer in range(num_layers):
        rows, cols = sizes[layer + 1], sizes[layer]
        W = np.zeros((rows, cols))
        for r in range(rows):
            W[r] = [float(x) for x in lines[i].strip().rstrip(",").split(",")][:cols]; i += 1
        b = np.zeros(rows)
        for r in range(rows):
            b[r] = float(lines[i].strip().rstrip(",")); i += 1
        Ws.append(W); bs.append(b)
    return Ws, bs


def forward(x, Ws, bs):
    h = x
    for k, (W, b) in enumerate(zip(Ws, bs)):
        h = h @ W.T + b
        if k < len(Ws) - 1:
            h = np.maximum(h, 0.0)
    return h


def q(a, ps=(0, 5, 25, 50, 75, 95, 100)):
    return {f"p{p}": round(float(np.percentile(a, p)), 4) for p in ps} if len(a) else {}


def main():
    with h5py.File(DATA, "r") as f:
        images = f["y_train"][:].reshape(-1, 128).astype(np.float64)
        labels = f["X_train"][:].astype(np.float64)
    Ws, bs = load_nnet(NNET)
    pred = forward(images, Ws, bs)
    tcte, the = labels[:, 0], labels[:, 1]
    pcte, phe = pred[:, 0], pred[:, 1]

    eligible = np.abs(tcte) > 2.0
    wrong = eligible & (np.sign(tcte) != np.sign(pcte))
    widx = np.where(wrong)[0]
    n_elig, n_wrong = int(eligible.sum()), int(wrong.sum())

    res = {
        "provenance": {"script": "characterize_p4_wrong_sign.py",
                       "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                       "method": "released .nnet numpy forward on SK_DownsampledGANFocusAreaData.h5",
                       "definition": "wrong-sign = sign(true_CTE)!=sign(pred_CTE) among |true_CTE|>2 m"},
        "counts": {"n_images": int(len(images)), "n_eligible_abs_cte_gt2": n_elig,
                   "n_wrong_sign": n_wrong, "rate": round(n_wrong / n_elig, 5)},
    }

    # --- side split ---
    left = wrong & (tcte > 2.0)      # true left, predicted right
    right = wrong & (tcte < -2.0)    # true right, predicted left
    res["by_side"] = {
        "true_left_pred_right": {"n": int(left.sum()),
                                 "of_eligible_left": int((eligible & (tcte > 2)).sum())},
        "true_right_pred_left": {"n": int(right.sum()),
                                 "of_eligible_right": int((eligible & (tcte < -2)).sum())},
    }

    # --- magnitude bands ---
    bands = [("2-3", 2, 3), ("3-5", 3, 5), ("5-8", 5, 8), (">8", 8, 1e9)]
    res["by_magnitude_band_abs_true_cte"] = {}
    for name, lo, hi in bands:
        bmask = eligible & (np.abs(tcte) >= lo) & (np.abs(tcte) < hi)
        wmask = wrong & (np.abs(tcte) >= lo) & (np.abs(tcte) < hi)
        res["by_magnitude_band_abs_true_cte"][name] = {
            "eligible": int(bmask.sum()), "wrong": int(wmask.sum()),
            "rate": round(int(wmask.sum()) / max(1, int(bmask.sum())), 4)}

    # --- distribution of the wrong-sign set in (true_CTE, true_HE, pred_CTE) ---
    res["wrong_sign_distribution"] = {
        "true_cte": q(tcte[wrong]), "true_he": q(the[wrong]),
        "pred_cte": q(pcte[wrong]), "pred_he": q(phe[wrong]),
        "abs_true_cte": q(np.abs(tcte[wrong])), "abs_pred_cte": q(np.abs(pcte[wrong])),
    }

    # --- decision-boundary profile ---
    # |pred_cte| small => marginal (just over 0); large => confidently wrong.
    ap = np.abs(pcte[wrong])
    res["boundary_profile"] = {
        "abs_pred_cte_quantiles": q(ap),
        "frac_marginal_abs_pred_cte_lt_0p5": round(float(np.mean(ap < 0.5)), 4),
        "frac_marginal_abs_pred_cte_lt_1p0": round(float(np.mean(ap < 1.0)), 4),
        "frac_confident_abs_pred_cte_gt_2p0": round(float(np.mean(ap > 2.0)), 4),
        "note": "small |pred_CTE| = network places the image just on the wrong side of 0 (near its CTE decision boundary); large = confidently wrong",
    }

    # --- image-support / nearest-neighbour profile (subsample for O(n_wrong * N)) ---
    rng = np.random.default_rng(0)
    sub = widx if n_wrong <= 400 else rng.choice(widx, 400, replace=False)
    nn_true_cte, nn_opposite_frac_k, nn_dist = [], [], []
    K = 10
    for wi in sub:
        d = np.linalg.norm(images - images[wi], axis=1)
        d[wi] = np.inf
        order = np.argsort(d)
        nn = order[0]
        nn_true_cte.append(tcte[nn]); nn_dist.append(d[nn])
        knn = order[:K]
        # opposite side = neighbour true_cte has opposite sign to this image's true_cte
        opp = np.sign(tcte[knn]) != np.sign(tcte[wi])
        nn_opposite_frac_k.append(float(opp.mean()))
    nn_true_cte = np.array(nn_true_cte)
    # baseline: nearest-neighbour distance for a matched random eligible sample
    base = rng.choice(np.where(eligible)[0], min(400, n_elig), replace=False)
    base_dist = []
    for bi in base:
        d = np.linalg.norm(images - images[bi], axis=1); d[bi] = np.inf
        base_dist.append(d.min())
    res["support_profile"] = {
        "subsample_size": int(len(sub)),
        "nearest_real_L2_median": round(float(np.median(nn_dist)), 4),
        "baseline_eligible_nearest_real_L2_median": round(float(np.median(base_dist)), 4),
        "frac_wrong_whose_nearest_neighbour_is_opposite_side": round(float(np.mean(np.sign(nn_true_cte) != np.sign(tcte[sub]))), 4),
        "mean_kNN_opposite_side_fraction_k10": round(float(np.mean(nn_opposite_frac_k)), 4),
        "reading": "high opposite-side kNN fraction => wrong-sign images sit among opposite-side images (a confusable geometric neighbourhood); low => isolated",
    }

    # --- intra-set clustering vs matched random baseline ---
    def mean_pairwise(idxs):
        X = images[idxs]
        # mean pairwise L2 via sample to bound cost
        s = idxs if len(idxs) <= 300 else rng.choice(idxs, 300, replace=False)
        Xs = images[s]
        dm = np.linalg.norm(Xs[:, None, :] - Xs[None, :, :], axis=2)
        iu = np.triu_indices(len(s), 1)
        return float(dm[iu].mean())
    rand_elig = rng.choice(np.where(eligible)[0], min(300, n_elig), replace=False)
    res["clustering"] = {
        "wrong_sign_mean_pairwise_L2": round(mean_pairwise(widx), 4),
        "random_eligible_mean_pairwise_L2": round(mean_pairwise(rand_elig), 4),
        "reading": "smaller wrong-sign pairwise distance than random eligible => the wrong-sign set is more localized than a random eligible subset",
    }

    OUT.write_text(json.dumps(res, indent=2) + "\n")
    # compact print
    print(json.dumps({k: res[k] for k in ("counts", "by_side", "by_magnitude_band_abs_true_cte",
                                          "boundary_profile", "support_profile", "clustering")}, indent=2))
    print("Saved", OUT.name)


if __name__ == "__main__":
    main()
