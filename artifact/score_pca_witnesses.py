"""
Re-extract SAT witnesses from the local-PCA P3 verification (which only saved
summary stats), then score each witness against (a) the trained plausibility
classifier and (b) the nearest real image in the released dataset.

Output: pca_witness_scores.json with per-witness scores. The scores indicate
whether the witnesses look like plausible images or like artifacts of the
linear approximation leaving the manifold.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import time
from datetime import datetime, timezone

import h5py
import numpy as np
from maraboupy import Marabou


NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
CLF_WEIGHTS = "plausibility_classifier.npz"
PCA_RESULTS = "verification_p3_local_pca.json"
OUTPUT_JSON = "pca_witness_scores.json"

NUM_PIXELS = 128
SUM_TARGET = 64.0
PIXEL_MIN, PIXEL_MAX = 0.0, 1.0
P3_CTE_BOUND = 10.0
P3_HE_BOUND = 90.0
D = 8
CTE_BIN_HALF = 0.5
HE_BIN_HALF = 5.0
NUM_IMAGES = 20
P3_CENTERED_THRESHOLD = 1.0


def load_data():
    with h5py.File(DATA_PATH, "r") as f:
        return (f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float64),
                f["X_train"][:].astype(np.float64))


def select_centered_sample(labels):
    cte = labels[:, 0]
    centered_idx = np.where(np.abs(cte) < P3_CENTERED_THRESHOLD)[0]
    return centered_idx[np.linspace(0, len(centered_idx) - 1, NUM_IMAGES, dtype=int)]


def build_local_pca(images, labels, idx, d=D):
    cte = labels[:, 0]
    he = labels[:, 1]
    x0 = images[int(idx)]
    cte_q, he_q = cte[int(idx)], he[int(idx)]
    mask = ((np.abs(cte - cte_q) <= CTE_BIN_HALF) &
            (np.abs(he - he_q) <= HE_BIN_HALF))
    mask[int(idx)] = False
    neighbors = images[mask]
    D_mat = neighbors - x0
    _, S, Vt = np.linalg.svd(D_mat, full_matrices=False)
    V_top = Vt[:d]
    proj = D_mat @ V_top.T
    sigma_proj = proj.std(axis=0)
    return x0, V_top, sigma_proj


SUB_QUERY_MAP = {
    "a": (192, ">=", P3_CTE_BOUND),
    "b": (192, "<=", -P3_CTE_BOUND),
    "c": (193, ">=", P3_HE_BOUND),
    "d": (193, "<=", -P3_HE_BOUND),
}


def feasibility(x0, V_top, alphas):
    half_widths = np.sum(np.abs(V_top.T) * alphas[None, :], axis=1)
    pixel_lo = np.maximum(PIXEL_MIN, x0 - half_widths)
    pixel_hi = np.minimum(PIXEL_MAX, x0 + half_widths)
    return pixel_lo, pixel_hi


def extract_witness(x0, V_top, alphas, out_var, direc, bound, timeout_s=300):
    pixel_lo, pixel_hi = feasibility(x0, V_top, alphas)
    if np.any(pixel_lo > pixel_hi):
        return None, "region_empty_after_clip"
    net = Marabou.read_nnet(NNET_PATH)
    input_vars = [int(v) for v in net.inputVars[0].flatten()]
    for i in range(NUM_PIXELS):
        net.setLowerBound(input_vars[i], float(pixel_lo[i]))
        net.setUpperBound(input_vars[i], float(pixel_hi[i]))
    d = V_top.shape[0]
    z_vars = [int(net.getNewVariable()) for _ in range(d)]
    for j in range(d):
        net.setLowerBound(z_vars[j], float(-alphas[j]))
        net.setUpperBound(z_vars[j], float(alphas[j]))
    for i in range(NUM_PIXELS):
        col = V_top[:, i]
        nz = np.where(np.abs(col) > 1e-12)[0]
        vars_eq = [input_vars[i]] + [z_vars[j] for j in nz]
        coeffs_eq = [1.0] + [float(-col[j]) for j in nz]
        net.addEquality(vars_eq, coeffs_eq, float(x0[i]))
    net.addEquality(list(input_vars), [1.0] * NUM_PIXELS, SUM_TARGET)
    if direc == ">=":
        net.setLowerBound(out_var, float(bound))
    else:
        net.setUpperBound(out_var, float(bound))
    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=timeout_s)
    status, vals, _ = net.solve(verbose=False, options=opts)
    status = status.strip().lower()
    if status != "sat":
        return None, status
    cx = np.array([vals[v] for v in input_vars])
    return cx, "sat"


def load_classifier():
    w = np.load(CLF_WEIGHTS)
    return [(w["c1_W"], w["c1_b"]),
            (w["c2_W"], w["c2_b"]),
            (w["score_W"], w["score_b"])]


def score_clf(x, layers):
    if x.ndim == 1:
        x = x[None, :]
    h = x
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    return float(h.flatten()[0])


def prob_from_logit(logit):
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))


def nearest_real_distance(witness, all_images):
    diffs = all_images - witness
    l2 = np.linalg.norm(diffs, axis=1)
    j = int(np.argmin(l2))
    return float(l2[j]), j


def main():
    print("Loading data, classifier, and PCA results...", flush=True)
    images, labels = load_data()
    clf = load_classifier()
    with open(PCA_RESULTS) as f:
        pca_data = json.load(f)

    # Real-image baseline classifier scores
    real_logits = np.array([score_clf(images[i], clf) for i in range(0, len(images), 50)])
    real_probs = 1.0 / (1.0 + np.exp(-np.clip(real_logits, -30, 30)))
    print(f"Real-image classifier baseline (n={len(real_logits)}): "
          f"logit mean {real_logits.mean():.3f}, prob mean {real_probs.mean():.3f}, "
          f"prob median {np.median(real_probs):.3f}", flush=True)

    sample = select_centered_sample(labels)

    out = {
        "provenance": {
            "script": "score_pca_witnesses.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pca_results_file": PCA_RESULTS,
            "classifier_weights": CLF_WEIGHTS,
        },
        "real_image_baseline": {
            "n_sampled": int(len(real_logits)),
            "logit_mean": round(float(real_logits.mean()), 4),
            "logit_median": round(float(np.median(real_logits)), 4),
            "prob_mean": round(float(real_probs.mean()), 4),
            "prob_median": round(float(np.median(real_probs)), 4),
            "prob_min": round(float(real_probs.min()), 4),
            "prob_max": round(float(real_probs.max()), 4),
        },
        "witnesses": [],
    }

    pca_cache = {}
    sat_cases = []
    for alpha_key, alpha_val in [("alpha_2.0", 2.0), ("alpha_3.0", 3.0)]:
        per_image = pca_data["results_by_alpha"][alpha_key]["per_image"]
        for img_key, img in per_image.items():
            for sq_tag, sq_res in img["sub_queries"].items():
                if sq_res["status"] == "sat":
                    sat_cases.append((img_key, img["dataset_index"], sq_tag, alpha_val))

    print(f"\nFound {len(sat_cases)} SAT cases to extract witnesses for.", flush=True)

    for n, (img_key, dataset_idx, sq_tag, alpha_val) in enumerate(sat_cases):
        if dataset_idx not in pca_cache:
            pca_cache[dataset_idx] = build_local_pca(images, labels, dataset_idx)
        x0, V_top, sigma_proj = pca_cache[dataset_idx]
        alphas = alpha_val * sigma_proj
        out_var, direc, bound = SUB_QUERY_MAP[sq_tag]
        t0 = time.time()
        witness, status = extract_witness(x0, V_top, alphas, out_var, direc, bound)
        elapsed = time.time() - t0
        if witness is None:
            print(f"  [{n+1}/{len(sat_cases)}] {img_key} idx={dataset_idx} sub={sq_tag} "
                  f"alpha={alpha_val} -> {status} ({elapsed:.1f}s)", flush=True)
            out["witnesses"].append({
                "image_key": img_key,
                "dataset_index": int(dataset_idx),
                "sub_query": sq_tag,
                "alpha_sigma": alpha_val,
                "status": status,
            })
            continue
        clf_logit = score_clf(witness, clf)
        clf_prob = prob_from_logit(clf_logit)
        x0_l2 = float(np.linalg.norm(witness - x0))
        x0_linf = float(np.max(np.abs(witness - x0)))
        nn_l2, nn_idx = nearest_real_distance(witness, images)
        nn_logit = score_clf(images[nn_idx], clf)
        nn_prob = prob_from_logit(nn_logit)
        rec = {
            "image_key": img_key,
            "dataset_index": int(dataset_idx),
            "sub_query": sq_tag,
            "alpha_sigma": alpha_val,
            "status": "sat",
            "elapsed_s": round(elapsed, 4),
            "classifier_logit": round(clf_logit, 4),
            "classifier_prob_real": round(clf_prob, 4),
            "l2_from_x0": round(x0_l2, 4),
            "linf_from_x0": round(x0_linf, 4),
            "l2_to_nearest_real": round(nn_l2, 4),
            "nearest_real_index": int(nn_idx),
            "nearest_real_classifier_prob": round(nn_prob, 4),
            "witness_mean": round(float(witness.mean()), 4),
            "witness_min": round(float(witness.min()), 4),
            "witness_max": round(float(witness.max()), 4),
        }
        out["witnesses"].append(rec)
        print(f"  [{n+1}/{len(sat_cases)}] {img_key} idx={dataset_idx} sub={sq_tag} "
              f"alpha={alpha_val}: clf_prob={clf_prob:.3f}, l2_to_nearest={nn_l2:.3f} "
              f"({elapsed:.1f}s)", flush=True)

    # Summary stats over all witnesses
    scored = [w for w in out["witnesses"] if w.get("status") == "sat"]
    if scored:
        probs = np.array([w["classifier_prob_real"] for w in scored])
        nn_dists = np.array([w["l2_to_nearest_real"] for w in scored])
        out["summary"] = {
            "n_witnesses_scored": int(len(scored)),
            "classifier_prob_real": {
                "min": round(float(probs.min()), 4),
                "max": round(float(probs.max()), 4),
                "mean": round(float(probs.mean()), 4),
                "median": round(float(np.median(probs)), 4),
            },
            "l2_to_nearest_real_image": {
                "min": round(float(nn_dists.min()), 4),
                "max": round(float(nn_dists.max()), 4),
                "mean": round(float(nn_dists.mean()), 4),
                "median": round(float(np.median(nn_dists)), 4),
            },
            "by_alpha": {},
        }
        for av in (2.0, 3.0):
            sub = [w for w in scored if w["alpha_sigma"] == av]
            if sub:
                p = np.array([w["classifier_prob_real"] for w in sub])
                n = np.array([w["l2_to_nearest_real"] for w in sub])
                out["summary"]["by_alpha"][f"alpha_{av}"] = {
                    "n": len(sub),
                    "clf_prob_mean": round(float(p.mean()), 4),
                    "clf_prob_median": round(float(np.median(p)), 4),
                    "nn_l2_mean": round(float(n.mean()), 4),
                    "nn_l2_median": round(float(np.median(n)), 4),
                }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}", flush=True)
    print("\n=== SUMMARY ===", flush=True)
    if "summary" in out:
        s = out["summary"]
        print(f"Witnesses scored: {s['n_witnesses_scored']}", flush=True)
        print(f"Classifier prob (real): mean {s['classifier_prob_real']['mean']:.3f}, "
              f"median {s['classifier_prob_real']['median']:.3f}, "
              f"range [{s['classifier_prob_real']['min']:.3f}, "
              f"{s['classifier_prob_real']['max']:.3f}]", flush=True)
        print(f"L2 to nearest real image: mean {s['l2_to_nearest_real_image']['mean']:.3f}, "
              f"median {s['l2_to_nearest_real_image']['median']:.3f}, "
              f"range [{s['l2_to_nearest_real_image']['min']:.3f}, "
              f"{s['l2_to_nearest_real_image']['max']:.3f}]", flush=True)
        b = out["real_image_baseline"]
        print(f"Real-image baseline: clf_prob median {b['prob_median']:.3f}, "
              f"mean {b['prob_mean']:.3f}", flush=True)


if __name__ == "__main__":
    main()
