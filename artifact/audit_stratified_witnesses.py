"""
Stratified witness audit across all five region families:
  raw pixel box, box+mean, generic-math, runway-calibrated math, AE.

For every SAT witness across the released JSON files, score:
  - plausibility-classifier probability
  - L2 distance to nearest real centered image
  - in-range fraction (#pixels in [0,1])
  - global TV
  - Fourier high-frequency L1
  - photometric residual L_inf (least-squares fit)
  - per-pair Lipschitz violations vs p99 calibration
  - |CTE| and |HE| output magnitudes

Then aggregate by region family and write
  stratified_witness_audit.json

The point: make a single compact comparison that lets readers see whether
runway-calibrated regions actually produce more real-looking witnesses than
generic math, and whether AE witnesses are closer to real than both.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import re
from datetime import datetime, timezone
from collections import defaultdict

import h5py
import numpy as np


DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
CLF_WEIGHTS = "plausibility_classifier.npz"
NUM_PIXELS = 128


# ---------- region-family routing ----------
def classify_region(json_path, label):
    """Map (file, dot-walk label) to one of:
       raw_pixel_box, box_plus_mean, generic_math, runway_calibrated_math, ae_region
       Returns None if the witness isn't in scope for this audit.
    """
    f = json_path.lower()
    lab = label.lower()

    if "ae_latent_quantile" in f or "ae_plus_math" in f or "autoencoder" in f or "plausibility_constrained" in f:
        return "ae_region"
    if "per_row_mean" in f or "object_specific_math" in f:
        return "runway_calibrated_math"
    if "tv_bound" in f or "fourier_bound" in f or "photometric_bound" in f or "math_combined" in f or \
       "p4_right_strong_math" in f or "math_regions_p3" in f or "p3_tight_tv" in f or \
       "p1_patterned_math" in f or "math_exhaustive" in f:
        return "generic_math"
    if "mean_constrained" in f or "p3_mean_constrained" in f:
        return "box_plus_mean"
    if "results_v2" in f or "results_p3_p4" in f or "results_p5_p6" in f or \
       "results.json" in f or "p3_label_box" in f or "p3_local_pca" in f or \
       "full_sample" in f or "p2_comparison" in f or "p3_certified_radius" in f or \
       "strong_wrong_sign_sweep" in f:
        return "raw_pixel_box"
    return None


# ---------- shared scorers ----------
def load_classifier():
    w = np.load(CLF_WEIGHTS)
    return [(w["c1_W"], w["c1_b"]),
            (w["c2_W"], w["c2_b"]),
            (w["score_W"], w["score_b"])]


def score_clf(x, layers):
    h = x[None, :] if x.ndim == 1 else x
    for i, (W, b) in enumerate(layers):
        h = h @ W + b
        if i < len(layers) - 1:
            h = np.maximum(h, 0)
    return float(h.flatten()[0])


def prob_from_logit(logit):
    return 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))


def tv(x):
    img = x.reshape(8, 16)
    return float(np.abs(np.diff(img, axis=1)).sum() + np.abs(np.diff(img, axis=0)).sum())


def fourier_hf_l1(x):
    img = x.reshape(8, 16)
    C = np.fft.fft2(img)
    K = np.arange(8)[:, None]
    L = np.arange(16)[None, :]
    mask = (K >= 4) | (L >= 8)
    return float((np.abs(C.real) + np.abs(C.imag))[mask].sum())


def photometric_residual_linf(x):
    """min over (alpha, beta) of || x - alpha*b1 - beta*b2 ||_inf, by LP.
       b1 uniform, b2 horizontal gradient (mean-zero, normalized)."""
    b1 = np.ones(NUM_PIXELS) / np.sqrt(NUM_PIXELS)
    h = np.tile(np.arange(16) - 7.5, 8).astype(float)
    h -= h.mean()
    b2 = h / np.linalg.norm(h)
    alpha = float(x @ b1)
    beta = float(x @ b2)
    r = x - alpha * b1 - beta * b2
    return float(np.abs(r).max())


def per_pair_violations(x, taus_p99):
    """Number of adjacent pairs whose |Δ| exceeds the p99 calibration tau."""
    img = x.reshape(8, 16)
    diffs = []
    for i in range(8):
        for j in range(16):
            if j + 1 < 16:
                diffs.append(abs(img[i, j] - img[i, j+1]))
            if i + 1 < 8:
                diffs.append(abs(img[i, j] - img[i+1, j]))
    diffs = np.array(diffs)
    return int((diffs > taus_p99).sum()), int(len(diffs))


# ---------- witness-extraction walker ----------
def walk_for_witnesses(obj, prefix=""):
    """Yield (label, witness_vec_np) for every SAT witness with a full 128-vector."""
    out = []
    if isinstance(obj, dict):
        if (obj.get('status') == 'sat'
                and 'witness' in obj
                and isinstance(obj['witness'], list)
                and len(obj['witness']) == NUM_PIXELS):
            out.append((prefix.strip(' .'), np.asarray(obj['witness'], dtype=np.float64)))
        else:
            for k, v in obj.items():
                out.extend(walk_for_witnesses(v, prefix + "." + str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(walk_for_witnesses(v, prefix + f"[{i}]"))
    return out


# ---------- main ----------
def main():
    print("Loading classifier, dataset...", flush=True)
    clf = load_classifier()
    with h5py.File(DATA_PATH, "r") as f:
        all_images = f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float64)
        labels = f["X_train"][:]
    centered = all_images[np.abs(labels[:, 0]) < 1.0]
    print(f"  N_centered = {len(centered)}", flush=True)

    # Calibrate per-pair p99 taus from centered subset (same family as paper)
    pairs = []
    for i in range(8):
        for j in range(16):
            if j + 1 < 16:
                pairs.append((i * 16 + j, i * 16 + j + 1))
            if i + 1 < 8:
                pairs.append((i * 16 + j, (i + 1) * 16 + j))
    taus_p99 = np.array([
        float(np.percentile(np.abs(centered[:, a] - centered[:, b]), 99))
        for a, b in pairs
    ])
    print(f"  per-pair p99: n_pairs={len(pairs)}, median={np.median(taus_p99):.3f}", flush=True)

    # Real-centered baseline (sampled to keep cost down)
    real_sample = centered[::20]
    real_records = []
    for x in real_sample:
        logit = score_clf(x, clf)
        real_records.append({
            "classifier_prob": prob_from_logit(logit),
            "tv": tv(x),
            "fourier_l1": fourier_hf_l1(x),
            "photo_residual": photometric_residual_linf(x),
            "in_range_frac": float(((x >= 0.0) & (x <= 1.0)).mean()),
            "per_pair_violations_frac": per_pair_violations(x, taus_p99)[0] / len(pairs),
        })
    real_df = {k: np.array([r[k] for r in real_records]) for k in real_records[0]}
    print(f"  real-centered baseline n={len(real_sample)}, "
          f"prob median={np.median(real_df['classifier_prob']):.3f}", flush=True)

    # Scan every verification JSON
    json_files = sorted([f for f in os.listdir(".") if f.startswith("verification_") and f.endswith(".json")])
    by_family = defaultdict(list)

    # Stratified rerun witnesses live in stratified_witnesses_rerun.json with
    # explicit family names in the JSON structure (by_family.<fam>.queries.<q>).
    rerun_path = "stratified_witnesses_rerun.json"
    if os.path.exists(rerun_path):
        d = json.load(open(rerun_path))
        for fam_name, fam_data in d.get("by_family", {}).items():
            # Normalize the explicit family names back into the audit's family taxonomy.
            if fam_name.startswith("generic_math"):
                fam_tag = "generic_math"
            elif fam_name.startswith("runway_calibrated"):
                fam_tag = "runway_calibrated_math"
            elif fam_name == "raw_pixel_box":
                fam_tag = "raw_pixel_box"
            elif fam_name == "box_plus_mean":
                fam_tag = "box_plus_mean"
            else:
                fam_tag = fam_name
            for q_name, q in fam_data.get("queries", {}).items():
                if isinstance(q, dict) and q.get("status", "").startswith("sat") and \
                        "witness" in q and isinstance(q["witness"], list) and len(q["witness"]) == NUM_PIXELS:
                    w = np.asarray(q["witness"], dtype=np.float64)
                    logit = score_clf(w, clf)
                    l2 = float(np.linalg.norm(centered - w, axis=1).min())
                    nv, np_ = per_pair_violations(w, taus_p99)
                    rec = {
                        "source_file": f"{rerun_path}#{fam_name}",
                        "label": q_name,
                        "family": fam_tag,
                        "classifier_prob": prob_from_logit(logit),
                        "classifier_logit": logit,
                        "l2_to_nearest_real": l2,
                        "tv": tv(w),
                        "fourier_l1": fourier_hf_l1(w),
                        "photo_residual": photometric_residual_linf(w),
                        "in_range_frac": float(((w >= 0.0) & (w <= 1.0)).mean()),
                        "per_pair_violations_frac": nv / np_,
                        "witness_mean": float(w.mean()),
                        "output_CTE": q.get("CTE"),
                        "output_HE": q.get("HE"),
                    }
                    by_family[fam_tag].append(rec)

    for jf in json_files:
        try:
            d = json.load(open(jf))
        except Exception as e:
            print(f"  [skip] {jf}: {e}", flush=True)
            continue
        witnesses = walk_for_witnesses(d)
        if not witnesses:
            continue
        for lab, w in witnesses:
            fam = classify_region(jf, lab)
            if fam is None:
                continue
            logit = score_clf(w, clf)
            l2 = float(np.linalg.norm(centered - w, axis=1).min())
            nv, np_ = per_pair_violations(w, taus_p99)
            rec = {
                "source_file": jf,
                "label": lab[:120],
                "family": fam,
                "classifier_prob": prob_from_logit(logit),
                "classifier_logit": logit,
                "l2_to_nearest_real": l2,
                "tv": tv(w),
                "fourier_l1": fourier_hf_l1(w),
                "photo_residual": photometric_residual_linf(w),
                "in_range_frac": float(((w >= 0.0) & (w <= 1.0)).mean()),
                "per_pair_violations_frac": nv / np_,
                "witness_mean": float(w.mean()),
            }
            by_family[fam].append(rec)

    # Aggregate
    aggregate = {}
    summary_keys = ["classifier_prob", "l2_to_nearest_real", "tv",
                    "fourier_l1", "photo_residual", "in_range_frac",
                    "per_pair_violations_frac"]
    for fam, recs in by_family.items():
        if not recs:
            continue
        arr = {k: np.array([r[k] for r in recs]) for k in summary_keys}
        aggregate[fam] = {
            "n_witnesses": len(recs),
            "classifier_prob_above_0.5": int((arr["classifier_prob"] > 0.5).sum()),
            "classifier_prob_below_0.1": int((arr["classifier_prob"] < 0.1).sum()),
            **{k: {
                "median": round(float(np.median(arr[k])), 4),
                "mean": round(float(arr[k].mean()), 4),
                "min": round(float(arr[k].min()), 4),
                "max": round(float(arr[k].max()), 4),
            } for k in summary_keys},
        }

    out = {
        "provenance": {
            "script": "audit_stratified_witnesses.py",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "classifier_weights": CLF_WEIGHTS,
            "n_pairs": len(pairs),
            "per_pair_p99_median_calibration": float(np.median(taus_p99)),
        },
        "real_centered_baseline": {
            "n_sampled": int(len(real_sample)),
            **{k: {
                "median": round(float(np.median(real_df[k])), 4),
                "mean": round(float(real_df[k].mean()), 4),
            } for k in real_df},
        },
        "by_family": aggregate,
        "witnesses": {fam: recs for fam, recs in by_family.items()},
    }

    with open("stratified_witness_audit.json", "w") as f:
        json.dump(out, f, indent=2, default=float)

    # Console summary
    print("\n=== STRATIFIED AUDIT SUMMARY ===\n", flush=True)
    header = f"{'family':28s} {'n':>4s} {'cls med':>8s} {'cls mean':>9s} {'>.5':>5s} {'<.1':>5s} {'L2 med':>7s} {'TV med':>7s} {'in-rng':>7s} {'ppviol':>7s}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    fams_order = ["raw_pixel_box", "box_plus_mean", "generic_math",
                  "runway_calibrated_math", "ae_region"]
    for fam in fams_order:
        if fam not in aggregate:
            continue
        a = aggregate[fam]
        print(f"{fam:28s} {a['n_witnesses']:4d} "
              f"{a['classifier_prob']['median']:8.3f} "
              f"{a['classifier_prob']['mean']:9.3f} "
              f"{a['classifier_prob_above_0.5']:5d} "
              f"{a['classifier_prob_below_0.1']:5d} "
              f"{a['l2_to_nearest_real']['median']:7.3f} "
              f"{a['tv']['median']:7.3f} "
              f"{a['in_range_frac']['median']:7.3f} "
              f"{a['per_pair_violations_frac']['median']:7.3f}",
              flush=True)
    rb = out["real_centered_baseline"]
    print("-" * len(header), flush=True)
    print(f"{'real_centered_baseline':28s} {rb['n_sampled']:4d} "
          f"{rb['classifier_prob']['median']:8.3f} "
          f"{rb['classifier_prob']['mean']:9.3f} "
          f"{'-':>5s} {'-':>5s} "
          f"{'-':>7s} "
          f"{rb['tv']['median']:7.3f} "
          f"{rb['in_range_frac']['median']:7.3f} "
          f"{rb['per_pair_violations_frac']['median']:7.3f}",
          flush=True)

    print("\nSaved stratified_witness_audit.json", flush=True)


if __name__ == "__main__":
    main()
