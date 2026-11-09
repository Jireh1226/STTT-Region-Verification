#!/usr/bin/env python3
"""Scale-calibration sanity check for the matched random-subspace control.

Preempts the "your random control is not matched to PCA" objection by showing that,
at the same per-direction sigma scale and alpha, the PCA and random subspace regions
share the invariants a matched control must share, and differ only in orientation:

  - z-box volume  = prod_j (2*alpha*sigma_j)   -- identical (same sigma, alpha, d)
  - L2 reach      = ||V^T z|| at the box corner = sqrt(sum_j (alpha*sigma_j)^2)
                    -- identical for any orthonormal V (isometry)
  - orientation-dependent, reported for both: per-pixel half-width distribution,
    fraction of pixels whose raw envelope leaves [0,1] (valid-pixel constraint
    activity), and effective post-clip per-pixel box volume.

Pure numpy on the released datasets. Output: scale_calibration_check.json.
"""
import json
from pathlib import Path
import numpy as np
import h5py

ROOT = Path(__file__).resolve().parent
D = 8
ALPHA = 1.0
SEEDS = [0, 1, 2]


def random_zero_mean_basis(seed, idx, d, n):
    rng = np.random.default_rng(int(seed) * 1_000_003 + int(idx))
    cols = []
    for _ in range(d):
        v = rng.normal(size=n); v = v - v.mean()
        for q in cols:
            v = v - np.dot(v, q) * q
        v = v / np.linalg.norm(v)
        cols.append(v)
    return np.vstack(cols)


def metrics(x0, V, sigma, alpha):
    """Region invariants and orientation-dependent pixel-space quantities."""
    box = alpha * sigma                       # half-widths in z
    logvol = float(np.sum(np.log(2 * box)))   # log z-box volume
    l2_reach = float(np.sqrt(np.sum(box ** 2)))
    hw = np.abs(V).T @ box                     # per-pixel half-width
    lo = x0 - hw; hi = x0 + hw
    clipped = ((lo < -1e-9) | (hi > 1 + 1e-9))
    # effective post-clip per-pixel box (log) volume
    eff_lo = np.maximum(0.0, lo); eff_hi = np.minimum(1.0, hi)
    w = np.clip(eff_hi - eff_lo, 1e-12, None)
    eff_logvol = float(np.sum(np.log(w)))
    return {"z_logvol": logvol, "l2_reach": l2_reach,
            "pixel_halfwidth_median": float(np.median(hw)),
            "pixel_halfwidth_max": float(hw.max()),
            "frac_pixels_clipped": float(clipped.mean()),
            "eff_postclip_pixelbox_logvol": eff_logvol}


def agg(rows):
    keys = rows[0].keys()
    return {k: round(float(np.mean([r[k] for r in rows])), 5) for k in keys}


def taxinet_anchors():
    with h5py.File(ROOT / "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5", "r") as f:
        images = f["y_train"][:].reshape(-1, 128).astype(np.float64)
        labels = f["X_train"][:].astype(np.float64)
    cte, he = labels[:, 0], labels[:, 1]
    cen = np.where(np.abs(cte) < 1.0)[0]
    sample = cen[np.linspace(0, len(cen) - 1, 40, dtype=int)]  # 40 anchors is plenty for calibration
    out = []
    for idx in sample:
        x0 = images[int(idx)]
        mask = (np.abs(cte - cte[idx]) <= 0.5) & (np.abs(he - he[idx]) <= 5.0); mask[idx] = False
        nb = images[mask]
        if len(nb) < D + 1:
            continue
        Dm = nb - x0
        _, S, Vt = np.linalg.svd(Dm, full_matrices=False)
        V = Vt[:D]; sig = (Dm @ V.T).std(0)
        out.append((int(idx), x0, V, sig, 128))
    return out


def lard_anchors():
    d = np.load(ROOT / "external_lard_dataset.npz", allow_pickle=True)
    import json as _j
    tgt = _j.load(open(ROOT / "external_lard_target.json"))
    q1, q2 = tgt["thresholds_train"]
    Xtr = d["Xtr"]; Xte = d["Xte"]
    v = np.asarray(d[tgt["chosen_target"] + "_tr"], dtype=np.float64)
    ytr = np.ones(len(v), np.int64); ytr[v <= q1] = 0; ytr[v >= q2] = 2
    man = _j.load(open(ROOT / "external_lard_manifest.json"))
    out = []
    for a in man["anchors"]:
        i, c = a["anchor_test_idx"], a["true_class"]
        x0 = Xte[i].astype(np.float64)
        same = np.where(ytr == c)[0]
        nbr = same[np.argsort(np.linalg.norm(Xtr[same] - x0, axis=1))[:200]]
        Dm = Xtr[nbr] - x0
        _, S, Vt = np.linalg.svd(Dm, full_matrices=False)
        V = Vt[:D]; sig = (S[:D] / np.sqrt(len(nbr) - 1)).astype(np.float64)
        out.append((i, x0, V, sig, Xtr.shape[1]))
    return out


def run(name, anchors):
    pca_rows, rnd_rows = [], []
    max_z_gap, max_l2_gap = 0.0, 0.0
    for idx, x0, Vp, sig, n in anchors:
        mp = metrics(x0, Vp, sig, ALPHA); pca_rows.append(mp)
        for seed in SEEDS:
            Vr = random_zero_mean_basis(seed, idx, D, n)
            mr = metrics(x0, Vr, sig, ALPHA); rnd_rows.append(mr)
            max_z_gap = max(max_z_gap, abs(mp["z_logvol"] - mr["z_logvol"]))
            max_l2_gap = max(max_l2_gap, abs(mp["l2_reach"] - mr["l2_reach"]))
    return {"n_anchors": len(anchors), "d": D, "alpha": ALPHA,
            "pca_mean": agg(pca_rows), "random_mean": agg(rnd_rows),
            "max_abs_z_logvol_gap_pca_vs_random": round(max_z_gap, 9),
            "max_abs_l2_reach_gap_pca_vs_random": round(max_l2_gap, 9),
            "matched_invariants": ("z-box volume and L2 reach are identical for PCA and random "
                                   "(same sigma, alpha, d; orthonormal isometry); regions differ only in orientation.")}


def main():
    out = {"experiment": "scale_calibration_check", "note":
           "Confirms the random 8D control is matched to local PCA in z-box volume and L2 reach; "
           "any difference is orientation, reflected in per-pixel half-widths and valid-pixel clipping activity."}
    out["TinyTaxiNet_P3"] = run("taxinet", taxinet_anchors())
    try:
        out["LARD"] = run("lard", lard_anchors())
    except Exception as e:
        out["LARD"] = {"error": str(e)}
    (ROOT / "scale_calibration_check.json").write_text(json.dumps(out, indent=2) + "\n")
    for k in ("TinyTaxiNet_P3", "LARD"):
        b = out[k]
        if "error" in b:
            print(k, "ERROR", b["error"]); continue
        print(f"=== {k} (d={D}, alpha={ALPHA}sigma, n={b['n_anchors']}) ===")
        print(f"  z-box logvol gap PCA-vs-random (max): {b['max_abs_z_logvol_gap_pca_vs_random']:.2e}  (0 => identical volume)")
        print(f"  L2-reach gap PCA-vs-random (max):     {b['max_abs_l2_reach_gap_pca_vs_random']:.2e}  (0 => identical reach)")
        print(f"  per-pixel halfwidth median  PCA={b['pca_mean']['pixel_halfwidth_median']:.4f}  random={b['random_mean']['pixel_halfwidth_median']:.4f}")
        print(f"  frac pixels clipped         PCA={b['pca_mean']['frac_pixels_clipped']:.4f}  random={b['random_mean']['frac_pixels_clipped']:.4f}")
        print(f"  eff post-clip pixel logvol  PCA={b['pca_mean']['eff_postclip_pixelbox_logvol']:.2f}  random={b['random_mean']['eff_postclip_pixelbox_logvol']:.2f}")
    print("Saved scale_calibration_check.json")


if __name__ == "__main__":
    main()
