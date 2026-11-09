#!/usr/bin/env python3
"""Coverage-vs-verification frontier for TinyTaxiNet P3 subspace regions.

x-axis: verification rate (from the dimension sweep, local PCA vs matched random).
y-axis: same-state variance captured by the d-subspace, a membership/support metric
        defined the same way for both families. PCA captures the top-d data variance;
        a random d-subspace captures ~d/n. Parameterized by d in {2,4,8,16,32,64}.

Shows that local PCA holds high coverage while its verification rate collapses with d
(it finds violations), whereas the random subspace keeps ~100% verification with near
-zero data coverage: high verification rate does not imply a data-faithful region.

Outputs: p3_coverage_verification_frontier_figure.png,
         p3_coverage_verification_frontier_data.json
"""
import json
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DIMS = [2, 4, 8, 16, 32, 64]


def variance_capture():
    with h5py.File(ROOT / "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5", "r") as f:
        im = f["y_train"][:].reshape(-1, 128).astype(np.float64)
        la = f["X_train"][:].astype(np.float64)
    cte, he = la[:, 0], la[:, 1]
    cen = np.where(np.abs(cte) < 1.0)[0]
    sample = cen[np.linspace(0, len(cen) - 1, 60, dtype=int)]
    rng = np.random.default_rng(0)
    pca = {d: [] for d in DIMS}; rnd = {d: [] for d in DIMS}
    for idx in sample:
        x0 = im[idx]
        m = (np.abs(cte - cte[idx]) <= 0.5) & (np.abs(he - he[idx]) <= 5.0); m[idx] = False
        nb = im[m] - x0
        if len(nb) < 65:          # same >= d_max+1 neighbour requirement as the sweep
            continue
        _, S, Vt = np.linalg.svd(nb, full_matrices=False)
        tot = (nb ** 2).sum()
        for d in DIMS:
            pca[d].append((S[:d] ** 2).sum() / tot)
            cols = []
            for _ in range(d):
                v = rng.normal(size=128); v -= v.mean()
                for q in cols:
                    v -= v @ q * q
                cols.append(v / np.linalg.norm(v))
            Vr = np.vstack(cols)
            rnd[d].append(((nb @ Vr.T) ** 2).sum() / tot)
    return {d: float(np.mean(pca[d])) for d in DIMS}, {d: float(np.mean(rnd[d])) for d in DIMS}


def main():
    pca_cov, rnd_cov = variance_capture()
    sweep = json.load(open(ROOT / "p3_dimension_sweep_results.json"))["tallies"]
    pca_ver = {d: sweep["pca"][str(d)]["verified_pct"] for d in DIMS}
    rnd_ver = {d: sweep["random"][str(d)]["verified_pct"] for d in DIMS}

    data = {"dims": DIMS,
            "pca": {str(d): {"verified_pct": pca_ver[d], "heldout_variance_capture": round(pca_cov[d], 4)} for d in DIMS},
            "random": {str(d): {"verified_pct": rnd_ver[d], "heldout_variance_capture": round(rnd_cov[d], 4)} for d in DIMS},
            "linf_reference": {"verified_pct": 20.8, "coverage": "n/a (full-dim pixel box; no low-dim membership)"}}
    (ROOT / "p3_coverage_verification_frontier_data.json").write_text(json.dumps(data, indent=2) + "\n")

    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5})
    fig, ax = plt.subplots(figsize=(4.4, 3.15))
    px = [pca_ver[d] for d in DIMS]; py = [100 * pca_cov[d] for d in DIMS]
    rx = [rnd_ver[d] for d in DIMS]; ry = [100 * rnd_cov[d] for d in DIMS]
    ax.plot(px, py, "o-", color="#0072B2", lw=2, label="local PCA")
    ax.plot(rx, ry, "s--", color="#009E73", lw=2, label="matched random")
    for d in [2, 8, 16, 64]:
        ax.annotate(f"d={d}", (pca_ver[d], 100 * pca_cov[d]), fontsize=6, color="#0072B2",
                    xytext=(3, 3), textcoords="offset points")
    for d in [2, 64]:
        yoff = 3
        ax.annotate(f"d={d}", (rnd_ver[d], 100 * rnd_cov[d]), fontsize=6, color="#009E73",
                    xytext=(3, yoff), textcoords="offset points")
    ax.axvline(20.8, color="gray", ls=":", lw=1)
    ax.annotate("mean+clip $L_\\infty$\n(20.8\%, full-dim)", (20.8, 60), fontsize=7, color="gray",
                ha="left", xytext=(4, 0), textcoords="offset points")
    ax.set_xlabel("verification rate (% verified)")
    ax.set_ylabel("same-state variance captured (%)")
    ax.set_xlim(17, 104)
    ax.set_ylim(-4, 105)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="lower center")
    fig.tight_layout(); fig.savefig(ROOT / "p3_coverage_verification_frontier_figure.png", dpi=300)
    print("PCA  (ver%,cov%):", [(pca_ver[d], round(100 * pca_cov[d], 1)) for d in DIMS])
    print("RAND (ver%,cov%):", [(rnd_ver[d], round(100 * rnd_cov[d], 1)) for d in DIMS])
    print("Saved p3_coverage_verification_frontier_figure.png + _data.json")


if __name__ == "__main__":
    main()
