"""
Output-margin analysis: per image, correlate the nominal CTE margin with the
certified safety radius eps*. The CTE margin is min(10 - CTE_pred, 10 + CTE_pred);
the certified radius comes from verification_p3_certified_radius.json.

Hypothesis: images whose nominal CTE prediction is closer to the +/-10 m
violation thresholds should have smaller eps*, because a smaller perturbation
is enough to push the network past the bound.
"""
import json

import h5py
import numpy as np
from maraboupy import Marabou


NNET_PATH = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA_PATH = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
RADIUS_JSON = "verification_p3_certified_radius.json"
OUTPUT_JSON = "p3_margin_vs_radius.json"

P3_CTE_BOUND = 10.0
P3_HE_BOUND = 90.0


def main():
    # Load data and certified radii
    radius = json.load(open(RADIUS_JSON))
    with h5py.File(DATA_PATH, "r") as f:
        images = f["y_train"][:].reshape(-1, 128).astype(np.float64)
        labels = f["X_train"][:].astype(np.float64)

    # Get nominal predictions
    net = Marabou.read_nnet(NNET_PATH)

    rows = []
    for entry in radius["per_image"]:
        idx = entry["dataset_index"]
        x0 = images[idx]
        out = np.array(net.evaluate([x0])).flatten()
        cte_pred, he_pred = float(out[0]), float(out[1])

        cte_margin = float(P3_CTE_BOUND - abs(cte_pred))
        he_margin = float(P3_HE_BOUND - abs(he_pred))
        rows.append({
            "dataset_index": idx,
            "true_CTE": entry["true_CTE"],
            "CTE_pred": round(cte_pred, 4),
            "HE_pred": round(he_pred, 4),
            "cte_margin": round(cte_margin, 4),
            "he_margin": round(he_margin, 4),
            "certified_radius": entry["radius"],
        })

    cte_m = np.array([r["cte_margin"] for r in rows])
    he_m = np.array([r["he_margin"] for r in rows])
    eps_star = np.array([r["certified_radius"] for r in rows])

    def pearson(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return float(np.corrcoef(a, b)[0, 1])

    def spearman(a, b):
        from scipy.stats import spearmanr
        rho, p = spearmanr(a, b)
        return float(rho), float(p)

    rho_cte_p, p_cte_p = spearman(cte_m, eps_star)
    rho_he_p, p_he_p = spearman(he_m, eps_star)

    summary = {
        "n_images": len(rows),
        "cte_margin_min": float(cte_m.min()),
        "cte_margin_median": float(np.median(cte_m)),
        "cte_margin_max": float(cte_m.max()),
        "he_margin_min": float(he_m.min()),
        "he_margin_median": float(np.median(he_m)),
        "he_margin_max": float(he_m.max()),
        "pearson_cte_margin_vs_eps": pearson(cte_m, eps_star),
        "pearson_he_margin_vs_eps": pearson(he_m, eps_star),
        "spearman_cte_margin_vs_eps": rho_cte_p,
        "spearman_p_cte": p_cte_p,
        "spearman_he_margin_vs_eps": rho_he_p,
        "spearman_p_he": p_he_p,
    }

    print("Per-image margin vs. certified radius:\n")
    print(f"{'idx':>5} {'true_CTE':>9} {'CTE_pred':>9} {'cte_margin':>11} "
          f"{'he_margin':>10} {'eps*':>7}")
    for r in sorted(rows, key=lambda x: x["certified_radius"]):
        print(f"{r['dataset_index']:>5} {r['true_CTE']:>+9.3f} "
              f"{r['CTE_pred']:>+9.3f} {r['cte_margin']:>11.3f} "
              f"{r['he_margin']:>10.3f} {r['certified_radius']:>7.4f}")
    print(f"\nCTE margin range: [{summary['cte_margin_min']:.3f}, "
          f"{summary['cte_margin_max']:.3f}], median "
          f"{summary['cte_margin_median']:.3f}")
    print(f"HE  margin range: [{summary['he_margin_min']:.3f}, "
          f"{summary['he_margin_max']:.3f}], median "
          f"{summary['he_margin_median']:.3f}")
    print(f"\nPearson  CTE-margin vs eps*: {summary['pearson_cte_margin_vs_eps']:+.3f}")
    print(f"Pearson  HE-margin  vs eps*: {summary['pearson_he_margin_vs_eps']:+.3f}")
    print(f"Spearman CTE-margin vs eps*: {summary['spearman_cte_margin_vs_eps']:+.3f}  "
          f"(p = {summary['spearman_p_cte']:.4f})")
    print(f"Spearman HE-margin  vs eps*: {summary['spearman_he_margin_vs_eps']:+.3f}  "
          f"(p = {summary['spearman_p_he']:.4f})")

    out = {"per_image": rows, "summary": summary}
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
