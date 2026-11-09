"""Print the portfolio's verification results from saved JSON using only Python."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize(root):
    def load(name):
        return json.loads((root / name).read_text())

    pixel = load("verification_full_sample_centered.json")["P3_mean_clip"]["epsilon_0.05"]
    pca = load("verification_p3_local_pca_expanded.json")["results_by_alpha"]["alpha_1.0"]
    pca_counts = dict(Counter(row["category"] for row in pca["per_image"].values()))
    random = load("verification_p3_random_subspace_all_alpha1_seeds012.json")
    random_counts = {
        key: dict(Counter(row["category"] for row in value["per_image"].values()))
        for key, value in random["results_by_seed_alpha"].items()
    }
    wrong = load("reproduced_empirical_baseline_direct.json")["wrong_sign"]["offset_abs_cte_gt_2"]
    compact = load("mnist_protocol_results.json")
    mnist_rows = load("external_mnistfc_matrix_results.json")["results"]
    mnist_rows += load("external_mnistfc_1sigma_rest27_results.json")["results"]
    mnist_pca = {}
    for row in mnist_rows:
        if row["region"] == "R4_bounded" and row.get("alpha") == 1.0:
            idx = row["anchor_idx"]
            if idx in mnist_pca:
                assert mnist_pca[idx] == row["status"], "Conflicting MNIST records"
            mnist_pca[idx] = row["status"]
    lard = load("external_lard_results.json")["results"]
    lard_broad = [row for row in lard if row["region"] == "R1_pixel_ball" and row.get("eps") == 0.4]
    lard_pca = {
        str(scale): dict(Counter(row["status"] for row in lard
                                if row["region"] == "R4_local_pca_bounded"
                                and row.get("alpha") == scale))
        for scale in (1.0, 2.0, 3.0)
    }
    return {
        "source": "Saved experiment records in artifact/",
        "tinytaxinet": {
            "pixel_region_epsilon_0.05": {key: pixel[key] for key in ("n_total", "verified", "violated")},
            "local_pca_one_sigma": pca_counts,
            "random_eight_dimensions_one_sigma": random_counts,
            "wrong_side_predictions": {"wrong": wrong["wrong"], "images": wrong["n"], "rate": wrong["rate"]},
        },
        "compact_mnist": {
            "test_images": compact["provenance"]["n_anchors"],
            "per_image_queries": len(compact["per_anchor"]),
        },
        "vnncomp_mnist": {
            "local_pca_one_sigma": dict(Counter(mnist_pca.values())),
            "matched_random_queries": dict(Counter(
                row["status"] for row in load("external_mnistfc_random_subspace_results.json")["results"])),
        },
        "lard_google_earth_studio": {
            "broad_pixel_region_epsilon_0.4": dict(Counter(row["status"] for row in lard_broad)),
            "valid_pixel_local_pca": lard_pca,
            "matched_random_queries": dict(Counter(
                row["status"] for row in load("external_lard_random_subspace_results.json")["results"])),
        },
        "module_checks": {
            name: load(name)["summary"]
            for name in ("module_smoke_taxinet.json", "module_smoke_mnist.json")
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path(__file__).resolve().parent / "artifact")
    args = parser.parse_args()
    print(json.dumps(summarize(args.artifact), indent=2))
