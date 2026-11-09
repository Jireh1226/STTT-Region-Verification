"""
Three-way categorization of the P2 sample (CTE 1.0m and CTE 1.5m thresholds):
  (a) nominally wrong: |CTE_pred - CTE_true| >= threshold OR |HE_pred - HE_true| >= 5deg
  (b) nominally correct but fragile: passes at x0, but some sub-query returns SAT
      (i.e. epsilon-ball of 0.02 contains an input violating the spec)
  (c) verified robust: every sub-query returns UNSAT in the epsilon-ball

This separates baseline accuracy from local input-space fragility, which the
combined "0/20 verified" headline conflates.
"""
import json

P2_RESULTS = "verification_results_v2.json"
HE_BOUND = 5.0  # degrees


def categorize(image_entry, cte_bound):
    cte_err = abs(image_entry["nominal_CTE_error"])
    he_err = abs(image_entry["nominal_HE_error"])
    nominally_correct = (cte_err < cte_bound) and (he_err < HE_BOUND)
    verified = bool(image_entry.get("verified", False))

    if not nominally_correct:
        return "nominally_wrong"
    if not verified:
        return "fragile"
    return "robust"


def analyze_threshold(p2_block, cte_bound):
    results = p2_block["results"]
    counts = {"nominally_wrong": 0, "fragile": 0, "robust": 0}
    by_image = []

    for k in sorted(results.keys(), key=lambda s: int(s.split("_")[1])):
        e = results[k]
        cat = categorize(e, cte_bound)
        counts[cat] += 1

        # Margin: how far from the tightest violated bound is the nominal prediction
        cte_margin = cte_bound - abs(e["nominal_CTE_error"])
        he_margin = HE_BOUND - abs(e["nominal_HE_error"])
        # Fragility witness: which sub-queries (if any) return SAT among nominally-correct images
        sub_status = {q: e["sub_queries"][q]["status"] for q in e["sub_queries"]}
        sat_subs = [q for q, s in sub_status.items() if s == "sat"]

        by_image.append({
            "image": k,
            "dataset_index": e["dataset_index"],
            "CTE_true": e["CTE_true"],
            "HE_true": e["HE_true"],
            "CTE_pred": e["CTE_predicted"],
            "HE_pred": e["HE_predicted"],
            "nominal_CTE_error": e["nominal_CTE_error"],
            "nominal_HE_error": e["nominal_HE_error"],
            "cte_margin": round(cte_margin, 4),
            "he_margin": round(he_margin, 4),
            "category": cat,
            "verified": bool(e.get("verified", False)),
            "sat_sub_queries": sat_subs,
        })

    return counts, by_image


def main():
    d = json.load(open(P2_RESULTS))
    out = {}
    for tag, cte_bound in [("CTE_1.0m", 1.0), ("CTE_1.5m", 1.5)]:
        block_key = f"P2_{tag}"
        block = d["P2"][block_key]
        counts, by_image = analyze_threshold(block, cte_bound)
        out[tag] = {
            "cte_bound": cte_bound,
            "he_bound": HE_BOUND,
            "epsilon": block["epsilon"],
            "counts": counts,
            "fraction_robust": round(counts["robust"] / 20, 4),
            "fraction_fragile_given_correct": round(
                counts["fragile"] / max(1, counts["fragile"] + counts["robust"]), 4),
            "by_image": by_image,
        }

    with open("p2_three_way_categorization.json", "w") as f:
        json.dump(out, f, indent=2)

    print("P2 three-way categorization (epsilon = 0.02)\n")
    print(f"{'Threshold':<12} {'Wrong':>8} {'Fragile':>10} {'Robust':>8}  "
          f"{'P(fragile|correct)':>18}")
    print("-" * 70)
    for tag in ["CTE_1.0m", "CTE_1.5m"]:
        c = out[tag]["counts"]
        denom = c["fragile"] + c["robust"]
        pf = c["fragile"] / denom if denom else 0
        print(f"{tag:<12} {c['nominally_wrong']:>8} {c['fragile']:>10} "
              f"{c['robust']:>8}  {pf:>17.2%}")

    print("\nDetailed per-image breakdown (CTE_1.0m):")
    print(f"{'#':>3} {'idx':>5} {'CTE_pred':>9} {'CTE_err':>8} {'HE_err':>7} "
          f"{'cte_marg':>9} {'category':<18} {'sat_subs'}")
    for r in out["CTE_1.0m"]["by_image"]:
        n = r["image"].split("_")[1]
        print(f"{n:>3} {r['dataset_index']:>5} {r['CTE_pred']:>9.3f} "
              f"{r['nominal_CTE_error']:>8.3f} {r['nominal_HE_error']:>7.3f} "
              f"{r['cte_margin']:>9.3f} {r['category']:<18} "
              f"{','.join(s.split('img')[1].rstrip('abcd')+s[-1] for s in r['sat_sub_queries']) or '-'}")


if __name__ == "__main__":
    main()
