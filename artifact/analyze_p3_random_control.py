#!/usr/bin/env python3
"""Compare the matched random-subspace P3 control against local PCA and the
mean-constrained L-infinity ball. Writes p3_random_control_summary.json and
prints a short textual summary.

Local-PCA reference numbers come from verification_p3_local_pca_expanded.json
(alpha 1sigma, 2sigma on all 893) and verification_p3_local_pca_expanded_alpha3_200.json
(alpha 3sigma on the 200-anchor subset). The L-infinity mean+clip P3 baseline at
epsilon=0.05 is 186/893 (Table tab:p3_results in the paper).
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LINF_EPS05 = {"verified": 186, "n": 893, "label": "mean+clip L_inf, eps=0.05"}

RANDOM_FILES = {
    1.0: "verification_p3_random_subspace_all_alpha1_seeds012.json",
    2.0: "verification_p3_random_subspace_all_alpha2_seeds012.json",
    3.0: "verification_p3_random_subspace_200_alpha3_seeds012.json",
}


def load(path):
    p = ROOT / path
    return json.load(open(p)) if p.exists() else None


def cte_he_breakdown_localpca(per_image):
    cte = he = 0
    for e in per_image.values():
        if e.get("category") not in ("sat",):
            continue
        sq = e.get("sub_queries", {})
        if any(sq.get(t, {}).get("status") == "sat" for t in ("a", "b")):
            cte += 1
        if any(sq.get(t, {}).get("status") == "sat" for t in ("c", "d")):
            he += 1
    return {"anchors_cte_sat": cte, "anchors_he_sat": he}


def localpca_ref():
    ref = {}
    exp = load("verification_p3_local_pca_expanded.json")
    a3 = load("verification_p3_local_pca_expanded_alpha3_200.json")
    for alpha, src in ((1.0, exp), (2.0, exp), (3.0, a3)):
        if src is None:
            continue
        a = src["results_by_alpha"].get(f"alpha_{alpha}")
        if not a:
            continue
        ref[alpha] = {
            "verified": a["verified"], "sat": a["sat"], "timeout": a["timeout"],
            "n": a["sample_size"],
            **cte_he_breakdown_localpca(a["per_image"]),
        }
    return ref


def random_ref():
    ref = {}
    for alpha, fname in RANDOM_FILES.items():
        d = load(fname)
        if d is None:
            continue
        rows = [r for r in d["results_by_seed_alpha"].values()
                if r["alpha"] == alpha and r.get("status", "complete") == "complete"]
        if not rows:
            continue
        n = rows[0]["sample_size"]
        ver = sorted(r["verified"] for r in rows)
        sat = sorted(r["sat"] for r in rows)
        to = sorted(r["timeout"] for r in rows)
        # aggregate CTE/HE across seeds (sum of sat-subquery axis hits)
        cte = sum(r["sat_subquery_breakdown"]["cte_upper"] + r["sat_subquery_breakdown"]["cte_lower"] for r in rows)
        he = sum(r["sat_subquery_breakdown"]["he_upper"] + r["sat_subquery_breakdown"]["he_lower"] for r in rows)
        ref[alpha] = {
            "n": n, "n_seeds": len(rows), "seeds": [r["seed"] for r in rows],
            "verified_min": ver[0], "verified_median": statistics.median(ver), "verified_max": ver[-1],
            "sat_min": sat[0], "sat_median": statistics.median(sat), "sat_max": sat[-1],
            "timeout_min": to[0], "timeout_median": statistics.median(to), "timeout_max": to[-1],
            "sat_subquery_cte_hits_total": cte, "sat_subquery_he_hits_total": he,
            "basis_validation": d.get("basis_validation"),
            "mean_invariant_check": d.get("mean_invariant_check"),
            "per_seed_verified": {r["seed"]: r["verified"] for r in rows},
        }
    return ref


def pct(x, n):
    return round(100.0 * x / n, 1)


def main():
    lp = localpca_ref()
    rnd = random_ref()
    summary = {"linf_baseline_eps05": LINF_EPS05, "local_pca": lp, "random_subspace": rnd,
               "comparison": {}}
    lines = []
    lines.append(f"L_inf mean+clip P3 @eps=0.05: {LINF_EPS05['verified']}/{LINF_EPS05['n']} "
                 f"({pct(LINF_EPS05['verified'], LINF_EPS05['n'])}%) verified")
    for alpha in (1.0, 2.0, 3.0):
        if alpha not in rnd or alpha not in lp:
            continue
        r, l = rnd[alpha], lp[alpha]
        n = l["n"]
        # gain decomposition vs L_inf baseline (only meaningful on full 893 runs)
        linf_v = LINF_EPS05["verified"] if n == 893 else None
        cmp = {
            "n": n,
            "local_pca_verified": l["verified"], "local_pca_pct": pct(l["verified"], n),
            "random_verified_min": r["verified_min"], "random_verified_median": r["verified_median"],
            "random_verified_max": r["verified_max"],
            "random_pct_median": pct(r["verified_median"], n),
            "random_sat_median": r["sat_median"], "random_timeout_median": r["timeout_median"],
            "random_cte_vs_he": {"cte_hits": r["sat_subquery_cte_hits_total"],
                                 "he_hits": r["sat_subquery_he_hits_total"]},
        }
        if linf_v is not None:
            total_gain = l["verified"] - linf_v
            dim_gain = r["verified_median"] - linf_v          # explained by low-dim region
            align_gain = l["verified"] - r["verified_median"]  # explained by PCA alignment
            cmp["linf_verified"] = linf_v
            cmp["total_gain_over_linf"] = total_gain
            cmp["dimension_gain_random_over_linf"] = dim_gain
            cmp["alignment_gain_pca_over_random"] = align_gain
            cmp["dimension_share_pct"] = round(100.0 * dim_gain / total_gain, 1) if total_gain else None
            cmp["alignment_share_pct"] = round(100.0 * align_gain / total_gain, 1) if total_gain else None
        summary["comparison"][f"alpha_{alpha}"] = cmp
        seg = (f"alpha={alpha}sigma (n={n}): local PCA {l['verified']}/{n} ({pct(l['verified'],n)}%)  vs  "
               f"random median {r['verified_median']}/{n} ({pct(r['verified_median'],n)}%) "
               f"[min {r['verified_min']}, max {r['verified_max']}]  "
               f"random SAT med {r['sat_median']}, timeout med {r['timeout_median']}")
        if linf_v is not None:
            seg += (f"\n     gain over L_inf: total {cmp['total_gain_over_linf']}, "
                    f"dimension(random) {cmp['dimension_gain_random_over_linf']} "
                    f"({cmp['dimension_share_pct']}%), "
                    f"alignment(PCA) {cmp['alignment_gain_pca_over_random']} "
                    f"({cmp['alignment_share_pct']}%)")
        lines.append(seg)

    summary["textual"] = lines
    json.dump(summary, open(ROOT / "p3_random_control_summary.json", "w"), indent=2)
    print("\n".join(lines))
    print("\nSaved p3_random_control_summary.json")


if __name__ == "__main__":
    main()
