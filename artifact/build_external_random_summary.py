#!/usr/bin/env python3
"""Summarize the external matched random-subspace controls (LARD + mnist_fc)
against the existing local-PCA and broad-region results, and state the
interpretation. Writes external_random_control_summary.json + prints a summary.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(p):
    fp = ROOT / p
    return json.load(open(fp)) if fp.exists() else None


out = {"experiment": "external_matched_random_subspace_controls",
       "question": "How much of the external local-PCA witness removal is explained by matched random 8D subspaces?",
       "control": "deterministic zero-mean orthonormal random 8D basis; same anchors, property, per-anchor local-PCA sigma scale, valid-pixel/mean/clip conventions, solver; seeds 0,1,2."}

# ---- LARD ----
lard = load("external_lard_random_subspace_results.json")
lard_interp = load("external_lard_interpretation.json")
if lard:
    agg = defaultdict(Counter)
    for r in lard["results"]:
        agg[(r["region"], r["alpha"])][r["status"]] += 1
    out["LARD"] = {
        "network": "1152-64-32-3 ReLU MLP (pitch tercile), external LARD V2",
        "broad_R1@0.4_SAT_baseline": lard_interp["region_tallies"]["R1_pixel_ball@0.4"] if lard_interp else None,
        "local_pca_bounded_removed": {a: lard_interp["headline"]["removed_by"].get(f"R4_local_pca_bounded@a{a}")
                                      for a in ("1.0", "2.0", "3.0")} if lard_interp else None,
        "R2_same_extent_removed": lard_interp["headline"]["removed_by"].get("R2_mean_band@0.4") if lard_interp else None,
        "random_subspace": {f"{reg}@{al}": dict(cnt) for (reg, al), cnt in sorted(agg.items())},
        "basis_validation": lard["basis_validation"],
        "reading": "Random 8D subspaces are UNSAT on all 150 queries at every alpha for both bounded and unbounded encodings. They remove every broad-ball witness removed by local PCA. The same-extent brightness band (R2) removes no witnesses, separating the LARD removal from pixel-ball extent. Valid-pixel constraints do not change the random-control result.",
    }

# ---- mnist_fc ----
mf = load("external_mnistfc_random_subspace_results.json")
mf_val = load("external_mnistfc_random_subspace_validation.json")
matrix = load("external_mnistfc_matrix_results.json")
if mf:
    agg = defaultdict(Counter)
    for r in mf["results"]:
        agg[r["alpha"]][r["status"]] += 1
    bsat = [r for r in mf["results"] if r["broad_sat_anchor"] and r["alpha"] == 1.0]
    # local-PCA matrix comparison
    lp = defaultdict(Counter)
    if matrix:
        for r in matrix["results"]:
            if r["region"] == "R4_bounded":
                lp[r["alpha"]][r["status"]] += 1
    out["mnist_fc"] = {
        "network": "VNN-COMP mnist_fc mnist-net_256x2 (not trained by us)",
        "local_pca_bounded_matrix": {a: dict(lp[a]) for a in sorted(lp)},
        "local_pca_headline": "bounded local PCA @1sigma UNSAT on all 50; removes 23/23 resolved broad witnesses",
        "random_subspace_bounded": {f"{a}sigma": dict(agg[a]) for a in sorted(agg)},
        "random_broad_SAT_anchors@1sigma": {"n": len(bsat), "status": dict(Counter(r["status"] for r in bsat))},
        "output_indexing_PASS": mf_val["validation"]["output_indexing"]["PASS"] if mf_val else None,
        "basis_validation": mf_val["validation"]["basis_validation"] if mf_val else None,
        "reading": "Random 8D subspaces with valid pixels are UNSAT on all 450 queries (50 anchors x 3 seeds x 3 alphas), including 69/69 broad-SAT-anchor queries at 1 sigma. They remove every resolved broad witness at every alpha, at least as completely as local PCA, which has one SAT at 2 sigma. The mnist_fc removal is explained by a bounded low-dimensional region.",
    }

# ---- timeout hygiene (if rerun present) ----
rerun = load("external_mnistfc_broad_timeout_rerun_results.json")
if rerun:
    ag = defaultdict(Counter)
    for r in rerun["results"]:
        ag[r["eps"]][r["status"]] += 1
    out["mnist_fc_broad_timeout_rerun_10x"] = {f"eps_{e}": dict(ag[e]) for e in sorted(ag)}

out["unified_conclusion"] = ("Across TinyTaxiNet (P3), external LARD, and external mnist_fc, a matched random 8D "
                             "subspace verifies or removes at least as well as local PCA. The local-PCA verification-rate "
                             "and broad-witness-removal gains are explained by low-dimensional-region effects; local PCA "
                             "still provides a data-aligned support model and recovers witnesses random subspaces miss in P3.")

(ROOT / "external_random_control_summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
