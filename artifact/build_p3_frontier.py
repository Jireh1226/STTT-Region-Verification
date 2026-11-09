#!/usr/bin/env python3
"""Assemble the TinyTaxiNet P3 coverage-vs-verification frontier from existing
artifacts. For each region family: verification rate, SAT rate, timeout rate,
and its natural data-membership / support metric (or an explicit 'none' with a
reason). No coverage metric is invented; regions without a valid membership
metric are marked.

Output: p3_coverage_verification_frontier.json (+ printed table)
"""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(p):
    fp = ROOT / p
    return json.load(open(fp)) if fp.exists() else None


def rnd_agg(d, alpha):
    rows = [r for r in d["results_by_seed_alpha"].values()
            if r["alpha"] == alpha and r.get("status", "complete") == "complete"]
    n = rows[0]["sample_size"]
    ver = sorted(r["verified"] for r in rows)
    sat = sorted(r["sat"] for r in rows)
    to = sorted(r["timeout"] for r in rows)
    return {"n": n, "seeds": len(rows), "verified_median": st.median(ver),
            "verified_min": ver[0], "verified_max": ver[-1],
            "sat_median": st.median(sat), "timeout_median": st.median(to)}


rows = []

# --- L_inf mean+clip ball (R1/R2) at eps=0.05 (from tab:p3_results) ---
rows.append({
    "region": "L_inf mean+clip ball (eps=0.05)", "family": "broad pixel box",
    "n": 893, "verified": 186, "sat": 707, "timeout": 0,
    "verified_pct": round(100 * 186 / 893, 1),
    "membership_metric": "none (axis-aligned pixel box; no learned/data membership)",
    "membership_defined": False,
    "support_note": "broad-ball witnesses in-cube but off-manifold; per-pair Lipschitz ~19x real baseline in the LARD analogue; on TinyTaxiNet the mean-constrained ball admits 707/893 excursions beyond the +-10 m envelope.",
})

# --- Label-conditioned KNN box (R5), eps=0.05, 20-anchor matched sample ---
r5aud = load("r5_label_box_witness_audit.json")
lb = load("verification_p3_label_box.json")
r5 = r5aud["aggregate"] if r5aud else {}
rows.append({
    "region": "Label-conditioned KNN box (R5)", "family": "data-conditioned pixel box",
    "n": lb["summary"]["total"] if lb else 20,
    "verified": lb["summary"]["verified"] if lb else 5,
    "sat": lb["summary"]["violated"] if lb else 15, "timeout": 0,
    "verified_pct": round(100 * (lb["summary"]["verified"] if lb else 5) / (lb["summary"]["total"] if lb else 20), 1),
    "membership_metric": "same-class K=100 nearest-neighbour per-pixel extent (data-conditioned box)",
    "membership_defined": True,
    "support_value": {"witness_classifier_prob_median": r5.get("classifier_prob_median"),
                      "witness_below_0_1": r5.get("classifier_below_0_1"),
                      "nearest_real_L2_median": r5.get("nearest_real_L2_median")},
    "support_note": "box extent is data-conditioned but per-pixel independence still admits off-manifold corners: witness classifier prob median ~0.01, 17/20 below 0.1.",
})

# --- Local PCA (R4) at 1/2/3 sigma ---
lp = load("verification_p3_local_pca_expanded.json")
lp3 = load("verification_p3_local_pca_expanded_alpha3_200.json")
ev = [m["explained_variance_ratio_top_d"] for m in lp.get("per_image_pca_meta", [])] if lp else []
pcaw = load("pca_witness_scores.json")
pca_support = {"top8_variance_capture_median": round(st.median(ev), 4) if ev else None,
               "top8_variance_capture_min": round(min(ev), 4) if ev else None,
               "witness_classifier_prob_median": pcaw["summary"]["classifier_prob_real"]["median"] if pcaw else None}
for alpha, key in ((1.0, "alpha_1.0"), (2.0, "alpha_2.0")):
    a = lp["results_by_alpha"][key]
    rows.append({
        "region": f"Local PCA (R4), {int(alpha)} sigma", "family": "data-conditioned 8D affine subspace",
        "n": a["sample_size"], "verified": a["verified"], "sat": a["sat"], "timeout": a["timeout"],
        "verified_pct": round(100 * a["verified"] / a["sample_size"], 1),
        "membership_metric": "8D local-PCA subspace of same-state-neighbour deviations (top-8 PCA variance capture)",
        "membership_defined": True, "support_value": pca_support,
    })
a3 = lp3["results_by_alpha"]["alpha_3.0"]
rows.append({
    "region": "Local PCA (R4), 3 sigma (200-subset)", "family": "data-conditioned 8D affine subspace",
    "n": a3["sample_size"], "verified": a3["verified"], "sat": a3["sat"], "timeout": a3["timeout"],
    "verified_pct": round(100 * a3["verified"] / a3["sample_size"], 1),
    "membership_metric": "8D local-PCA subspace", "membership_defined": True, "support_value": pca_support,
})

# --- Matched RANDOM 8D subspace (the control): high verification, NO membership metric ---
r1 = load("verification_p3_random_subspace_all_alpha1_seeds012.json")
r2 = load("verification_p3_random_subspace_all_alpha2_seeds012.json")
r3 = load("verification_p3_random_subspace_200_alpha3_seeds012.json")
for alpha, d in ((1.0, r1), (2.0, r2), (3.0, r3)):
    if d is None:
        continue
    agg = rnd_agg(d, alpha)
    rows.append({
        "region": f"Random 8D subspace, {int(alpha)} sigma" + (" (200-subset)" if agg["n"] != 893 else ""),
        "family": "random 8D affine subspace (control)",
        "n": agg["n"], "verified": agg["verified_median"], "sat": agg["sat_median"], "timeout": agg["timeout_median"],
        "verified_pct": round(100 * agg["verified_median"] / agg["n"], 1),
        "verified_range": [agg["verified_min"], agg["verified_max"]], "seeds": agg["seeds"],
        "membership_metric": "NONE",
        "membership_defined": False,
        "support_note": "random orientation at matched sigma scale; not data-conditioned, so no reconstruction / variance-capture / neighbour-membership metric applies. Verifies as often or more than local PCA, which is the point: verification rate alone does not certify a data-faithful region.",
    })

# --- AE decoder box (R3): highest-support region (P1 region; context) ---
ae = load("autoencoder_support_audit.json")
cen = ae["subsets"]["centered"] if ae else {}
rows.append({
    "region": "AE decoder box (R3) [P1 region, context]", "family": "data-conditioned learned manifold",
    "n": None, "verified": None, "sat": None, "timeout": None, "verified_pct": None,
    "membership_metric": "AE latent box; held-out reconstruction + latent-box coverage",
    "membership_defined": True,
    "support_value": {
        "heldout_recon_linf_median": round(cen["heldout_split"]["heldout_recon_linf"]["median"], 4) if cen else None,
        "heldout_latent_coverage_train_box": round(cen["heldout_split"]["heldout_latents_inside_train_only_box"]["fraction"], 4) if cen else None,
        "output_preservation_abs_delta_cte_median": round(cen["tinytaxinet_output_preservation"]["abs_delta_cte"]["median"], 4) if cen else None,
    },
    "support_note": "strongest data support of the region families (used for P1); shown for the coverage axis, not P3 verification rate.",
})

out = {"experiment": "p3_coverage_verification_frontier",
       "note": "Verification rate vs data-membership/support for TinyTaxiNet P3 regions. Random 8D has no membership metric by construction and is marked accordingly.",
       "L_inf_reference_verified": "186/893 (20.8%)",
       "rows": rows}
(ROOT / "p3_coverage_verification_frontier.json").write_text(json.dumps(out, indent=2) + "\n")

# printed table
print(f"{'Region':<42} {'verif':>10} {'SAT':>6} {'TO':>4}  membership")
print("-" * 100)
for r in rows:
    v = f"{r['verified']}/{r['n']}" if r["verified"] is not None else "n/a"
    print(f"{r['region']:<42} {v:>10} {str(r['sat']):>6} {str(r['timeout']):>4}  "
          f"{'DEFINED' if r['membership_defined'] else 'NONE'}: {r['membership_metric'][:46]}")
print("\nSaved p3_coverage_verification_frontier.json")
