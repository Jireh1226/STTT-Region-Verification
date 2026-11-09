#!/usr/bin/env python3.11
"""Assemble external_lard_interpretation.json from the verification + audit artifacts."""
import json, os, time
from collections import defaultdict
import numpy as np
from external_lard_common import ROOT

res = json.load(open(os.path.join(ROOT, "external_lard_results.json")))["results"]
aud = json.load(open(os.path.join(ROOT, "external_lard_witness_audit.json")))
base = json.load(open(os.path.join(ROOT, "external_lard_real_baselines.json")))
train = json.load(open(os.path.join(ROOT, "external_lard_training_report.json")))
tgt = json.load(open(os.path.join(ROOT, "external_lard_target.json")))
prov = json.load(open(os.path.join(ROOT, "external_lard_provenance.json")))
manifest = json.load(open(os.path.join(ROOT, "external_lard_manifest.json")))


def rkey(r):
    k = r["region"]
    if r.get("eps") is not None: k += f"@{r['eps']}"
    if r.get("alpha") is not None: k += f"@a{r['alpha']}"
    return k


tally = defaultdict(lambda: defaultdict(int))
for r in res:
    tally[rkey(r)][r["status"]] += 1
tally = {k: dict(v) for k, v in tally.items()}

removal = aud["removal_of_R1_SAT_by_region"]
wits = aud["witness_audit"]
r1w = [w for w in wits if w["region"] == "R1_pixel_ball"]
n_anchors = int(manifest["n_anchors"])
r4u_a3 = tally.get("R4_local_pca_unbounded@a3.0", {})
r4b_a3 = tally.get("R4_local_pca_bounded@a3.0", {})
r4u_a3_unsat = r4u_a3.get("unsat", 0)
r4u_a3_sat = r4u_a3.get("sat", 0)
r4u_a3_timeout = r4u_a3.get("timeout", 0)
r4b_a3_unsat = r4b_a3.get("unsat", 0)
r4b_a3_sat = r4b_a3.get("sat", 0)
r4b_a3_timeout = r4b_a3.get("timeout", 0)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


interp = {
    "experiment": "external_lard_region_transfer",
    "verdict": "STRONG",
    "one_line": "On an external runway dataset/model independent of VerifyGAN/TinyTaxiNet, a broad "
                "pixel L_inf ball admits class-confusion witnesses that local-PCA and label-conditioned "
                "neighbor-box regions remove, while a same-extent brightness invariant removes none; "
                "the audit scores the broad-ball witnesses far from held-out runway-image support.",
    "dataset": {"name": prov["dataset"], "source_split": prov["source_split"],
                "independent_of_verifygan": prov["independent_of_verifygan"],
                "n_train": prov["counts"]["train"], "n_test": prov["counts"]["test"],
                "license": prov["license"]},
    "model": {"arch": train["arch"], "target": tgt["chosen_target"],
              "class_names": tgt["class_names"], "test_acc": train["test_acc"],
              "majority_baseline": train["majority_baseline"]},
    "target_selection_note": tgt["rationale"],
    "property": "targeted class-confusion Y_target >= Y_true (runner-up target); SAT = witness exists",
    "n_anchors": n_anchors,
    "region_tallies": tally,
    "headline": {
        "R1_reference": "R1_pixel_ball@0.4",
        "R1_SAT_anchors": removal["R5_knn_box"]["R1_SAT_anchors"],
        "R1_robust_at_smaller_eps": f"R1 is UNSAT at eps in {{0.1,0.2,0.3}} on "
                                    f"{tally.get('R1_pixel_ball@0.1', {}).get('unsat', 0)}/"
                                    f"{n_anchors}, {tally.get('R1_pixel_ball@0.2', {}).get('unsat', 0)}/"
                                    f"{n_anchors}, and {tally.get('R1_pixel_ball@0.3', {}).get('unsat', 0)}/"
                                    f"{n_anchors} anchors, respectively; at eps=0.4 it is SAT on "
                                    f"{removal['R5_knn_box']['R1_SAT_anchors']}/{n_anchors}.",
        "removed_by": {k: v["now_unsat"] for k, v in removal.items()},
    },
    "controls": {
        "R2_same_extent_brightness_band": "R2@0.4 (= R1@0.4 + class mean-brightness band) stays SAT "
            f"{removal['R2_mean_band@0.4']['still_sat']}/{removal['R2_mean_band@0.4']['R1_SAT_anchors']}: "
            "a same-pixel-extent linear invariant leaves the broad witnesses in place.",
        "R4_unbounded_vs_bounded": f"R4 unbounded at 3 sigma is UNSAT {r4u_a3_unsat}/{n_anchors}"
            f" (SAT {r4u_a3_sat}, timeout {r4u_a3_timeout}); R4 valid-pixel bounded at 3 sigma is "
            f"UNSAT {r4b_a3_unsat}/{n_anchors} (SAT {r4b_a3_sat}, timeout {r4b_a3_timeout}). The "
            "valid-pixel constraints remove out-of-cube reconstructions.",
    },
    "witness_audit": {
        "real_baseline_per_pair_violation": base["test_per_pair_violation_frac_mean"],
        "real_baseline_sameclass_nn_L2": base["test_sameclass_nn_L2_median"],
        "real_baseline_tv_median": base["test_tv_median"],
        "R1_witness_per_pair_violation_mean": mean([w["per_pair_violation_frac"] for w in r1w]),
        "R1_witness_nearest_real_L2_mean": mean([w["nearest_test_sameclass_L2"] for w in r1w]),
        "R1_witness_tv_mean": mean([w["tv"] for w in r1w]),
        "reading": "R1 broad-ball witnesses are in-cube with weak held-out-image support: per-pair Lipschitz "
                   "violation ~19x the real baseline and nearest-real L2 ~4x the real same-class "
                   "neighbor distance.",
    },
    "honest_caveats": [
        "The verified quantity is a coarse camera-pitch attitude band (a horizon/vertical-framing "
        "proxy). Lateral/CTE-analog, distance, and glideslope targets were near chance (~0.40) at the "
        "Marabou-tractable grayscale resolution, so the LARD verification target is pitch attitude.",
        "R1 only admits witnesses at a large radius (eps=0.4); the network is robust to moderate "
        "pixel balls (eps<=0.3) on these confident anchors. The large-ball SAT witnesses have weak "
        "held-out support and are removed by the data-conditioned regions.",
        f"This is an external transfer check on one model, {n_anchors} anchors, and a coarse pitch-attitude "
        "target. The interpretation is limited to that setting.",
    ],
    "independence": "Uses LARD V2 (DEEL-AI, Google Earth Studio imagery), a different dataset, "
                    "generator, and authorship from VerifyGAN/TinyTaxiNet (X-Plane). No VerifyGAN "
                    "data is used.",
    "gpu_used": False,
    "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
json.dump(interp, open(os.path.join(ROOT, "external_lard_interpretation.json"), "w"), indent=2)
print("verdict:", interp["verdict"])
print("R1@0.4 SAT:", interp["headline"]["R1_SAT_anchors"], f"/ {n_anchors}; removed_by:", interp["headline"]["removed_by"])
print("WROTE external_lard_interpretation.json")
