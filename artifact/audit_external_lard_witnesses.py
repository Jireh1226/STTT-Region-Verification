#!/usr/bin/env python3.11
"""
Witness audit for the external LARD-V2 region-tagged verification.

Calibrates plausibility baselines on TRAIN (per-pair Lipschitz threshold, TV, nearest-real L2)
and scores every SAT witness against HELD-OUT TEST images:
  - valid_pixel_frac / recon range  (does the witness leave the [0,1] image cube?)
  - nearest_test_sameclass_L2       (how far from a real same-class runway image?)
  - per_pair_violation_frac         (fraction of adjacent pixel pairs exceeding train 99th pct)
  - tv                              (total variation)

Also computes the headline removal matrix: of anchors where R1 is SAT, how many does each
tighter / data-conditioned region (R5, R4 unbounded, R4 valid-pixel bounded) turn UNSAT.
"""
import json, os, time
from collections import defaultdict
import numpy as np
from external_lard_common import ROOT, load_dataset, load_model, labels_split

COMPOSED = os.path.join(ROOT, "external_lard_composed_models")
TOL = 1e-4


def adjacency(H, W):
    pairs = []
    for r in range(H):
        for c in range(W):
            i = r * W + c
            if c + 1 < W:
                pairs.append((i, r * W + c + 1))
            if r + 1 < H:
                pairs.append((i, (r + 1) * W + c))
    return np.array(pairs)


def tv(x, pairs):
    return float(np.abs(x[pairs[:, 0]] - x[pairs[:, 1]]).sum())


def viol_frac(x, pairs, tau):
    return float((np.abs(x[pairs[:, 0]] - x[pairs[:, 1]]) > tau).mean())


def main():
    d = load_dataset()
    Ws, bs, q1, q2, H, W = load_model()
    Xtr, Xte = d["Xtr"], d["Xte"]
    ytr = labels_split(d, "tr"); yte = labels_split(d, "te")
    pairs = adjacency(H, W)

    # ---- train-calibrated baselines ----
    diffs = np.abs(Xtr[:, pairs[:, 0]] - Xtr[:, pairs[:, 1]])      # (Ntr, Npairs)
    tau = np.quantile(diffs, 0.99, axis=0)                         # per-pair 99th pct
    test_viol = [viol_frac(Xte[k], pairs, tau) for k in range(0, len(Xte), 5)]
    test_tv = [tv(Xte[k], pairs) for k in range(0, len(Xte), 5)]
    # nearest-real (same-class) L2 among test images
    nn_same = []
    for c in range(3):
        idx = np.where(yte == c)[0]
        if len(idx) < 2:
            continue
        for a in idx[:60]:
            o = idx[idx != a]
            nn_same.append(float(np.linalg.norm(Xte[o] - Xte[a], axis=1).min()))
    baselines = {
        "script": "audit_external_lard_witnesses.py",
        "test_per_pair_violation_frac_mean": float(np.mean(test_viol)),
        "test_tv_median": float(np.median(test_tv)),
        "test_sameclass_nn_L2_median": float(np.median(nn_same)),
        "per_pair_tau_q99_calibration": "train, per-pair 99th percentile of |x_i - x_j|",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json.dump(baselines, open(os.path.join(ROOT, "external_lard_real_baselines.json"), "w"), indent=2)

    # ---- score SAT witnesses ----
    res = json.load(open(os.path.join(ROOT, "external_lard_results.json")))["results"]
    audit = []
    pca_cache = {}
    for r in res:
        if r["status"] != "sat":
            continue
        i = r["anchor_test_idx"]; c = r["true_class"]; reg = r["region"]
        if reg.startswith("R4"):
            key = f"a{i}"
            if key not in pca_cache:
                z = np.load(os.path.join(COMPOSED, f"{key}_pca.npz"))
                pca_cache[key] = (z["V"], z["x0"])
            V, x0 = pca_cache[key]
            zz = np.array(r["witness_z"], dtype=np.float64)
            x = x0 + V.T @ zz
        else:
            x = np.array(r["witness"], dtype=np.float64)
        same = np.where(yte == c)[0]
        nn = float(np.linalg.norm(Xte[same] - x, axis=1).min()) if len(same) else None
        audit.append({
            "anchor_test_idx": i, "true_class": c, "target_class": r["target_class"],
            "region": reg, "alpha": r.get("alpha"),
            "valid_pixel_frac": float(((x >= -TOL) & (x <= 1 + TOL)).mean()),
            "recon_min": float(x.min()), "recon_max": float(x.max()),
            "nearest_test_sameclass_L2": nn,
            "per_pair_violation_frac": viol_frac(np.clip(x, 0, 1), pairs, tau),
            "tv": tv(np.clip(x, 0, 1), pairs),
        })

    # ---- headline removal matrix ----
    def rkey(r):
        k = r["region"]
        if r.get("eps") is not None:
            k += f"@{r['eps']}"
        if r.get("alpha") is not None:
            k += f"@a{r['alpha']}"
        return k

    flat = defaultdict(dict)
    for r in res:
        flat[r["anchor_test_idx"]][rkey(r)] = r["status"]

    # broadest pixel ball is the reference for "what broad regions admit"
    R1_REF = "R1_pixel_ball@0.4"
    r1_sat = [a for a in flat if flat[a].get(R1_REF) == "sat"]
    region_keys = sorted({k for a in flat for k in flat[a] if not k.startswith("R1_pixel_ball")})
    removal = {}
    for rk in region_keys:
        removed = sum(1 for a in r1_sat if flat[a].get(rk) == "unsat")
        sat = sum(1 for a in r1_sat if flat[a].get(rk) == "sat")
        to = sum(1 for a in r1_sat if flat[a].get(rk) == "timeout")
        removal[rk] = {"R1_ref": R1_REF, "R1_SAT_anchors": len(r1_sat), "now_unsat": removed,
                       "still_sat": sat, "timeout": to}

    tallies = defaultdict(lambda: defaultdict(int))
    for a in flat:
        for k, v in flat[a].items():
            tallies[k][v] += 1

    out = {
        "provenance": {"script": "audit_external_lard_witnesses.py",
                       "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "baselines": baselines,
        "region_tallies": {k: dict(v) for k, v in tallies.items()},
        "R1_SAT_anchors": len(r1_sat),
        "removal_of_R1_SAT_by_region": removal,
        "n_sat_witnesses_audited": len(audit),
        "witness_audit": audit,
    }
    json.dump(out, open(os.path.join(ROOT, "external_lard_witness_audit.json"), "w"), indent=2)
    print("R1 SAT anchors:", len(r1_sat))
    print("removal of R1-SAT by tighter regions:")
    for k, v in removal.items():
        print(f"  {k:28s} {v}")
    print("WROTE external_lard_real_baselines.json, external_lard_witness_audit.json")


if __name__ == "__main__":
    main()
