#!/usr/bin/env python3
"""Analysis + witness audit for the bounded external mnist_fc transfer run.

Combines the bounded results (R4+pixel-bounds, R3+pixel-bounds) with the
existing unbounded results (R1, R4, R3) to answer whether the data-conditioned
transfer survives valid-pixel constraints. Bounded SAT witnesses are scored
RAW (reconstructed without clipping); validity in [0,1] is reported.

Outputs: external_mnistfc_bounded_transfer_witness_audit.json
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import region_audit as ra

ROOT = Path(__file__).resolve().parent
AE = ROOT / "mnist_class_aes.npz"
SHAPE = (28, 28)


def load_mnist():
    m = np.load(os.path.expanduser("~/.keras/datasets/mnist.npz"))
    return (m["x_train"].reshape(-1, 784) / 255.0, m["y_train"],
            m["x_test"].reshape(-1, 784) / 255.0, m["y_test"])


def load(f):
    p = ROOT / f
    return json.loads(p.read_text())["results"] if p.exists() else []


def main():
    Xtr, ytr, Xte, yte = load_mnist()
    ae = np.load(AE)
    pairs, taus = ra.calibrate_per_pair(Xtr, SHAPE, q=99)

    def ppv(x):
        return float(np.mean([abs(x[a] - x[b]) > t for (a, b), t in zip(pairs, taus)]))

    r1 = load("external_mnistfc_r1_r2_results.json")
    r4u = load("external_mnistfc_r4_results.json")
    r3u = load("external_mnistfc_r3_results.json")
    bnd = load("external_mnistfc_bounded_transfer_results.json")

    # per-anchor verdict map
    A = defaultdict(dict)
    for r in r1:
        A[r["anchor_idx"]]["R1_%.2f" % r["eps"]] = r["status"]
    for r in r4u:
        A[r["anchor_idx"]]["R4u_%.0f" % r["alpha"]] = r["status"]
    for r in r3u:
        A[r["anchor_idx"]]["R3u"] = r["status"]
    for r in bnd:
        if r["region"].startswith("R4"):
            A[r["anchor_idx"]]["R4b_%.0f" % r["alpha"]] = r["status"]
        else:
            A[r["anchor_idx"]]["R3b"] = r["status"]

    done = sorted(a for a in A if "R3b" in A[a])  # anchors with bounded run done

    # region SAT/UNSAT/timeout tallies (over anchors with bounded done)
    tallies = {}
    for key in ["R1_0.05", "R4u_1", "R4b_1", "R4u_2", "R4b_2", "R4u_3", "R4b_3", "R3u", "R3b"]:
        tallies[key] = dict(Counter(A[a].get(key) for a in done))

    # R1@0.05 SAT removed by R4 (unbounded) and by R4+bounds (at 1 sigma)
    r1sat = [a for a in done if A[a].get("R1_0.05") == "sat"]
    removed_R4u = sum(1 for a in r1sat if A[a].get("R4u_1") == "unsat")
    removed_R4b = sum(1 for a in r1sat if A[a].get("R4b_1") == "unsat")
    r4b1_timeout_on_r1sat = sum(1 for a in r1sat if A[a].get("R4b_1") == "timeout")

    # R3 SAT eliminated by pixel bounds (R3u sat AND R3b unsat); report timeouts
    r3u_sat = [a for a in done if A[a].get("R3u") == "sat"]
    r3_eliminated = sum(1 for a in r3u_sat if A[a].get("R3b") == "unsat")
    r3b_timeout = sum(1 for a in r3u_sat if A[a].get("R3b") == "timeout")

    # unbounded R4 SAT that flip to bounded UNSAT (out-of-cube), per alpha
    flips = {}
    for al in ["1", "2", "3"]:
        sat_u = [a for a in done if A[a].get(f"R4u_{al}") == "sat"]
        flips[f"R4@{al}sigma_unbounded_SAT_to_bounded_UNSAT"] = \
            f"{sum(1 for a in sat_u if A[a].get(f'R4b_{al}') == 'unsat')}/{len(sat_u)}"

    # witness audit for bounded SAT witnesses (raw, unclipped)
    waudit = []
    for r in bnd:
        if r["status"] != "sat" or "witness_z" not in r:
            continue
        c = r["true_class"]; z = np.asarray(r["witness_z"])
        if r["region"].startswith("R4"):
            x0 = Xte[r["anchor_idx"]]
            Xc = Xtr[ytr == c]
            nb = Xc[np.argsort(np.linalg.norm(Xc - x0[None, :], axis=1))[:100]]
            _, _, Vt = np.linalg.svd(nb - x0[None, :], full_matrices=False)
            x = x0 + Vt[:8].T @ z
        else:
            h = np.maximum(ae[f"dec1_W_class{c}"] @ z + ae[f"dec1_b_class{c}"], 0.0)
            x = ae[f"dec2_W_class{c}"] @ h + ae[f"dec2_b_class{c}"]
        Xec = Xte[yte == c]
        waudit.append({"anchor_idx": r["anchor_idx"], "region": r["region"],
                       "alpha": r.get("alpha"),
                       "frac_in_unit": round(float(((x >= -1e-6) & (x <= 1 + 1e-6)).mean()), 4),
                       "recon_min": round(float(x.min()), 3), "recon_max": round(float(x.max()), 3),
                       "nearest_test_sameclass_L2": round(float(np.min(np.linalg.norm(Xec - x[None, :], axis=1))), 3),
                       "per_pair_violation_frac": round(ppv(np.clip(x, 0, 1)), 3)})

    out = {
        "provenance": {"script": "audit_external_mnistfc_bounded.py",
                       "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                       "anchors_with_bounded_done": len(done)},
        "region_tallies": tallies,
        "R1@0.05_SAT_anchors": len(r1sat),
        "R1_SAT_removed_by_R4_unbounded_1sigma": removed_R4u,
        "R1_SAT_removed_by_R4_bounded_1sigma": removed_R4b,
        "R1_SAT_with_R4_bounded_1sigma_timeout": r4b1_timeout_on_r1sat,
        "R3_SAT_eliminated_by_pixel_bounds": r3_eliminated,
        "R3_bounded_timeout_count": r3b_timeout,
        "out_of_cube_flips": flips,
        "n_bounded_sat_witnesses": len(waudit),
        "witness_audit": waudit,
    }
    (ROOT / "external_mnistfc_bounded_transfer_witness_audit.json").write_text(
        json.dumps(out, indent=2) + "\n")

    print(f"anchors done: {len(done)}")
    for k, v in tallies.items():
        print(f"  {k:8s}: {v}")
    print(f"R1@0.05 SAT anchors: {len(r1sat)}")
    print(f"  removed by R4 unbounded@1s: {removed_R4u}")
    print(f"  removed by R4 BOUNDED@1s:   {removed_R4b}  (R4b@1s timeout on R1-SAT: {r4b1_timeout_on_r1sat})")
    print(f"R3 unbounded SAT: {len(r3u_sat)} | eliminated by bounds: {r3_eliminated} | bounded timeout: {r3b_timeout}")
    print(f"out-of-cube flips (R4 unbounded SAT -> bounded UNSAT): {flips}")
    if waudit:
        print(f"bounded SAT witnesses: {len(waudit)}, frac_in_unit median "
              f"{np.median([w['frac_in_unit'] for w in waudit]):.3f}")
    print("Wrote external_mnistfc_bounded_transfer_witness_audit.json")


if __name__ == "__main__":
    main()
