#!/usr/bin/env python3
"""Matched random-subspace control for the external VNN-COMP mnist_fc bounded
local-PCA removals.

Same 50-anchor manifest, same target labels, same valid-pixel composed two-output
ONNX path (logits Y_0..9, pixels Y_10..793 with 0<=pixel<=1), same output-index
validation, same Marabou CLI settings, and the SAME per-anchor local-PCA scale
vector sigma (from B.pca_basis). The ONLY change: the 8 basis directions are
deterministic zero-mean orthonormal RANDOM directions instead of PCA.

Critical comparison: bounded local PCA at 1sigma is UNSAT on all 50 and removes
23/23 resolved broad witnesses. Does a random 8D subspace also remove them?

Runs bounded (valid-pixel) queries. Broad-SAT anchors first. Incremental saves.
Output: external_mnistfc_random_subspace_results.json
        external_mnistfc_random_subspace_validation.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import external_mnistfc_matrix as M      # cli, pca_basis, auto_indexing_validation, witness_checks
import verify_external_mnistfc_bounded_transfer as B  # compose_pca_two, emit_bounded, load_mnist, D

ROOT = M.ROOT
OUT = ROOT / "external_mnistfc_random_subspace_results.json"
VAL = ROOT / "external_mnistfc_random_subspace_validation.json"


def random_zero_mean_basis(seed, idx, d, n):
    rng = np.random.default_rng(int(seed) * 1_000_003 + int(idx))
    cols = []
    for _ in range(d):
        v = rng.normal(size=n); v = v - v.mean()
        for q in cols:
            v = v - np.dot(v, q) * q
        nrm = np.linalg.norm(v)
        while nrm < 1e-12:
            v = rng.normal(size=n); v = v - v.mean()
            for q in cols:
                v = v - np.dot(v, q) * q
            nrm = np.linalg.norm(v)
        cols.append(v / nrm)
    return np.vstack(cols)


def basis_checks(V):
    d = V.shape[0]
    return (float(np.max(np.abs(V @ V.T - np.eye(d)))), float(np.max(np.abs(V.mean(axis=1)))))


def now():
    return datetime.now(timezone.utc).isoformat()


def save(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="1.0")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    Xtr, ytr, Xte, yte = B.load_mnist()
    manifest = json.load(open(ROOT / "external_mnistfc_matrix_manifest.json"))
    anchors = manifest["anchors"]

    # broad-SAT anchors (R1@0.05 SAT) from the matrix results -> run these first
    mres = json.load(open(ROOT / "external_mnistfc_matrix_results.json"))["results"]
    bsat = {r["anchor_idx"] for r in mres
            if r["region"] == "R1_pixel_ball" and r["eps"] == 0.05 and r["status"] == "sat"}
    ordered = ([a for a in anchors if a["anchor_idx"] in bsat] +
               [a for a in anchors if a["anchor_idx"] not in bsat])

    # output-index validation (compose_pca_two graph is basis-independent for indexing;
    # validate the exact path, abort on failure) -- Rule: validate before bounded queries.
    val = {"output_indexing": M.auto_indexing_validation(anchors, Xtr, ytr, Xte),
           "basis_validation": {"worst_orthonormality_error": 0.0, "worst_zero_mean_error": 0.0,
                                "n_bases": 0, "within_1e-9": True},
           "witness_checks": []}
    save({"provenance": {"script": "external_mnistfc_random_subspace.py", "timestamp_utc": now()},
          "validation": val}, VAL)
    print("INDEXING VALIDATION:", json.dumps(val["output_indexing"]), flush=True)
    if not val["output_indexing"]["PASS"]:
        print("ABORT: output-indexing validation failed.", flush=True)
        sys.exit(2)

    records = []
    bv = val["basis_validation"]
    t0 = time.time()
    for a in ordered:
        i, c, t = a["anchor_idx"], a["true_class"], a["target_class"]
        x0 = Xte[i]
        _, sig = M.pca_basis(x0, Xtr, ytr, c)   # SAME local-PCA sigma scale
        is_bsat = i in bsat
        for seed in seeds:
            V = random_zero_mean_basis(seed, i, B.D, 784)
            oe, ze = basis_checks(V)
            bv["worst_orthonormality_error"] = max(bv["worst_orthonormality_error"], oe)
            bv["worst_zero_mean_error"] = max(bv["worst_zero_mean_error"], ze)
            bv["n_bases"] += 1
            if oe > 1e-9 or ze > 1e-9:
                bv["within_1e-9"] = False
            m = B.compose_pca_two(V, x0, f"rand_a{i}_s{seed}")
            for alpha in alphas:
                vp = ROOT / "_rand_r4.vnnlib"
                B.emit_bounded(-alpha * sig, alpha * sig, c, t, vp)
                st, dt, wit = M.cli(m, vp, want_witness=True)
                vp.unlink(missing_ok=True)
                rec = {"anchor_idx": i, "true_class": c, "target_class": t,
                       "region": "R4_random_bounded", "seed": seed, "alpha": alpha,
                       "broad_sat_anchor": bool(is_bsat), "status": st, "runtime_s": dt,
                       "basis_ortho_err": round(oe, 12), "basis_zero_mean_err": round(ze, 12)}
                if st == "sat" and wit and len(wit) >= B.D:
                    z = [wit[j] for j in range(B.D)]
                    rec["witness_z"] = z
                    wc = M.witness_checks(z, x0, V, sig, alpha, c, t, m)
                    rec["witness_checks"] = wc
                    val["witness_checks"].append({"anchor_idx": i, "seed": seed, "alpha": alpha, **wc})
                records.append(rec)
                save({"provenance": {"script": "external_mnistfc_random_subspace.py", "timestamp_utc": now(),
                                     "n_anchors": len(anchors), "alphas": alphas, "seeds": seeds,
                                     "control": "zero-mean orthonormal random 8D basis; local-PCA sigma reused"},
                      "results": records}, OUT)
                save({"provenance": {"script": "external_mnistfc_random_subspace.py"}, "validation": val}, VAL)
                print(f"  R4rand a{i} s{seed} @{alpha}s {'[bSAT]' if is_bsat else ''}: {st} ({dt:.0f}s)", flush=True)
    print(f"DONE random-subspace mnist_fc ({len(records)} records, {time.time()-t0:.0f}s); "
          f"basis worst_ortho={bv['worst_orthonormality_error']:.2e}", flush=True)


if __name__ == "__main__":
    main()
