#!/usr/bin/env python3
"""Parallel driver for the mnist_fc matched random-subspace control (for jarvis:
many cores, independent Marabou solves). Same validated compose/emit/cli path as
external_mnistfc_random_subspace.py; only the orchestration is parallel.

Order: broad-SAT anchors and 1 sigma first (the critical comparison), then the rest.
Output: external_mnistfc_random_subspace_results.json
        external_mnistfc_random_subspace_validation.json
"""
from __future__ import annotations
import argparse, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import external_mnistfc_matrix as M
import verify_external_mnistfc_bounded_transfer as B

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
    ap.add_argument("--alphas", default="1.0,2.0,3.0")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    Xtr, ytr, Xte, yte = B.load_mnist()
    manifest = json.load(open(ROOT / "external_mnistfc_matrix_manifest.json"))
    anchors = manifest["anchors"]
    mres = json.load(open(ROOT / "external_mnistfc_matrix_results.json"))["results"]
    bsat = {r["anchor_idx"] for r in mres
            if r["region"] == "R1_pixel_ball" and r["eps"] == 0.05 and r["status"] == "sat"}
    ordered = ([a for a in anchors if a["anchor_idx"] in bsat] +
               [a for a in anchors if a["anchor_idx"] not in bsat])

    # indexing validation once (abort on failure)
    val = {"output_indexing": M.auto_indexing_validation(anchors, Xtr, ytr, Xte),
           "basis_validation": {"worst_orthonormality_error": 0.0, "worst_zero_mean_error": 0.0,
                                "n_bases": 0, "within_1e-9": True},
           "witness_checks": []}
    print("INDEXING VALIDATION:", json.dumps(val["output_indexing"]), flush=True)
    if not val["output_indexing"]["PASS"]:
        print("ABORT: indexing failed"); sys.exit(2)

    # pre-compose per (anchor, seed): sigma (local-PCA), random V, composed two-output net
    prep = {}
    bv = val["basis_validation"]
    for a in ordered:
        i, c, t = a["anchor_idx"], a["true_class"], a["target_class"]
        x0 = Xte[i]
        _, sig = M.pca_basis(x0, Xtr, ytr, c)   # SAME local-PCA sigma scale
        for seed in seeds:
            V = random_zero_mean_basis(seed, i, B.D, 784)
            oe, ze = basis_checks(V)
            bv["worst_orthonormality_error"] = max(bv["worst_orthonormality_error"], oe)
            bv["worst_zero_mean_error"] = max(bv["worst_zero_mean_error"], ze)
            bv["n_bases"] += 1
            if oe > 1e-9 or ze > 1e-9:
                bv["within_1e-9"] = False
            m = B.compose_pca_two(V, x0, f"rand_a{i}_s{seed}")
            prep[(i, seed)] = (x0, sig, V, m, c, t, i in bsat, oe, ze)
    print(f"composed {len(prep)} nets; launching solves ({args.workers} workers)", flush=True)

    # tasks: broad-SAT first, alpha ascending (1 sigma first)
    tasks = []
    for alpha in sorted(alphas):
        for a in ordered:
            for seed in seeds:
                tasks.append((a["anchor_idx"], seed, alpha))

    def work(task):
        i, seed, alpha = task
        x0, sig, V, m, c, t, isb, oe, ze = prep[(i, seed)]
        vp = ROOT / f"_rand_{i}_{seed}_{alpha}.vnnlib"
        B.emit_bounded(-alpha * sig, alpha * sig, c, t, vp)
        st, dt, wit = M.cli(m, vp, want_witness=True)
        vp.unlink(missing_ok=True)
        rec = {"anchor_idx": i, "true_class": c, "target_class": t,
               "region": "R4_random_bounded", "seed": seed, "alpha": alpha,
               "broad_sat_anchor": bool(isb), "status": st, "runtime_s": dt,
               "basis_ortho_err": round(oe, 12), "basis_zero_mean_err": round(ze, 12)}
        if st == "sat" and wit and len(wit) >= B.D:
            z = [wit[j] for j in range(B.D)]
            rec["witness_z"] = z
            rec["witness_checks"] = M.witness_checks(z, x0, V, sig, alpha, c, t, m)
        return rec

    records = []
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, tk): tk for tk in tasks}
        for fut in as_completed(futs):
            rec = fut.result()
            records.append(rec)
            if "witness_checks" in rec:
                val["witness_checks"].append({"anchor_idx": rec["anchor_idx"], "seed": rec["seed"],
                                              "alpha": rec["alpha"], **rec["witness_checks"]})
            done += 1
            if done % 10 == 0 or done == len(tasks):
                save({"provenance": {"script": "run_mnistfc_random_parallel.py", "timestamp_utc": now(),
                                     "n_anchors": len(anchors), "alphas": alphas, "seeds": seeds,
                                     "control": "zero-mean orthonormal random 8D basis; local-PCA sigma reused"},
                      "results": records}, OUT)
                save({"provenance": {"script": "run_mnistfc_random_parallel.py"}, "validation": val}, VAL)
                print(f"  {done}/{len(tasks)} done ({time.time()-t0:.0f}s)", flush=True)
    save({"provenance": {"script": "run_mnistfc_random_parallel.py", "timestamp_utc": now(),
                         "n_anchors": len(anchors), "alphas": alphas, "seeds": seeds},
          "results": records}, OUT)
    save({"provenance": {"script": "run_mnistfc_random_parallel.py"}, "validation": val}, VAL)
    print(f"DONE {len(records)} records, {time.time()-t0:.0f}s; basis worst_ortho={bv['worst_orthonormality_error']:.2e}", flush=True)


if __name__ == "__main__":
    main()
