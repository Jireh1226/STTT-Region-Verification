#!/usr/bin/env python3
"""TinyTaxiNet P3 dimension sweep: verification rate as a function of subspace
dimension d, for PCA vs matched random subspaces, on the centered P3 anchors.

For each anchor: build the local-PCA basis once up to d_max from the same-state
neighbour deviations (same neighbour bins, mean invariant, clipping, thresholds,
solver as verify_p3_local_pca.py). For each d in DIMS: verify P3 (all four
sub-queries UNSAT) under (i) the top-d PCA subspace and (ii) matched random
zero-mean orthonormal d-subspaces (seeds), both at the same per-direction sigma
scale and alpha. Chunk-aware (--start/--count over the ordered centered indices)
for parallel processes.

Output: <--output> (incremental).
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import verify_p3_local_pca as p3

SUBQ = [("a", 192, ">=", p3.P3_CTE_BOUND), ("b", 192, "<=", -p3.P3_CTE_BOUND),
        ("c", 193, ">=", p3.P3_HE_BOUND), ("d", 193, "<=", -p3.P3_HE_BOUND)]


def full_pca(images, labels, idx, dmax):
    cte, he = labels[:, 0], labels[:, 1]
    x0 = images[int(idx)]; cq, hq = cte[int(idx)], he[int(idx)]
    mask = (np.abs(cte - cq) <= p3.CTE_BIN_HALF) & (np.abs(he - hq) <= p3.HE_BIN_HALF)
    mask[int(idx)] = False
    nb = images[mask]; m = len(nb)
    if m < dmax + 1:
        return None
    Dm = nb - x0
    _, S, Vt = np.linalg.svd(Dm, full_matrices=False)
    Vfull = Vt[:dmax]
    sig = (Dm @ Vfull.T).std(0)
    return x0, Vfull, sig, int(m)


def random_basis(seed, idx, d, n=128):
    rng = np.random.default_rng(int(seed) * 1_000_003 + int(idx) * 97 + d)
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


def verify_region(x0, V, sigma, alpha):
    alphas = alpha * sigma
    statuses = [p3.solve_one(x0, V, None, alphas, var, direc, b)["status"] for _, var, direc, b in SUBQ]
    if "region_empty_after_clip" in statuses:
        return "region_empty"
    if "sat" in statuses:
        return "sat"
    if "timeout" in statuses:
        return "timeout"
    if all(s == "unsat" for s in statuses):
        return "verified"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="2,4,8,16,32,64")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=-1)   # -1 = to end
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    dims = [int(x) for x in args.dims.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    dmax = max(dims)

    images, labels = p3.load_data()
    cen = p3.centered_indices(labels)
    end = len(cen) if args.count < 0 else min(len(cen), args.start + args.count)
    chunk = cen[args.start:end]

    records = []
    t0 = time.time()
    for k, idx in enumerate(chunk):
        pc = full_pca(images, labels, idx, dmax)
        if pc is None:
            records.append({"anchor_idx": int(idx), "empty_neighbor": True})
            continue
        x0, Vfull, sig, m = pc
        for d in dims:
            cat = verify_region(x0, Vfull[:d], sig[:d], args.alpha)
            records.append({"anchor_idx": int(idx), "d": d, "basis": "pca", "seed": None,
                            "alpha": args.alpha, "n_neighbors": m, "category": cat})
            for seed in seeds:
                Vr = random_basis(seed, idx, d)
                cat = verify_region(x0, Vr, sig[:d], args.alpha)
                records.append({"anchor_idx": int(idx), "d": d, "basis": "random", "seed": seed,
                                "alpha": args.alpha, "n_neighbors": m, "category": cat})
        if (k + 1) % 5 == 0 or k + 1 == len(chunk):
            Path(args.output).write_text(json.dumps(
                {"provenance": {"script": "verify_p3_dimension_sweep.py", "dims": dims, "seeds": seeds,
                                "alpha": args.alpha, "start": args.start, "count": len(chunk)},
                 "results": records}, indent=2) + "\n")
            print(f"  {k+1}/{len(chunk)} anchors ({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE chunk start={args.start} n={len(chunk)} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
