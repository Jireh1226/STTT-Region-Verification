#!/usr/bin/env python3.11
"""Dimension sweep for the LARD bounded local-PCA vs matched random subspace.

For d in {2,4,8,16,32,64}, PCA basis (top-d singular vectors) and matched random
zero-mean orthonormal d-basis (seeds 0,1,2), at the same per-direction sigma scale
(top-d singular values / sqrt(n-1)) and the same valid-pixel bounded encoding, on
the fixed 50 LARD anchors at alpha=1 sigma. Records UNSAT/SAT/timeout so we can plot
verification (witness-removal) rate as a function of subspace dimension for the two
basis families, and read off the practical d-threshold where the subspace rate
converges to the full-region rate.

Output: lard_dimension_sweep_results.json (incremental).
"""
import json, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from maraboupy import Marabou
from external_lard_common import ROOT, load_dataset, load_model, labels_split, compose_pca_nnet

TIMEOUT = 120
DIMS = [2, 4, 8, 16, 32, 64]
SEEDS = [0, 1, 2]
ALPHA = 1.0
PCA_NEIGH = 200
OPT = Marabou.createOptions(timeoutInSeconds=TIMEOUT, snc=True, numWorkers=4, verbosity=0)
COMPOSED = os.path.join(ROOT, "external_lard_composed_models")
OUT = os.path.join(ROOT, "lard_dimension_sweep_results.json")


def random_zero_mean_basis(seed, anchor_idx, d, n):
    rng = np.random.default_rng(int(seed) * 1_000_003 + int(anchor_idx) * 97 + d)
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


def solve(net, c, t, n_in):
    iv = net.inputVars[0].flatten(); ov = net.outputVars[0].flatten()
    net.addInequality([int(ov[c]), int(ov[t])], [1.0, -1.0], 0.0)
    t0 = time.time()
    code, vals, _ = net.solve(options=OPT, verbose=False)
    dt = round(time.time() - t0, 2)
    s = str(code).lower()
    status = "sat" if s == "sat" else ("unsat" if s == "unsat" else "timeout" if "time" in s else "error")
    return status, dt


def solve_bounded(Ws, bs, V, x0, sigma, c, t, alpha, tag):
    d = V.shape[0]; dpix = len(x0)
    comp = os.path.join(COMPOSED, f"{tag}.nnet")
    compose_pca_nnet(comp, Ws, bs, V, x0, d, 3)
    net = Marabou.read_nnet(comp); iv = [int(v) for v in net.inputVars[0].flatten()]
    for k, v in enumerate(iv):
        net.setLowerBound(v, -alpha * sigma[k]); net.setUpperBound(v, alpha * sigma[k])
    box = alpha * sigma; maxdev = np.abs(V).T @ box
    for pix in range(dpix):
        if x0[pix] - maxdev[pix] >= -1e-9 and x0[pix] + maxdev[pix] <= 1.0 + 1e-9:
            continue
        net.addInequality(iv, list(V[:, pix]), 1.0 - x0[pix])
        net.addInequality(iv, list(-V[:, pix]), x0[pix])
    st, dt = solve(net, c, t, d)
    os.remove(comp)
    return st, dt


def main():
    d0 = load_dataset(); Ws, bs, q1, q2, H, W = load_model()
    Xtr = d0["Xtr"]; Xte = d0["Xte"]; ytr = labels_split(d0, "tr"); dpix = Xtr.shape[1]
    manifest = json.load(open(os.path.join(ROOT, "external_lard_manifest.json")))
    dmax = max(DIMS)

    results = []
    t_start = time.time()
    for a in manifest["anchors"]:
        i, c, t = a["anchor_test_idx"], a["true_class"], a["target_class"]
        x0 = Xte[i].astype(np.float64)
        same = np.where(ytr == c)[0]
        dists = np.linalg.norm(Xtr[same] - x0, axis=1)
        nbr = same[np.argsort(dists)[:PCA_NEIGH]]
        Dmat = Xtr[nbr] - x0
        _, S, Vt = np.linalg.svd(Dmat, full_matrices=False)
        Vfull = Vt[:dmax]
        sig_full = (S[:dmax] / np.sqrt(max(1, len(nbr) - 1))).astype(np.float64)
        for d in DIMS:
            sigma = sig_full[:d]
            # PCA-d
            st, dt = solve_bounded(Ws, bs, Vfull[:d], x0, sigma, c, t, ALPHA, f"dsw_a{i}_pca_d{d}")
            results.append({"anchor_test_idx": i, "true_class": c, "target_class": t,
                            "d": d, "basis": "pca", "seed": None, "alpha": ALPHA,
                            "status": st, "runtime_s": dt})
            # random-d (3 seeds)
            for seed in SEEDS:
                Vr = random_zero_mean_basis(seed, i, d, dpix)
                st, dt = solve_bounded(Ws, bs, Vr, x0, sigma, c, t, ALPHA, f"dsw_a{i}_rnd{seed}_d{d}")
                results.append({"anchor_test_idx": i, "true_class": c, "target_class": t,
                                "d": d, "basis": "random", "seed": seed, "alpha": ALPHA,
                                "status": st, "runtime_s": dt})
        json.dump({"provenance": {"script": "lard_dimension_sweep.py", "dims": DIMS, "seeds": SEEDS,
                                  "alpha": ALPHA, "pca_neigh": PCA_NEIGH, "n_anchors": manifest["n_anchors"],
                                  "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                   "results": results}, open(OUT, "w"), indent=2)
        print(f"anchor {i} done [{time.time()-t_start:.0f}s]", flush=True)
    print(f"WROTE {OUT} ({len(results)} records, {time.time()-t_start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
