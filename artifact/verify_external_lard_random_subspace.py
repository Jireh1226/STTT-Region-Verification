#!/usr/bin/env python3.11
"""Matched random-subspace control for the external LARD-V2 R4 local-PCA removals.

Same 50 held-out anchors, same trained LARD pitch MLP, same targeted class-confusion
property (Y_target >= Y_true), same solver settings, same valid-pixel (R4-bounded)
convention, and the SAME per-anchor local-PCA scale vector sigma_k (from the 200
same-class nearest-neighbour deviation SVD). The ONLY change: the 8 basis directions
are deterministic zero-mean orthonormal RANDOM directions instead of the PCA ones.

Question: do the broad-ball (R1@0.4) removals persist under a random 8D subspace?
If random 8D removes the same witnesses PCA does, the removal is a low-dimension /
valid-pixel effect, not evidence that the PCA orientation matters.

Runs R4 unbounded and R4 valid-pixel bounded at alpha in {1,2,3} sigma, seeds {0,1,2}.
Output: external_lard_random_subspace_results.json (incremental).
"""
import json, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from maraboupy import Marabou
from external_lard_common import ROOT, load_dataset, load_model, labels_split, compose_pca_nnet

TIMEOUT = 120
ALPHAS = [1.0, 2.0, 3.0]
SEEDS = [0, 1, 2]
PCA_D = 8
PCA_NEIGH = 200
OPT = Marabou.createOptions(timeoutInSeconds=TIMEOUT, snc=True, numWorkers=4, verbosity=0)
COMPOSED = os.path.join(ROOT, "external_lard_composed_models")
OUT = os.path.join(ROOT, "external_lard_random_subspace_results.json")


def random_zero_mean_basis(seed, anchor_idx, d, n):
    """d orthonormal zero-mean row vectors, deterministic in (seed, anchor_idx)."""
    rng = np.random.default_rng(int(seed) * 1_000_003 + int(anchor_idx))
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


def solve(net, c, t, n_in):
    iv = net.inputVars[0].flatten(); ov = net.outputVars[0].flatten()
    net.addInequality([int(ov[c]), int(ov[t])], [1.0, -1.0], 0.0)   # Y_c - Y_t <= 0  (SAT => confusion)
    t0 = time.time()
    code, vals, _ = net.solve(options=OPT, verbose=False)
    dt = round(time.time() - t0, 1)
    s = str(code).lower()
    status = "sat" if s == "sat" else ("unsat" if s == "unsat" else "timeout" if "time" in s else "error")
    wit = [round(float(vals[int(iv[j])]), 6) for j in range(n_in)] if status == "sat" else None
    return status, wit, dt


def main():
    d = load_dataset(); Ws, bs, q1, q2, H, W = load_model()
    Xtr = d["Xtr"]; Xte = d["Xte"]; ytr = labels_split(d, "tr"); dpix = Xtr.shape[1]
    manifest = json.load(open(os.path.join(ROOT, "external_lard_manifest.json")))

    results = []
    worst_ortho = 0.0; worst_zm = 0.0; n_bases = 0
    t_start = time.time()
    for a in manifest["anchors"]:
        i, c, t = a["anchor_test_idx"], a["true_class"], a["target_class"]
        x0 = Xte[i].astype(np.float64)
        # SAME local-PCA scale sigma_k from the 200 same-class nearest-neighbour deviations
        same = np.where(ytr == c)[0]
        dists = np.linalg.norm(Xtr[same] - x0, axis=1)
        nbr = same[np.argsort(dists)[:PCA_NEIGH]]
        Dmat = Xtr[nbr] - x0
        _, S, _ = np.linalg.svd(Dmat, full_matrices=False)
        sigma = (S[:PCA_D] / np.sqrt(max(1, len(nbr) - 1))).astype(np.float64)

        for seed in SEEDS:
            V = random_zero_mean_basis(seed, i, PCA_D, dpix)
            oe, ze = basis_checks(V)
            worst_ortho = max(worst_ortho, oe); worst_zm = max(worst_zm, ze); n_bases += 1
            comp = os.path.join(COMPOSED, f"a{i}_randseed{seed}_R4.nnet")
            compose_pca_nnet(comp, Ws, bs, V, x0, PCA_D, 3)
            for alpha in ALPHAS:
                # unbounded z-box
                net = Marabou.read_nnet(comp); iv = net.inputVars[0].flatten()
                for k, v in enumerate(iv):
                    net.setLowerBound(int(v), -alpha * sigma[k]); net.setUpperBound(int(v), alpha * sigma[k])
                s, w, dt = solve(net, c, t, PCA_D)
                results.append({"anchor_test_idx": i, "true_class": c, "target_class": t,
                                "region": "R4_random_unbounded", "seed": seed, "alpha": alpha,
                                "status": s, "runtime_s": dt, "witness_z": w,
                                "basis_ortho_err": round(oe, 12), "basis_zero_mean_err": round(ze, 12)})
                # bounded: valid-pixel weighted-sum constraints (same convention as verify_external_lard.py)
                net = Marabou.read_nnet(comp); iv = [int(v) for v in net.inputVars[0].flatten()]
                for k, v in enumerate(iv):
                    net.setLowerBound(v, -alpha * sigma[k]); net.setUpperBound(v, alpha * sigma[k])
                box = alpha * sigma; maxdev = np.abs(V).T @ box; n_valid = 0
                for pix in range(dpix):
                    if x0[pix] - maxdev[pix] >= -1e-9 and x0[pix] + maxdev[pix] <= 1.0 + 1e-9:
                        continue
                    net.addInequality(iv, list(V[:, pix]), 1.0 - x0[pix])
                    net.addInequality(iv, list(-V[:, pix]), x0[pix])
                    n_valid += 1
                s, w, dt = solve(net, c, t, PCA_D)
                results.append({"anchor_test_idx": i, "true_class": c, "target_class": t,
                                "region": "R4_random_bounded", "seed": seed, "alpha": alpha,
                                "n_valid_constraints": n_valid, "status": s, "runtime_s": dt, "witness_z": w})
            os.remove(comp)
        # incremental save after each anchor
        json.dump({"provenance": {"script": "verify_external_lard_random_subspace.py",
                                  "solver": "maraboupy Marabou 2.0, snc, 4 workers",
                                  "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                  "n_anchors": manifest["n_anchors"], "timeout_s": TIMEOUT,
                                  "alphas": ALPHAS, "seeds": SEEDS, "pca_d": PCA_D, "pca_neigh": PCA_NEIGH,
                                  "control": "zero-mean orthonormal random 8D basis; local-PCA sigma reused"},
                   "basis_validation": {"worst_orthonormality_error": worst_ortho,
                                        "worst_zero_mean_error": worst_zm, "n_bases": n_bases,
                                        "within_1e-9": bool(worst_ortho < 1e-9 and worst_zm < 1e-9)},
                   "results": results}, open(OUT, "w"), indent=2)
        print(f"anchor {i} ({c}->{t}) done  [{time.time()-t_start:.0f}s]", flush=True)

    print(f"WROTE {OUT} ({len(results)} records); "
          f"basis worst_ortho={worst_ortho:.2e} worst_zero_mean={worst_zm:.2e}", flush=True)


if __name__ == "__main__":
    main()
