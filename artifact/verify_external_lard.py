#!/usr/bin/env python3.11
"""
Region-tagged verification of the external LARD-V2 runway pitch classifier (maraboupy API).

The Marabou *CLI* VNN-LIB parser does not accept weighted sums `(+ (* c X) ...)`, which the
valid-pixel (R4-bounded) region needs, so we drive Marabou through maraboupy, which supports
arbitrary linear constraints via addInequality. All regions are calibrated on TRAIN only;
anchors are held-out TEST and fixed in external_lard_manifest.json.

Regions (a-priori parameter grids; no tuning after results):
  R1  broad per-pixel Linf ball  [x0-eps, x0+eps] ∩ [0,1],  eps ∈ {0.1,0.2,0.3,0.4}
  R2  R1 + class mean-brightness band  sum(x) ∈ N*(mu_c ± 2 sigma_c)
  R5  label-conditioned per-pixel box over K=100 same-class train nearest neighbours ∩ [0,1]
  R4u local-PCA box (d=8) around anchor, |z_k| ≤ alpha*sigma_k,     alpha ∈ {1,2,3}
  R4b R4u + valid-pixel constraints 0 ≤ (x0 + V^T z)_i ≤ 1  (weighted-sum linear constraints)

Property: targeted class-confusion  Y_target >= Y_true  (SAT => a confusion witness exists).
"""
import json, os, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from maraboupy import Marabou
from external_lard_common import (ROOT, load_dataset, load_model, labels_split,
                                  compose_pca_nnet)

BASE_NNET = os.path.join(ROOT, "external_lard_mlp.nnet")
COMPOSED = os.path.join(ROOT, "external_lard_composed_models")
TIMEOUT = 120
EPS_GRID = [0.1, 0.2, 0.3, 0.4]
ALPHAS = [1.0, 2.0, 3.0]
MEAN_SIGMA = 2.0
K_R5 = 100
PCA_D = 8
PCA_NEIGH = 200
OPT = Marabou.createOptions(timeoutInSeconds=TIMEOUT, snc=True, numWorkers=4, verbosity=0)


def solve(net, c, t, n_in):
    """Add property Y_t >= Y_c and solve; return (status, ordered_input_witness|None, runtime)."""
    iv = net.inputVars[0].flatten(); ov = net.outputVars[0].flatten()
    net.addInequality([int(ov[c]), int(ov[t])], [1.0, -1.0], 0.0)   # Y_c - Y_t <= 0
    t0 = time.time()
    code, vals, _ = net.solve(options=OPT, verbose=False)
    dt = round(time.time() - t0, 1)
    s = str(code).lower()
    status = "sat" if s == "sat" else ("unsat" if s == "unsat" else "timeout" if "time" in s else "error")
    wit = None
    if status == "sat":
        wit = [round(float(vals[int(iv[j])]), 6) for j in range(n_in)]
    return status, wit, dt


def base_box(lo, hi):
    net = Marabou.read_nnet(BASE_NNET); iv = net.inputVars[0].flatten()
    for j, v in enumerate(iv):
        net.setLowerBound(int(v), float(lo[j])); net.setUpperBound(int(v), float(hi[j]))
    return net


def main():
    os.makedirs(COMPOSED, exist_ok=True)
    d = load_dataset(); Ws, bs, q1, q2, H, W = load_model()
    Xtr = d["Xtr"]; Xte = d["Xte"]; ytr = labels_split(d, "tr"); dpix = Xtr.shape[1]
    mean_tr = Xtr.mean(axis=1)
    band = {c: (float(mean_tr[ytr == c].mean()), float(mean_tr[ytr == c].std())) for c in range(3)}

    manifest = json.load(open(os.path.join(ROOT, "external_lard_manifest.json")))
    results = []
    for a in manifest["anchors"]:
        i, c, t = a["anchor_test_idx"], a["true_class"], a["target_class"]
        x0 = Xte[i].astype(np.float64); tag = f"a{i}"

        # R1 + R2 over eps grid
        for eps in EPS_GRID:
            lo = np.clip(x0 - eps, 0, 1); hi = np.clip(x0 + eps, 0, 1)
            s, w, dt = solve(base_box(lo, hi), c, t, dpix)
            results.append({"anchor_test_idx": i, "true_class": c, "target_class": t,
                            "region": "R1_pixel_ball", "eps": eps, "status": s, "runtime_s": dt,
                            "witness": w})
            mu, sg = band[c]
            net = base_box(lo, hi); iv = net.inputVars[0].flatten()
            net.addInequality([int(v) for v in iv], [1.0] * dpix, dpix * (mu + MEAN_SIGMA * sg))
            net.addInequality([int(v) for v in iv], [-1.0] * dpix, -dpix * (mu - MEAN_SIGMA * sg))
            s, w, dt = solve(net, c, t, dpix)
            results.append({"anchor_test_idx": i, "true_class": c, "target_class": t,
                            "region": "R2_mean_band", "eps": eps, "sigma": MEAN_SIGMA,
                            "status": s, "runtime_s": dt, "witness": w})

        # R5 label-conditioned KNN per-pixel box
        same = np.where(ytr == c)[0]
        dists = np.linalg.norm(Xtr[same] - x0, axis=1)
        nn = same[np.argsort(dists)[:K_R5]]
        loK = np.clip(Xtr[nn].min(0), 0, 1).astype(np.float64)
        hiK = np.clip(Xtr[nn].max(0), 0, 1).astype(np.float64)
        s, w, dt = solve(base_box(loK, hiK), c, t, dpix)
        results.append({"anchor_test_idx": i, "true_class": c, "target_class": t,
                        "region": "R5_knn_box", "K": K_R5, "status": s, "runtime_s": dt,
                        "witness": w,
                        "median_halfwidth": float(np.median((hiK - loK) / 2.0))})

        # R4 local-PCA (composed) over alpha grid
        nbr = same[np.argsort(dists)[:PCA_NEIGH]]
        Dmat = Xtr[nbr] - x0
        _, S, Vt = np.linalg.svd(Dmat, full_matrices=False)
        V = Vt[:PCA_D]
        sigma = (S[:PCA_D] / np.sqrt(max(1, len(nbr) - 1))).astype(np.float64)
        comp = os.path.join(COMPOSED, f"{tag}_R4.nnet")
        compose_pca_nnet(comp, Ws, bs, V, x0, PCA_D, 3)
        np.savez(os.path.join(COMPOSED, f"{tag}_pca.npz"), V=V, x0=x0, sigma=sigma)
        for alpha in ALPHAS:
            net = Marabou.read_nnet(comp); iv = net.inputVars[0].flatten()
            for k, v in enumerate(iv):
                net.setLowerBound(int(v), -alpha * sigma[k]); net.setUpperBound(int(v), alpha * sigma[k])
            s, w, dt = solve(net, c, t, PCA_D)
            results.append({"anchor_test_idx": i, "true_class": c, "target_class": t,
                            "region": "R4_local_pca_unbounded", "alpha": alpha,
                            "status": s, "runtime_s": dt, "witness_z": w})
            # bounded: valid-pixel weighted-sum constraints (prune provably-valid pixels)
            net = Marabou.read_nnet(comp); iv = [int(v) for v in net.inputVars[0].flatten()]
            for k, v in enumerate(iv):
                net.setLowerBound(v, -alpha * sigma[k]); net.setUpperBound(v, alpha * sigma[k])
            box = alpha * sigma; maxdev = np.abs(V).T @ box; n_valid = 0
            for pix in range(dpix):
                if x0[pix] - maxdev[pix] >= -1e-9 and x0[pix] + maxdev[pix] <= 1.0 + 1e-9:
                    continue
                net.addInequality(iv, list(V[:, pix]), 1.0 - x0[pix])    # sum <= 1 - x0
                net.addInequality(iv, list(-V[:, pix]), x0[pix])         # -sum <= x0  => sum >= -x0
                n_valid += 1
            s, w, dt = solve(net, c, t, PCA_D)
            results.append({"anchor_test_idx": i, "true_class": c, "target_class": t,
                            "region": "R4_local_pca_bounded", "alpha": alpha,
                            "n_valid_constraints": n_valid, "status": s, "runtime_s": dt, "witness_z": w})
        print(f"anchor {i} (true {c} -> target {t}) done", flush=True)

    out = {"provenance": {"script": "verify_external_lard.py", "solver": "maraboupy Marabou 2.0, snc, 4 workers",
                          "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          "n_anchors": manifest["n_anchors"], "timeout_s": TIMEOUT,
                          "eps_grid": EPS_GRID, "alphas": ALPHAS},
           "results": results}
    json.dump(out, open(os.path.join(ROOT, "external_lard_results.json"), "w"), indent=2)
    print("WROTE external_lard_results.json (", len(results), "records )")


if __name__ == "__main__":
    main()
