#!/usr/bin/env python3.11
"""Matched-dimension control for the TinyTaxiNet P1a/P1b AE decoder-box result.

The paper's R3 result: under the AE decoder-box region (an 8D latent box mapped
through the trained decoder, then TinyTaxiNet), P1a (CTE>=10) and P1b (CTE<=-10)
flip from SAT (pixel box) to UNSAT. This control asks whether that flip needs the
trained decoder's nonlinear support model, or is a generic 8D dimensionality
effect.

Control regions (LINEAR 8D affine, matched to the centered subset the AE uses):
  x = mu + V^T z,  z in [zmin, zmax]  (per-direction min/max of centered
  projections -- the linear analogue of the AE's centered latent min/max box),
  intersected with 0 <= x <= 1.
    - PCA : V = top-8 principal directions of centered image deviations.
    - RAND: V = deterministic zero-mean orthonormal random 8D (seeds 0,1,2).
mu is the centered-subset mean (mean 0.5), so mean(x)=0.5 holds automatically
(comparable to the AE 'with mean' variant). d=8 matches the AE latent dim.

Property: P1a CTE>=10, P1b CTE<=-10 on centered anchors/subset. Exact status
parsing. Output: verification_p1_random_affine_control.json.
"""
import itertools
import json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import h5py
from maraboupy import Marabou

NNET = "VerifyGAN/models/TinyTaxiNet.nnet"
DATA = "VerifyGAN/data/SK_DownsampledGANFocusAreaData.h5"
NUM_PIXELS = 128
D = 8
SEEDS = [0, 1, 2]
TIMEOUT = 600


def load_nnet_weights(path=NNET):
    """Minimal .nnet reader for validation-only forward evaluations."""
    lines = open(path).read().splitlines()
    i = 0
    while lines[i].startswith("//"):
        i += 1
    nl = int(lines[i].split(",")[0]); i += 1
    sizes = [int(x) for x in lines[i].split(",") if x.strip()]; i += 1
    i += 5
    Ws, bs = [], []
    for layer in range(nl):
        rows, cols = sizes[layer + 1], sizes[layer]
        W = np.zeros((rows, cols))
        for r in range(rows):
            W[r] = [float(x) for x in lines[i].strip().rstrip(",").split(",")][:cols]
            i += 1
        b = np.zeros(rows)
        for r in range(rows):
            b[r] = float(lines[i].strip().rstrip(","))
            i += 1
        Ws.append(W); bs.append(b)
    return Ws, bs


def forward_nnet(x, Ws, bs):
    h = np.asarray(x, dtype=np.float64)
    for layer, (W, b) in enumerate(zip(Ws, bs)):
        h = h @ W.T + b
        if layer < len(Ws) - 1:
            h = np.maximum(h, 0.0)
    return h


def load():
    with h5py.File(DATA, "r") as f:
        return f["y_train"][:].reshape(-1, NUM_PIXELS).astype(np.float64), f["X_train"][:].astype(np.float64)


def random_zero_mean_basis(seed, d, n):
    rng = np.random.default_rng(int(seed))
    cols = []
    for _ in range(d):
        v = rng.normal(size=n); v = v - v.mean()
        for q in cols:
            v = v - np.dot(v, q) * q
        cols.append(v / np.linalg.norm(v))
    return np.vstack(cols)


def solve_affine(mu, V, zmin, zmax, kind):
    """Compose x = mu + V^T z into TinyTaxiNet; z in [zmin,zmax]; x in [0,1]."""
    net = Marabou.read_nnet(NNET)
    iv = [int(v) for v in net.inputVars[0].flatten()]
    d = V.shape[0]
    z = [int(net.getNewVariable()) for _ in range(d)]
    for j in range(d):
        net.setLowerBound(z[j], float(zmin[j])); net.setUpperBound(z[j], float(zmax[j]))
    for i in range(NUM_PIXELS):
        col = V[:, i]; nz = np.where(np.abs(col) > 1e-12)[0]
        net.addEquality([iv[i]] + [z[j] for j in nz], [1.0] + [float(-col[j]) for j in nz], float(mu[i]))
        net.setLowerBound(iv[i], 0.0); net.setUpperBound(iv[i], 1.0)
    if kind == "P1a":
        net.setLowerBound(192, 10.0)
    else:                       # P1b
        net.setUpperBound(192, -10.0)
    opt = Marabou.createOptions(verbosity=0, timeoutInSeconds=TIMEOUT)
    t0 = time.time()
    code, vals, _ = net.solve(verbose=False, options=opt)
    dt = round(time.time() - t0, 2)
    s = str(code).strip().lower()
    status = "sat" if s == "sat" else ("unsat" if s == "unsat" else "timeout" if "time" in s else "error")
    ce = None
    if status == "sat":
        ce = {"CTE": round(float(vals[192]), 4), "HE": round(float(vals[193]), 4)}
    return {"status": status, "time_s": dt, "counterexample": ce}


def validation_search(mu, V, zmin, zmax, Ws, bs, seed=0, n_random=5000):
    """Corner plus random feasible-z search used only as a non-degeneracy check.

    This is not an optimizer and does not replace the exact Marabou UNSAT proof;
    it records reachable CTE magnitudes found inside the affine region.
    """
    zs = [np.where(bits, zmax, zmin) for bits in itertools.product([0, 1], repeat=len(zmin))]
    rng = np.random.default_rng(seed)
    zs.extend(rng.uniform(zmin, zmax, size=(n_random, len(zmin))))
    zs.append(np.zeros(len(zmin)))
    vals = []
    for z in zs:
        x = mu + V.T @ z
        if x.min() >= -1e-9 and x.max() <= 1.0 + 1e-9:
            vals.append(float(forward_nnet(x, Ws, bs)[0]))
    return {
        "method": f"all {2 ** len(zmin)} z-box corners plus {n_random} random feasible samples; not exact optimization",
        "valid_samples": len(vals),
        "max_pos_CTE": round(max(vals), 3),
        "max_neg_CTE": round(abs(min(vals)), 3),
    }


def main():
    images, labels = load()
    Ws, bs = load_nnet_weights()
    cen = np.abs(labels[:, 0]) < 1.0
    X = images[cen]
    mu = X.mean(0)
    dev = X - mu
    _, S, Vt = np.linalg.svd(dev, full_matrices=False)
    Vpca = Vt[:D]

    out = {"experiment": "p1_matched_dimension_control", "d": D, "n_centered": int(cen.sum()),
           "timeout_s": TIMEOUT,
           "baseline": {"pixel_box_P1a": "sat", "pixel_box_P1b": "sat",
                        "AE_decoder_box_P1a": "unsat", "AE_decoder_box_P1b": "unsat",
                        "source": "verification_ae_with_mean_constraint.json (ae_with_mean); pixel box from verification_autoencoder_composed.json"},
           "control": ("linear 8D affine region x=mu+V^T z over the centered projection min/max box, x in [0,1]; "
                       "mu is centered mean so mean(x)=0.5; d=8 matches AE latent dim"),
           "regions": {}}

    bases = [("pca", Vpca, None)]
    for seed in SEEDS:
        bases.append(("random", random_zero_mean_basis(seed, D, NUM_PIXELS), seed))

    for name, V, seed in bases:
        proj = dev @ V.T
        zmin, zmax = proj.min(0), proj.max(0)
        key = f"{name}" + (f"_seed{seed}" if seed is not None else "")
        # basis validation
        ortho = float(np.max(np.abs(V @ V.T - np.eye(D))))
        zmean = float(np.max(np.abs(V.mean(1))))
        rec = {"basis_ortho_err": round(ortho, 12), "basis_zero_mean_err": round(zmean, 12),
               "z_box": [ [round(float(a),4),round(float(b),4)] for a,b in zip(zmin,zmax) ]}
        for kind in ("P1a", "P1b"):
            r = solve_affine(mu, V, zmin, zmax, kind)
            rec[kind] = r
            print(f"{key} {kind}: {r['status']} ({r['time_s']}s)", flush=True)
        rec["validation_reached_CTE"] = validation_search(mu, V, zmin, zmax, Ws, bs)
        out["regions"][key] = rec
        json.dump(out, open("verification_p1_random_affine_control.json", "w"), indent=2)

    # interpretation
    def st(k, p): return out["regions"][k][p]["status"]
    rnd_flip = all(st(f"random_seed{s}", p) == "unsat" for s in SEEDS for p in ("P1a", "P1b"))
    pca_flip = all(st("pca", p) == "unsat" for p in ("P1a", "P1b"))
    out["interpretation"] = {
        "pca_affine_flips_to_unsat": pca_flip,
        "random_affine_flips_to_unsat": rnd_flip,
        "reading": ("If random 8D affine also flips P1a/P1b to UNSAT, the AE flip is (at least partly) a generic "
                    "low-dimensional-region effect; if random stays SAT while AE/PCA flip, the flip needs a "
                    "data-aligned or decoder-support model, not mere 8D dimensionality."),
    }
    out["validation_note"] = (
        "The validation_reached_CTE fields are a reproducible corner/random search used to show the affine "
        "regions are non-degenerate. They are not exact optima; the exact statements are the Marabou UNSAT "
        "verdicts for CTE >= 10 and CTE <= -10.")
    json.dump(out, open("verification_p1_random_affine_control.json", "w"), indent=2)
    print("SAVED verification_p1_random_affine_control.json")


if __name__ == "__main__":
    main()
